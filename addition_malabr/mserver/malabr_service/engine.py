"""MALABR inference engine -- see design_doc/phase1_design.md sections 7-9b.

Built in the order the design mandates (most safety-critical first):
  1. SlotAllocator + wipe-on-acquire   <- isolation
  2. chat template application (6c)
  3. single-threaded engine loop
  4. compaction (8)
  5. scheduler (9a/9b)

Only step 1 is present so far.
"""

import threading

import llama_cpp.llama_cpp as C


class SlotWipeError(RuntimeError):
    """A KV slot could not be proven empty. The slot is quarantined, never reused."""


class SlotAllocator:
    """Hands out llama.cpp sequence ids (slots), guaranteeing each is empty.

    The isolation rule, stated once because it is the whole design: WIPE ON
    ACQUIRE, not only on release. A release can be skipped -- a bug, or a crash
    between release and reuse. An acquire cannot: a session cannot exist without
    passing through it. Putting the safety-critical step where it cannot be
    skipped turns a missed release into a harmless inefficiency instead of a
    cross-session data leak.

    Measured, not assumed: a slot written to and handed back without a wipe
    still reports llama_memory_seq_pos_max() == 19 for a 20-token conversation.
    The KV really does persist; this class is what stops the next session
    reading it.
    """

    def __init__(self, mem, n_seq_max):
        self._mem = mem
        # A lock is required, not defensive: connection threads create sessions
        # concurrently (one per tab, and tabs arrive together after a window
        # switch).
        #
        # MEASURED, and NOT the failure mode first assumed here. An earlier
        # version of this comment claimed an unlocked pop() hands two tabs the
        # SAME slot. That is false -- list.pop() is atomic under the GIL, and a
        # 32-thread x 20-trial probe produced zero duplicates with the lock
        # removed. The real race is the CHECK-THEN-ACT pair below: threads all
        # pass "if not self._free", then race to pop an emptied list. With the
        # window widened to 2ms, no-lock produced 120 IndexErrors across 160
        # acquires; with the lock, zero. So the damage is a crash in session
        # creation for a request that should have been cleanly rejected as
        # "no capacity" -- not a cross-session leak.
        self._lock = threading.Lock()
        self._free = list(range(n_seq_max))
        self._in_use = set()
        # Slots whose wipe could not be verified. Deliberately leaked rather
        # than reused: losing capacity is recoverable, leaking a conversation
        # into another origin's session is not.
        self._quarantined = set()

    def acquire(self):
        """Return an empty slot id, or None if at capacity.

        Raises SlotWipeError only if a slot could not be proven empty; the slot
        is dropped from circulation in that case rather than handed out dirty.
        """
        with self._lock:
            if not self._free:
                return None          # at capacity -- caller rejects the session
            slot = self._free.pop()
            self._in_use.add(slot)

        # Wipe OUTSIDE the lock: llama_memory_seq_rm touches the model context,
        # and holding the allocator lock across it would serialise every other
        # tab's acquire behind one wipe. The slot is already private to this
        # caller (it is out of _free and in _in_use), so no other thread can
        # observe it mid-wipe.
        try:
            self._wipe_and_verify(slot)
        except SlotWipeError:
            with self._lock:
                self._in_use.discard(slot)
                self._quarantined.add(slot)   # NOT returned to _free
            raise
        return slot

    def release(self, slot):
        """Return a slot to the pool, wiped again (belt and braces)."""
        with self._lock:
            if slot in self._quarantined:
                return                       # stays out of circulation, silently
            if slot not in self._in_use:
                # Double release, or a slot never acquired. Returning it to
                # _free would put one slot in the list twice, and two sessions
                # would then be handed the same KV sequence -- the exact leak
                # this class exists to prevent. Refuse instead.
                raise ValueError(f"release of slot {slot} that is not in use")
            self._in_use.discard(slot)

        try:
            self._wipe_and_verify(slot)
        except SlotWipeError:
            with self._lock:
                self._quarantined.add(slot)
            raise

        with self._lock:
            self._free.append(slot)

    def _wipe_and_verify(self, slot):
        """Clear a sequence and PROVE it is empty. Both checks are load-bearing."""
        # llama_memory_seq_rm is declared to return bool. On this model every
        # call returned True (verified: full, partial, and empty-slot removal),
        # so this branch is defensive rather than observed. It is checked
        # because llama.cpp returns false where partial removal is unsupported
        # -- the sliding-window-attention case, which is exactly what section 8
        # compaction will exercise on models like gemma-3.
        if not C.llama_memory_seq_rm(self._mem, slot, -1, -1):
            raise SlotWipeError(f"llama_memory_seq_rm returned false for slot {slot}")

        # Independent confirmation. Trusting the return code alone would mean
        # trusting one bool; this asks the memory itself whether anything is
        # left. -1 is llama.cpp's "no positions in this sequence".
        pos_max = C.llama_memory_seq_pos_max(self._mem, slot)
        if pos_max != -1:
            raise SlotWipeError(
                f"slot {slot} still holds KV after wipe (pos_max={pos_max})")

    # -- introspection, for the section 12 harness ------------------------------

    @property
    def free_count(self):
        with self._lock:
            return len(self._free)

    @property
    def in_use_count(self):
        with self._lock:
            return len(self._in_use)

    @property
    def quarantined(self):
        with self._lock:
            return frozenset(self._quarantined)


class TemplateError(RuntimeError):
    """The model's chat template does not support incremental turn building."""


class ChatFormatter:
    """Builds prompt tokens for a session, one turn at a time (section 6c).

    Sessions keep their KV cache resident so each turn prefills only the NEW
    tokens. Re-rendering and re-prefilling the whole history every turn would
    defeat that entirely. So this class tracks exactly what text is already
    represented in the KV and hands back only the delta.

    Model-agnostic BY CONSTRUCTION. An earlier draft of section 6c specified
    appending a hardcoded '<|im_start|>{role}\\n{content}<|im_end|>\\n' fragment.
    That is ChatML, i.e. Qwen-specific -- it would silently produce malformed
    structure on gemma-3, whose template uses <start_of_turn>. Instead the delta
    is computed FROM the model's own template: render the full conversation, and
    take the suffix past what is already rendered. The template is the single
    source of truth and no marker is hardcoded anywhere.

    Why the delta can be tokenized on its own, which is the non-obvious part:
    BPE merges across concatenation boundaries in general -- measured,
    tok("hell")+tok("o") = [56095, 78] but tok("hello") = [14990]. Tokenizing a
    delta separately would therefore normally be unsafe. It is safe here only
    because every turn boundary begins at a SPECIAL token (the turn terminator),
    and special tokens are hard boundaries the BPE merge never crosses --
    verified: splitting 'I am Qwen.<|im_end|>' at the special token yields
    identical ids. verify_against_full() re-checks this property rather than
    trusting it.
    """

    def __init__(self, model, vocab):
        self._model = model
        self._vocab = vocab
        tmpl = C.llama_model_chat_template(model, None)
        if not tmpl:
            raise TemplateError("model carries no chat template")
        self._tmpl = tmpl
        self._messages = []      # full logical history, for re-rendering
        self._rendered = ""      # EXACTLY the text currently represented in KV

    # -- rendering ------------------------------------------------------------

    def _apply(self, messages, add_generation_prompt):
        import ctypes
        arr = (C.llama_chat_message * len(messages))()
        keep = []                      # keep byte objects alive past the loop
        for i, (role, content) in enumerate(messages):
            rb, cb = role.encode(), content.encode()
            keep.append((rb, cb))
            arr[i].role = ctypes.c_char_p(rb)
            arr[i].content = ctypes.c_char_p(cb)
        size = 65536
        while True:
            buf = ctypes.create_string_buffer(size)
            n = C.llama_chat_apply_template(
                self._tmpl, arr, len(messages), add_generation_prompt, buf, size)
            if n < 0:
                raise TemplateError(f"llama_chat_apply_template failed ({n})")
            if n <= size:
                return buf.raw[:n].decode("utf-8")
            size = n + 1               # documented API: retry with the needed size

    def _tokenize(self, text, parse_special=True):
        import ctypes
        b = text.encode("utf-8")
        if not b:
            return []
        cap = len(b) + 64
        buf = (C.llama_token * cap)()
        n = C.llama_tokenize(self._vocab, b, len(b), buf, cap,
                             False,            # add_special: the template already
                                               # emits any BOS/system preamble
                             parse_special)
        if n < 0:
            raise TemplateError(f"llama_tokenize overflow ({n})")
        return list(buf[:n])

    # -- the two operations the engine actually calls --------------------------

    def user_turn(self, text):
        """Record a user message and return ONLY the new tokens to prefill."""
        self._messages.append(("user", text))
        full = self._apply(self._messages, True)

        # Load-bearing invariant, checked rather than assumed: everything already
        # in the KV must still be a prefix of the newly rendered conversation. If
        # a template ever violates this (a summarising or history-rewriting
        # template would), incremental prefill is invalid and the KV would
        # silently disagree with what the model thinks it has seen. Fail loudly
        # instead -- this is the "subtly wrong output" class, not a crash class.
        if not full.startswith(self._rendered):
            raise TemplateError(
                "template broke the prefix property; incremental prefill is "
                "unsafe for this model")

        delta = full[len(self._rendered):]
        self._rendered = full
        return self._tokenize(delta)

    def assistant_generated(self, text):
        """Record what the model actually produced.

        The generated tokens are already in the KV, so this adds no tokens. It
        only keeps _rendered in step with reality. Note the KV does NOT contain
        the turn terminator: generation stops AT the end-of-generation token and
        that token is never decoded. The next user_turn's delta therefore begins
        with the terminator, which is exactly what makes that delta start on a
        special token.
        """
        self._messages.append(("assistant", text))
        self._rendered += text

    # -- self-check for the section 12 harness ---------------------------------

    def verify_against_full(self):
        """Assert incremental building matches a from-scratch render, in TOKENS.

        Section 6c only ever compared strings. The KV holds tokens, and BPE can
        merge across boundaries, so string equality does not imply token
        equality. This checks the property that actually matters.
        """
        truth = self._tokenize(self._apply(self._messages, True))
        rebuilt = self._tokenize(self._rendered)
        if truth != rebuilt:
            raise TemplateError(
                f"incremental render diverged from full render "
                f"({len(rebuilt)} tokens vs {len(truth)})")
        return True


def is_stop_token(vocab, token):
    """True if generation must stop at this token.

    Uses llama_vocab_is_eog, NOT a comparison against llama_vocab_eos. Measured
    on Qwen3-0.6B: eos() returns only <|im_end|> (151645), but SIX tokens are
    flagged end-of-generation -- </s>, <|endoftext|>, <|im_end|>, <|fim_pad|>,
    <|repo_name|>, <|file_sep|>. Checking eos alone would miss five of them and
    the model would run on until the output cap instead of ending the turn.
    """
    return bool(C.llama_vocab_is_eog(vocab, token))
