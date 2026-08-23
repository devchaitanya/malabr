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
    still reports llama_memory_seq_pos_max() == 12 for a 13-token conversation.
    The KV really does persist; this class is what stops the next session
    reading it.

    THREAD SPLIT -- the non-obvious part, and a correction to section 7.
    Section 7's sketch has acquire() call llama_memory_seq_rm directly, while
    its own concurrency note says connection threads create sessions
    concurrently. Those two statements cannot both hold. llama.h documents "The
    API is thread-safe" exactly once, scoped to the Tokenization section; the
    Memory section carries no such guarantee. Mutating KV from a connection
    thread while the engine thread is inside llama_decode() on the same context
    is therefore an undocumented data race, and section 7 separately names
    single-threadedness as a load-bearing invariant.

    So the work is split by thread, and the split is enforced, not documented:
      - acquire() / release()  -- ANY thread. Pure bookkeeping under a lock.
                                  Touches no model state whatsoever.
      - prepare(slot)          -- ENGINE THREAD ONLY. Does the actual wipe.
      - assert_ready(slot)     -- the guard the decode path calls, which makes
                                  "no token is ever decoded into an unwiped
                                  slot" a checked property rather than a hope.

    A slot handed out by acquire() is NOT yet safe to use. It carries a pending
    wipe until the engine thread clears it. That is why every slot in _free is
    treated as dirty regardless of how it got there.
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
        # removed. The real race is the CHECK-THEN-ACT pair in acquire():
        # threads all pass "if not self._free", then race to pop an emptied
        # list. With the window widened to 2ms, no-lock produced 120 IndexErrors
        # across 160 acquires; with the lock, zero. So the damage is a crash in
        # session creation for a request that should have been cleanly rejected
        # as "no capacity" -- not a cross-session leak.
        self._lock = threading.Lock()
        self._free = list(range(n_seq_max))
        self._in_use = set()
        # Slots handed out but not yet wiped by the engine thread. A slot here
        # must never receive a decoded token.
        self._pending_wipe = set()
        # Slots whose wipe could not be verified. Deliberately leaked rather
        # than reused: losing capacity is recoverable, leaking a conversation
        # into another origin's session is not.
        self._quarantined = set()

    # -- callable from ANY thread: bookkeeping only, no model calls ------------

    def acquire(self):
        """Reserve a slot, or return None if at capacity.

        The returned slot is NOT usable yet -- it carries a pending wipe that
        the engine thread must clear via prepare(). No model state is touched
        here, deliberately: see the THREAD SPLIT note above.
        """
        with self._lock:
            if not self._free:
                return None          # at capacity -- caller rejects the session
            slot = self._free.pop()
            self._in_use.add(slot)
            self._pending_wipe.add(slot)
            return slot

    def release(self, slot):
        """Return a slot to the pool. Bookkeeping only; no model call.

        There is deliberately no wipe here. Section 7's sketch wiped on release
        too, "belt and braces" -- but that wipe would have to run on a
        connection thread, which is exactly the race described above. Dropping
        it costs nothing, because the whole rationale for wipe-on-acquire is
        that release cannot be trusted to happen at all. Every slot in _free is
        treated as dirty regardless.
        """
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
            self._pending_wipe.discard(slot)
            self._free.append(slot)

    # -- ENGINE THREAD ONLY: everything below touches the model ----------------

    def prepare(self, slot):
        """Wipe a freshly acquired slot and PROVE it is empty.

        MUST be called on the engine thread, before the first decode into this
        slot. Raises SlotWipeError if the slot cannot be proven empty, in which
        case the slot is quarantined and never reused.
        """
        with self._lock:
            if slot not in self._in_use:
                raise ValueError(f"prepare of slot {slot} that is not in use")

        try:
            self._wipe_and_verify(slot)
        except SlotWipeError:
            with self._lock:
                self._in_use.discard(slot)
                self._pending_wipe.discard(slot)
                self._quarantined.add(slot)   # NOT returned to _free
            raise

        with self._lock:
            self._pending_wipe.discard(slot)

    def assert_ready(self, slot):
        """Guard for the decode path: refuse to use a slot that is not wiped.

        This is what turns "we wipe before use" from a convention into a checked
        invariant. A missed prepare() becomes a loud error at the decode call
        instead of a silent cross-session KV leak.
        """
        with self._lock:
            if slot in self._pending_wipe:
                raise SlotWipeError(
                    f"slot {slot} used before prepare() -- would expose the "
                    f"previous session's KV")
            if slot not in self._in_use:
                raise ValueError(f"slot {slot} is not in use")

    def _wipe_and_verify(self, slot):
        """Clear a sequence and PROVE it is empty. Both checks are load-bearing."""
        # llama_memory_seq_rm is declared to return bool. On this model every
        # call returned True (verified: full, partial, and empty-slot removal),
        # so this branch is defensive rather than observed-necessary. It is
        # checked because llama.cpp returns false where partial removal is
        # unsupported -- the sliding-window-attention case, which is exactly
        # what section 8 compaction will exercise on models like gemma-3.
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
    def pending_wipe(self):
        with self._lock:
            return frozenset(self._pending_wipe)

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


# ---------------------------------------------------------------------------
# Section 6d -- UTF-8 safe streaming
# ---------------------------------------------------------------------------

class Utf8Streamer:
    """Buffers token bytes so no frame ever splits a UTF-8 character.

    Measured, not hypothetical: scanning the first 20,000 vocabulary entries,
    282 tokens (~1.4%) produce bytes that are NOT valid UTF-8 standalone -- e.g.
    token 94 -> b'\\xa1', one piece of a multi-byte character from BPE's
    byte-level fallback. Emitting each token's bytes as produced means any
    non-English text, many symbols, or emoji eventually splits a character
    across two frames, and the browser-side decoder shows a replacement
    character or throws.

    So: accumulate bytes, emit only the prefix that decodes cleanly, and hold
    the trailing partial bytes for the next token.
    """

    # A UTF-8 character is at most 4 bytes, so a legitimate partial sequence can
    # never exceed 3 held bytes. Anything longer is genuinely malformed output
    # rather than an incomplete character, and holding it forever would stall
    # the stream silently -- flush it lossily instead of hanging.
    MAX_HELD_BYTES = 3

    def __init__(self):
        self._buf = bytearray()

    def push(self, raw):
        """Add token bytes; return the text safe to send now (may be '')."""
        self._buf.extend(raw)
        # Find the longest prefix that is valid UTF-8. Decoding the whole buffer
        # and catching the error tells us exactly where the good prefix ends,
        # which is cheaper and more precise than scanning byte patterns.
        try:
            text = self._buf.decode("utf-8")
            self._buf.clear()
            return text
        except UnicodeDecodeError as e:
            good = self._buf[:e.start].decode("utf-8")
            held = self._buf[e.start:]
            if len(held) > self.MAX_HELD_BYTES:
                # Not a partial character -- genuinely invalid bytes. Emit a
                # replacement rather than holding them forever.
                good += held.decode("utf-8", "replace")
                held = bytearray()
            self._buf = bytearray(held)
            return good

    def flush(self):
        """End of stream: emit whatever is left, lossily if incomplete.

        Trailing bytes here mean generation stopped mid-character. Showing a
        replacement character is more honest than silently dropping output.
        """
        if not self._buf:
            return ""
        text = self._buf.decode("utf-8", "replace")
        self._buf.clear()
        return text


# ---------------------------------------------------------------------------
# Section 8 / section 11a Phase D+G -- output cap
# ---------------------------------------------------------------------------

# Phase G: one flat, conservative outer bound that applies ALWAYS, whatever the
# calibrated curve says. The curve refines behaviour INSIDE this ceiling and can
# never replace it -- otherwise corrupted calibration data yields a correspondingly
# corrupted cap with no structural limit (section 11a Phase G).
ABSOLUTE_CEILING = 512
MIN_CAP = 64

# Section 7: bounds the OUTPUT side, which none of the KV bounds cover. If the
# reader stalls, the engine would otherwise produce into an unread queue forever.
MAX_OUTBOX_TOKENS = 2 * ABSOLUTE_CEILING


class OutputCap:
    """Position-aware cap on tokens per response (section 8's context-growth fix).

    A FLAT cap is wrong, and this is the one number the measurements most
    directly contradict: decode runs at 48.3 tok/s at position 64 but 5.2 tok/s
    at position 6000. A flat 512-token cap therefore costs ~11s early in a
    conversation and ~98s deep into one -- the same number meaning wildly
    different wall-clock time.

    Calibration (Phase D) hands over a FINISHED table; the engine never derives
    caps from raw curve data at runtime (section 12 test 31).
    """

    def __init__(self, table=None):
        # table: sorted [(position, cap)]. None => no calibration data yet, so
        # fall back to the floor rather than guessing high. Being too
        # conservative truncates a reply; being too generous blows the latency
        # budget the cap exists to enforce.
        self._table = sorted(table) if table else None

    def for_position(self, pos):
        if self._table is None:
            return MIN_CAP
        cap = self._table[0][1]
        for position, value in self._table:
            if pos >= position:
                cap = value
            else:
                break
        # Phase G's ceiling is applied here too, not only at table-build time,
        # so a corrupted or hand-edited table still cannot exceed it.
        return max(MIN_CAP, min(int(cap), ABSOLUTE_CEILING))
