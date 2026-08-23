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

    # -- checkpoint / restore, for section 6b's rollback -----------------------

    def checkpoint(self):
        """Capture enough state to undo an abandoned turn.

        Section 6b specifies rollback purely as KV positions. It predates
        section 6c, which introduced a SECOND piece of per-session state -- the
        rendered conversation -- that must roll back in lockstep. Rolling back
        one without the other desyncs the formatter from the KV, and the next
        turn is built on a conversation the model never saw.
        """
        return (len(self._messages), len(self._rendered))

    def restore(self, cp):
        n_messages, n_rendered = cp
        del self._messages[n_messages:]
        self._rendered = self._rendered[:n_rendered]

    # -- self-check for the section 12 harness ---------------------------------

    def verify_against_full(self):
        """Assert incremental building matches a from-scratch render, in TOKENS.

        Section 6c only ever compared strings. The KV holds tokens, and BPE can
        merge across boundaries, so string equality does not imply token
        equality. This checks the property that actually matters.
        """
        # What _rendered SHOULD be depends on which phase the session is in, and
        # the two are legitimately different -- an earlier version of this
        # method compared against the wrong one and reported a false divergence.
        #
        # _rendered tracks exactly what is in the KV. After a completed
        # assistant turn the KV holds the full render up to and including the
        # user turn, plus the generated text -- but NOT the turn terminator
        # (generation stops AT the end-of-generation token and never decodes it)
        # and NOT the next generation prompt.
        if self._messages and self._messages[-1][0] == "assistant":
            expected = self._apply(self._messages[:-1], True) + self._messages[-1][1]
        else:
            expected = self._apply(self._messages, True)
        truth = self._tokenize(expected)
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


# ---------------------------------------------------------------------------
# Sections 6 / 6b / 7 -- session state and the single-threaded engine loop
# ---------------------------------------------------------------------------

import queue
import time


# Response frame types on the wire (section 10a). Exactly one TERMINAL frame
# (COMPLETE or ERROR) ends every request -- see terminate() for why that is
# unconditional rather than best-effort.
FRAME_TOKEN = 0
FRAME_COMPLETE = 1
FRAME_ERROR = 2


class SessionState:
    IDLE = "idle"
    PENDING = "pending"          # prompt accepted, prefill not finished
    GENERATING = "generating"    # producing response tokens
    DEAD = "dead"                # torn down; never dispatched again


class OutboxFull(RuntimeError):
    """The client stopped reading. Section 7's policy is to disconnect it."""


class Session:
    """One conversation, pinned to one KV slot.

    Identity is (extension_id, tab_id, origin) -- origin included because
    without it, navigating a tab from a bank site to another site let the new
    site's content script inherit a session still holding the bank
    conversation (section 5g).
    """

    def __init__(self, key, slot, formatter, output_cap):
        self.key = key
        self.slot = slot
        self.formatter = formatter
        self.output_cap = output_cap

        self.state = SessionState.IDLE
        self.pos = 0                      # absolute KV position
        self.produced = 0                 # tokens emitted this response
        self.inbox_tokens = []            # prompt tokens awaiting prefill
        self.prefill_offset = 0
        self.outbox = queue.Queue(maxsize=MAX_OUTBOX_TOKENS)
        self.streamer = Utf8Streamer()
        self.sampler = None               # per-session: see Engine._make_sampler
        self._reply_bytes = bytearray()   # what the model produced this turn

        # Section 6b's TWO snapshots. One is not enough: pos_before_generation
        # is right for interrupting a response already streaming, but a replace
        # arriving DURING prefill of a large prompt has no earlier point to roll
        # back to. Both are absolute positions, and section 8's compact() owns
        # shifting them -- if that shift is ever skipped, rollback silently
        # targets the wrong position.
        self.pos_before_request = 0
        self.pos_before_generation = 0
        # The formatter's matching checkpoint. Must be captured and restored at
        # exactly the same moments as pos_before_request, or the two states
        # diverge -- see ChatFormatter.checkpoint().
        self.formatter_cp_before_request = None

        # Checked at the TOP of the loop, before dispatch -- same <=1-token
        # bound as tab-close cancellation, same flag pattern, no new machinery.
        self.pending_replace = None
        self.cancelled = False
        self.cancel_reason = None

    # -- outbox ------------------------------------------------------------

    def emit(self, frame_type, payload):
        """Queue one frame. Raises OutboxFull if the reader has stalled.

        Section 7 considered three policies and only one is defensible:
        blocking the engine stalls the single shared execution slot for every
        other session over one slow reader; dropping tokens silently corrupts
        the response with no indication; so a stalled reader is treated exactly
        like a dead one and routed through the ordinary teardown path.
        """
        try:
            self.outbox.put_nowait((frame_type, payload))
        except queue.Full:
            raise OutboxFull(f"session {self.key} outbox full ({MAX_OUTBOX_TOKENS})")

    def terminate(self, frame_type, payload=""):
        """Emit the single terminal frame, bypassing the queue bound if needed.

        Section 10a rule 2: a cancelled request MUST produce a terminal frame.
        Without one the browser waits out its 60s SO_RCVTIMEO instead of ending
        promptly. That makes this the one emission that must not fail because
        the queue is full -- the whole point is to unblock a stuck reader, so
        the bound that protects against a stuck reader cannot be allowed to
        prevent it.
        """
        try:
            self.outbox.put_nowait((frame_type, payload))
        except queue.Full:
            try:
                self.outbox.get_nowait()          # make room by dropping oldest
            except queue.Empty:
                pass
            try:
                self.outbox.put_nowait((frame_type, payload))
            except queue.Full:
                pass                              # reader is gone entirely


# Section 5d: two sampling configs, deliberately separate rather than one shared
# default. Canary/fidelity tests (§12 tests 1,4,18,27,32,40) assume deterministic
# output -- with any randomness they become probabilistic rather than pass/fail.
# Real chat must NOT be greedy: argmax produces flat, repetitive text. The GGUF
# carries no sampling defaults (25 metadata keys scanned, none sampling-related),
# so these are chosen deliberately, not read from the model.
SAMPLING_CHAT = {"temperature": 0.7, "top_p": 0.9, "top_k": 40}
SAMPLING_DETERMINISTIC = {"temperature": 0.0}


class Engine:
    """The single engine thread. Owns the model context; nothing else touches it.

    Single-threadedness is a NAMED INVARIANT, not incidental. It is what makes
    mid-batch cancellation impossible by construction: a session flipped to
    cancelled is excluded from selection entirely, before any batch is composed.
    Any new periodic mechanism must run inside this loop, never on a timer
    thread -- and that includes the slot wipe, which is why SlotAllocator splits
    prepare() (engine thread) from acquire() (any thread).
    """

    def __init__(self, ctx, model, vocab, allocator, sampling=None, seed=0):
        self._ctx = ctx
        self._model = model
        self._vocab = vocab
        self._alloc = allocator
        self._sampling = sampling if sampling is not None else SAMPLING_CHAT
        self._seed = seed

        self._sessions = {}                 # key -> Session
        self._reg_lock = threading.Lock()   # guards _sessions only
        self._running = False
        self._thread = None
        self.foreground_tab_id = -1         # section 5f: ONE key, not a per-session flag

    # -- registry (called from connection threads) --------------------------

    def get_or_create(self, key, formatter_factory, output_cap):
        """Look up or create a session. ONE atomic critical section.

        "Look up, and if absent create" must not be two steps -- two tabs
        arriving together would otherwise both find nothing and both create,
        and one of the two slots would leak with a session nobody can reach.
        """
        with self._reg_lock:
            s = self._sessions.get(key)
            if s is not None and s.state != SessionState.DEAD:
                return s, False
            slot = self._alloc.acquire()
            if slot is None:
                return None, False          # at capacity -- caller rejects
            s = Session(key, slot, formatter_factory(), output_cap)
            self._sessions[key] = s
            return s, True

    def evict_other_origins(self, ext_id, tab_id, origin):
        """Section 10a rule 1: creating a session for (ext, tab, origin) tears
        down any session with the same (ext, tab) and a DIFFERENT origin.

        This is what makes cross-origin isolation work with no navigation
        observer at all -- the next request from that tab simply carries a
        different origin, and the old session cannot survive it.
        """
        with self._reg_lock:
            doomed = [k for k in self._sessions
                      if k[0] == ext_id and k[1] == tab_id and k[2] != origin]
        for k in doomed:
            self.cancel(k, "cross-origin eviction")
        return doomed

    def cancel(self, key, reason):
        """Mark a session for teardown. Safe from any thread: sets flags only.

        The actual KV work happens on the engine thread. That separation is the
        same one SlotAllocator enforces, for the same reason.
        """
        with self._reg_lock:
            s = self._sessions.get(key)
        if s is None or s.state == SessionState.DEAD:
            return False                    # defined no-op (section 10a)
        s.cancelled = True
        s.cancel_reason = reason
        return True

    def submit(self, key, text):
        """Queue a new prompt. Section 6b: replaces any generation in flight."""
        with self._reg_lock:
            s = self._sessions.get(key)
        if s is None or s.state == SessionState.DEAD:
            return False
        # Do NOT touch pos or KV here -- that is engine-thread work. Only the
        # flag is set; the loop performs the rollback at its next top.
        s.pending_replace = text
        return True

    # -- engine thread ------------------------------------------------------

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, name="malabr-engine",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout=5.0):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self):
        while self._running:
            did_work = self._step()
            if not did_work:
                time.sleep(0.001)

    def _step(self):
        """One iteration. Returns True if any work was done.

        Split out from _run so tests can drive the loop deterministically
        instead of racing a background thread.
        """
        # Control flags FIRST, before anything is dispatched. This ordering is
        # what gives cancellation its <=1-token bound and makes mid-batch
        # cancellation unrepresentable.
        self._apply_control()

        s = self._pick()
        if s is None:
            return False
        try:
            if s.state == SessionState.PENDING:
                self._prefill_step(s)
            elif s.state == SessionState.GENERATING:
                self._decode_step(s)
            else:
                return False
        except OutboxFull as e:
            # Stalled reader: same teardown path as a dead one (section 7).
            s.terminate(FRAME_ERROR, "reader stalled")
            self._teardown(s, str(e))
        except SlotWipeError as e:
            s.terminate(FRAME_ERROR, "slot integrity failure")
            self._teardown(s, str(e))
        return True

    def _apply_control(self):
        with self._reg_lock:
            sessions = list(self._sessions.values())
        for s in sessions:
            if s.state == SessionState.DEAD:
                continue
            if s.cancelled:
                self._roll_back_partial(s)
                s.terminate(FRAME_ERROR, s.cancel_reason or "cancelled")
                self._teardown(s, s.cancel_reason)
                continue
            if s.pending_replace is not None:
                text = s.pending_replace
                s.pending_replace = None
                # Section 6b: the old turn is rolled back, never partially
                # remembered. A turn enters permanent context ONLY by reaching
                # EOS or the output cap.
                if s.state in (SessionState.PENDING, SessionState.GENERATING):
                    self._roll_back_partial(s)
                    s.terminate(FRAME_ERROR, "superseded")
                self._begin_turn(s, text)

    def _begin_turn(self, s, text):
        s.pos_before_request = s.pos            # snapshot 1 (section 6b)
        s.formatter_cp_before_request = s.formatter.checkpoint()
        s.inbox_tokens = s.formatter.user_turn(text)
        s.prefill_offset = 0
        s.produced = 0
        s.streamer = Utf8Streamer()
        s._reply_bytes = bytearray()
        s.state = SessionState.PENDING

    def _roll_back_partial(self, s):
        """Undo a turn that did not finish. Engine thread only.

        DIVERGES FROM SECTION 6b, deliberately. 6b rolls a GENERATING session
        back to pos_before_generation, keeping the user message and the
        generation prompt in KV. That is not a usable boundary: the KV at that
        point ends inside an OPEN assistant block (the template's trailing
        '<|im_start|>assistant\n'), so appending a new user turn after it
        produces malformed structure -- which is exactly what the formatter's
        prefix check caught.

        Rolling back to pos_before_request instead is also what 6b's own stated
        rule requires: "a turn only enters permanent context if it reaches EOS
        or the output cap. Any other termination is rolled back -- never
        partially remembered." A turn is the user message AND its response, so
        both go. pos_before_generation is still tracked because section 8's
        compaction has to shift it, and a future resume/regenerate feature
        would need it.
        """
        if s.state not in (SessionState.PENDING, SessionState.GENERATING):
            return
        target = s.pos_before_request
        if s.pos > target:
            C.llama_memory_seq_rm(self._alloc._mem, s.slot, target, s.pos)
            s.pos = target
        # The formatter must roll back with the KV, not after it.
        if s.formatter_cp_before_request is not None:
            s.formatter.restore(s.formatter_cp_before_request)
            s.formatter_cp_before_request = None
        s.state = SessionState.IDLE

    def _teardown(self, s, reason=None):
        s.state = SessionState.DEAD
        with self._reg_lock:
            if self._sessions.get(s.key) is s:
                del self._sessions[s.key]
        try:
            self._alloc.release(s.slot)
        except ValueError:
            pass                                # already released
        if s.sampler is not None:
            C.llama_sampler_free(s.sampler)
            s.sampler = None

    def _pick(self):
        """Choose one runnable session. Replaced by §9a/§9b's scheduler later.

        Cancelled and DEAD sessions are excluded HERE, before any dispatch --
        that exclusion is what makes mid-batch cancellation impossible rather
        than merely unlikely.
        """
        with self._reg_lock:
            runnable = [s for s in self._sessions.values()
                        if s.state in (SessionState.PENDING, SessionState.GENERATING)
                        and not s.cancelled]
        if not runnable:
            return None
        # Foreground first. One key on the registry, not a per-session boolean:
        # a boolean can represent "5 tabs foreground at once", which is a state
        # that must be unrepresentable (section 5f).
        fg = [s for s in runnable if s.key[1] == self.foreground_tab_id]
        return (fg or runnable)[0]

    # -- the two model-touching steps ---------------------------------------

    def _make_sampler(self, s):
        """One sampler PER SESSION, never shared.

        Not merely tidy: samplers carry state (RNG for dist, and any penalty
        samplers added later). A shared chain would couple sessions' sampling
        to each other, which is the same class of cross-session bleed the KV
        slot wipe exists to prevent -- just in a different piece of state.
        """
        params = C.llama_sampler_chain_default_params()
        chain = C.llama_sampler_chain_init(params)
        temp = self._sampling.get("temperature", 0.0)
        if temp <= 0.0:
            C.llama_sampler_chain_add(chain, C.llama_sampler_init_greedy())
        else:
            if "top_k" in self._sampling:
                C.llama_sampler_chain_add(
                    chain, C.llama_sampler_init_top_k(int(self._sampling["top_k"])))
            if "top_p" in self._sampling:
                C.llama_sampler_chain_add(
                    chain, C.llama_sampler_init_top_p(float(self._sampling["top_p"]), 1))
            C.llama_sampler_chain_add(chain, C.llama_sampler_init_temp(float(temp)))
            # Seed per session so a test can reproduce one session's stream
            # without every session sharing one global sequence.
            C.llama_sampler_chain_add(
                chain, C.llama_sampler_init_dist(self._seed + (s.slot * 7919)))
        return chain

    def _decode_batch(self, items):
        """items: [(token, pos, seq_id, want_logits)] -> logits index per item."""
        n = len(items)
        batch = C.llama_batch_init(n, 0, 1)
        try:
            batch.n_tokens = n
            idx = {}
            out_i = 0
            for i, (tok, pos, seq, want) in enumerate(items):
                batch.token[i] = tok
                batch.pos[i] = pos
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = seq
                batch.logits[i] = 1 if want else 0
                if want:
                    idx[i] = out_i
                    out_i += 1
            rc = C.llama_decode(self._ctx, batch)
            if rc != 0:
                # Checked, not ignored. The handoff records two past cases where
                # an unchecked return code produced physically impossible
                # numbers; a silent decode failure here would desync s.pos from
                # the KV and corrupt the conversation with no visible error.
                raise RuntimeError(f"llama_decode failed rc={rc}")
            return idx
        finally:
            C.llama_batch_free(batch)

    def _prefill_step(self, s):
        """Prefill the prompt. Chunking (§9b) replaces the all-at-once call."""
        # The wipe happens HERE, on the engine thread, because llama.h
        # guarantees thread-safety only for the tokenization API -- mutating KV
        # from the connection thread that called acquire() would race
        # llama_decode. Guarded so it fires only on a slot's first use: prepare()
        # clears the WHOLE sequence, so running it on a later turn would erase
        # the conversation instead of protecting it.
        if s.slot in self._alloc.pending_wipe:
            self._alloc.prepare(s.slot)
        self._alloc.assert_ready(s.slot)     # refuses an unwiped slot
        toks = s.inbox_tokens
        if not toks:
            s.state = SessionState.IDLE
            return
        items = [(t, s.pos + i, s.slot, i == len(toks) - 1)
                 for i, t in enumerate(toks)]
        self._decode_batch(items)
        s.pos += len(toks)
        s.prefill_offset = len(toks)

        # Snapshot 2 (section 6b), taken once prefill completes and BEFORE the
        # first output token, so an interrupted response rolls back to exactly
        # the end of the prompt rather than into the middle of it.
        s.pos_before_generation = s.pos
        if s.sampler is None:
            s.sampler = self._make_sampler(s)
        s.state = SessionState.GENERATING

    def _decode_step(self, s):
        """Produce exactly ONE token, then return to the scheduler.

        Returning after every single token IS the preemption mechanism --
        nothing ever commits to more than one token, so a foreground request
        arriving mid-background-generation waits at most one token (~50ms under
        the Phase 1 cpu.max quota; ~23ms unthrottled).
        """
        self._alloc.assert_ready(s.slot)
        tok = C.llama_sampler_sample(s.sampler, self._ctx, -1)

        cap = s.output_cap.for_position(s.pos)
        if is_stop_token(self._vocab, tok):
            self._finish_turn(s, FRAME_COMPLETE, "eos")
            return
        if s.produced >= cap:
            # The cap is a real completion, not an error: the turn DOES enter
            # permanent context (section 6b's rule names EOS and the output cap
            # as the two ways a turn is kept).
            self._finish_turn(s, FRAME_COMPLETE, "cap")
            return

        C.llama_sampler_accept(s.sampler, tok)
        raw = self._token_bytes(tok)
        s._reply_bytes.extend(raw)
        text = s.streamer.push(raw)
        if text:
            s.emit(FRAME_TOKEN, text)

        self._decode_batch([(tok, s.pos, s.slot, True)])
        s.pos += 1
        s.produced += 1

    def _finish_turn(self, s, frame_type, reason):
        tail = s.streamer.flush()
        if tail:
            s.emit(FRAME_TOKEN, tail)
        # Record what the model actually produced so the formatter's notion of
        # the conversation matches the KV exactly. Skipping this would make the
        # NEXT turn's delta wrong -- silently, with no error.
        s.formatter.assistant_generated(
            bytes(s._reply_bytes).decode("utf-8", "replace"))
        s.terminate(frame_type, reason)
        s.state = SessionState.IDLE
        s.produced = 0

    def _token_bytes(self, tok):
        import ctypes
        buf = (ctypes.c_char * 256)()
        n = C.llama_token_to_piece(self._vocab, tok, buf, 256, 0, True)
        if n < 0:
            raise RuntimeError(f"llama_token_to_piece failed ({n})")
        return buf.raw[:n]
