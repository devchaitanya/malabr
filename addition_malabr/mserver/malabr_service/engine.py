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

    def _expected_render(self):
        """The text that SHOULD currently be in the KV.

        Phase-dependent, and the two cases are legitimately different -- an
        earlier version compared against the wrong one and reported a false
        divergence. After a completed assistant turn the KV holds the full
        render up to and including the user turn plus the generated text, but
        NOT the turn terminator (generation stops AT the end-of-generation
        token and never decodes it) and NOT the next generation prompt.
        """
        if self._messages and self._messages[-1][0] == "assistant":
            return self._apply(self._messages[:-1], True) + self._messages[-1][1]
        return self._apply(self._messages, True)

    def drop_messages(self, start, count):
        """Remove messages that section 8 compaction evicted from the KV.

        Section 8 specifies compaction entirely in terms of KV positions and
        never mentions this. That is the same gap section 6b had: the formatter
        is a SECOND per-session state holding the conversation, and a KV
        mutation that does not update it leaves _rendered describing content the
        model no longer has. The next turn is then built against a conversation
        that does not exist.
        """
        del self._messages[start:start + count]
        self._rendered = self._expected_render()

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
        truth = self._tokenize(self._expected_render())
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


# Section 8 input cap. RESERVED_FOR_RESPONSE guarantees room is left for the
# model to ANSWER, not merely to fit the question -- without it a prompt could
# legally consume the entire budget and leave nothing to reply with.
RESERVED_FOR_RESPONSE = 256


class Turn:
    """One complete exchange (user delta + generated response) in KV.

    MUTABLE on purpose. Section 8's position-shift fix has to write corrected
    positions back into entries still in the list; tuples cannot do that, and
    the bug it fixes is precisely that other turns kept stale positions.

    Range is [start, end): from the first token of the user delta to the last
    token of the response. That boundary is not arbitrary -- every delta begins
    with the terminator that CLOSES the preceding assistant block (a
    consequence of section 6c deriving deltas from the template). So dropping a
    whole exchange leaves the survivor's assistant block to be closed by the
    NEXT surviving delta, and the result is well-formed with no repair step.
    """

    __slots__ = ("start", "end", "role", "msg_index")

    def __init__(self, start, end, role, msg_index):
        self.start = start
        self.end = end
        self.role = role
        self.msg_index = msg_index      # index of the USER message in formatter

    def __repr__(self):
        return f"Turn({self.start},{self.end},{self.role},m={self.msg_index})"


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

        # Section 8: completed exchanges, oldest first. turn_boundaries[0] is
        # the ANCHOR -- kept always, because it carries the template's one-time
        # system/tools preamble and StreamingLLM's finding that the first ~32
        # tokens act as attention anchors whose loss degrades output sharply.
        self.turn_boundaries = []
        self.budget = 2048              # n_ctx / n_seq_max; set by the caller

        # §9a aging: consecutive rounds this session was passed over.
        self.rounds_excluded = 0
        # §9b: consecutive rounds a PENDING session was skipped because the
        # leftover budget could not fund a minimum-size chunk. Not in §9b --
        # see build_batch for the starvation this prevents.
        self.prefill_stalls = 0
        # The token this session must decode next round. Set when sampled,
        # consumed when the next batch is composed.
        self.next_token = None

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
        # Held from calibration's Phase D output at startup, never re-derived
        # per round (§12 test 31).
        self.cost_curve = CostCurve()

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

        try:
            return self._run_round()
        except OutboxFull:
            return True
        except SlotWipeError:
            return True

    def _run_round(self):
        """One batched round: prefill chunks and decode tokens in ONE pass."""
        with self._reg_lock:
            sessions = list(self._sessions.values())

        # §8: the 95% check runs BEFORE composition, never after. One batched
        # decode advances several sessions' pos at once, so checking afterwards
        # races the batch -- a session one token from its ceiling gets pushed
        # past it before anything re-checks.
        for s in sessions:
            if s.state in (SessionState.PENDING, SessionState.GENERATING) \
                    and not s.cancelled and self.needs_compaction(s):
                self.compact(s)

        decode_picks, prefill_chunks, estimated_ms = build_batch(
            sessions, self.cost_curve, self.foreground_tab_id)
        if not decode_picks and not prefill_chunks:
            return False

        # Slots must be provably wiped before ANY token enters them. On the
        # engine thread, which is the whole point of the prepare/acquire split.
        for s in decode_picks:
            self._alloc.assert_ready(s.slot)
        for s, _ in prefill_chunks:
            if s.slot in self._alloc.pending_wipe:
                self._alloc.prepare(s.slot)
            self._alloc.assert_ready(s.slot)

        items, owner = [], {}
        for s in decode_picks:
            owner[len(items)] = ("decode", s)
            items.append((s.next_token, s.pos, s.slot, True))
        for s, chunk in prefill_chunks:
            for i, tok in enumerate(chunk):
                last = (i == len(chunk) - 1)
                completes = last and (s.prefill_offset + len(chunk) >= len(s.inbox_tokens))
                if completes:
                    owner[len(items)] = ("prefill", s)
                items.append((tok, s.pos + i, s.slot, completes))

        t0 = time.perf_counter()
        self._decode_batch(items)
        observed_ms = (time.perf_counter() - t0) * 1000.0

        # Each decode pick carried one token AT s.pos; it now occupies that
        # position. Advancing here, not in _sample_and_emit, keeps pos in step
        # with what the batch actually wrote -- the sampling that follows reads
        # logits produced BY these positions.
        for s in decode_picks:
            s.pos += 1

        # §9b part 2: feed the REAL round time back. Open-loop, a wrong cost
        # model silently violates the bound and only a dedicated experiment
        # would reveal it. Closed-loop, it shrinks batches until reality fits.
        self.cost_curve.observe_round(estimated_ms, observed_ms)

        for s, chunk in prefill_chunks:
            s.pos += len(chunk)
            s.prefill_offset += len(chunk)
            if s.prefill_offset >= len(s.inbox_tokens):
                s.pos_before_generation = s.pos
                if s.sampler is None:
                    s.sampler = self._make_sampler(s)
                s.state = SessionState.GENERATING

        for batch_i, (kind, s) in owner.items():
            if s.state != SessionState.GENERATING or s.cancelled:
                continue
            try:
                # batch index, NOT the output ordinal -- see _decode_batch.
                self._sample_and_emit(s, batch_i)
            except OutboxFull as e:
                s.terminate(FRAME_ERROR, "reader stalled")
                self._teardown(s, str(e))
        return True

    def _sample_and_emit(self, s, logits_index):
        """Produce exactly ONE token for this session, then return.

        Returning after every single token IS the preemption mechanism --
        nothing ever commits to more than one token, so a foreground request
        arriving mid-background-generation waits at most one round.
        """
        tok = C.llama_sampler_sample(s.sampler, self._ctx, logits_index)
        cap = s.output_cap.for_position(s.pos)
        if is_stop_token(self._vocab, tok):
            self._finish_turn(s, FRAME_COMPLETE, "eos")
            return
        if s.produced >= cap:
            self._finish_turn(s, FRAME_COMPLETE, "cap")
            return
        C.llama_sampler_accept(s.sampler, tok)
        raw = self._token_bytes(tok)
        s._reply_bytes.extend(raw)
        text = s.streamer.push(raw)
        if text:
            s.emit(FRAME_TOKEN, text)
        s.next_token = tok
        s.produced += 1

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
        # Section 8 gate 2: defense in depth at the point of no return. Gate 1
        # lives upstream at admission; one check point is a single point of
        # failure, and this one is structurally unskippable by anything above
        # it, because no turn can start without passing through here. If it ever
        # fires, something upstream is broken -- that is the signal, not silent
        # tolerance of the oversized input.
        max_input = s.budget - RESERVED_FOR_RESPONSE
        if len(s.inbox_tokens) > max_input:
            s.formatter.restore(s.formatter_cp_before_request)
            s.formatter_cp_before_request = None
            s.state = SessionState.IDLE
            s.inbox_tokens = []
            raise ValueError(
                f"gate 1 bypassed: {len(s.inbox_tokens)} tokens > "
                f"max_input {max_input}")
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
        """items: [(token, pos, seq_id, want_logits)] -> batch indices with logits.

        API subtlety that cost a crash: llama_get_logits_ith(ctx, i) -- which is
        what llama_sampler_sample calls -- takes the index WITHIN THE BATCH, not
        the ordinal among tokens that requested logits. llama.cpp maps it through
        an internal output_ids table. Passing the compacted ordinal aborts with
        GGML_ASSERT(logits != nullptr).

        This stayed latent while the engine sampled with index -1 ("last
        output"), which sidesteps the question. Batching makes several outputs
        per round real, so it had to be got right.
        """
        n = len(items)
        batch = C.llama_batch_init(n, 0, 1)
        try:
            batch.n_tokens = n
            idx = []
            for i, (tok, pos, seq, want) in enumerate(items):
                batch.token[i] = tok
                batch.pos[i] = pos
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = seq
                batch.logits[i] = 1 if want else 0
                if want:
                    idx.append(i)
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

    def _finish_turn(self, s, frame_type, reason):
        # Record the completed exchange BEFORE the streamer is flushed, so the
        # range covers exactly what went into the KV.
        s.turn_boundaries.append(
            Turn(s.pos_before_request, s.pos, "exchange",
                 len(s.formatter._messages)))   # index of the user msg
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

    # -- section 8: compaction ----------------------------------------------

    COMPACT_TRIGGER = 0.95
    TARGET_FREED_TOKENS = 512

    def needs_compaction(self, s):
        """Checked BEFORE a session enters a batch, never after.

        A correctness requirement, not a safety margin: one batched decode can
        advance several sessions' pos at once, so checking after the fact races
        the batch -- session B could be one token from its ceiling and get
        pushed past it before anything re-checked.
        """
        return s.pos >= s.budget * self.COMPACT_TRIGGER

    def compact(self, s):
        """Drop whole oldest exchanges, shifting EVERY live absolute position.

        Returns tokens freed. False-y result means nothing was droppable, which
        section 8 handles upstream by rejecting oversized input rather than
        letting compaction fail and deciding afterwards.
        """
        # Keep the anchor (index 0) and the most recent exchange. The anchor
        # carries the template preamble and the first ~32 tokens act as
        # attention anchors whose loss degrades output sharply.
        droppable = s.turn_boundaries[1:-1]
        freed = 0
        for turn in list(droppable):
            if freed >= self.TARGET_FREED_TOKENS:
                break
            n = turn.end - turn.start
            if n <= 0:
                continue

            # Invariant section 8 names but does not assert: nothing live may
            # sit strictly INSIDE a dropped range. Droppable turns are
            # fully-closed prior exchanges by construction, never the in-flight
            # one a snapshot could reference. Assert it rather than assume it --
            # a violation here silently rolls back to removed content.
            for name, val in (("pos_before_request", s.pos_before_request),
                              ("pos_before_generation", s.pos_before_generation)):
                if turn.start < val < turn.end:
                    raise RuntimeError(
                        f"{name}={val} lies inside dropped range "
                        f"[{turn.start},{turn.end}) -- compaction would corrupt it")

            C.llama_memory_seq_rm(self._alloc._mem, s.slot, turn.start, turn.end)
            C.llama_memory_seq_add(self._alloc._mem, s.slot, turn.end, -1, -n)

            # THE FIX. Shifting s.pos alone is not enough -- every other absolute
            # position recorded anywhere refers to the same shifted space, and
            # each one that is missed silently points at the wrong content.
            for other in s.turn_boundaries:
                if other is turn:
                    continue
                if other.start >= turn.end:
                    other.start -= n
                    other.end -= n
            if s.pos_before_request >= turn.end:
                s.pos_before_request -= n
            if s.pos_before_generation >= turn.end:
                s.pos_before_generation -= n
            s.pos -= n

            # The formatter is the OTHER state holding this conversation, and
            # section 8 never mentions it. Dropping KV without dropping the
            # matching messages leaves _rendered describing content the model no
            # longer has -- the same gap section 6b had with rollback.
            s.formatter.drop_messages(turn.msg_index, 2)
            for other in s.turn_boundaries:
                if other is not turn and other.msg_index > turn.msg_index:
                    other.msg_index -= 2

            s.turn_boundaries.remove(turn)
            freed += n
        return freed

    def _token_bytes(self, tok):
        import ctypes
        buf = (ctypes.c_char * 256)()
        n = C.llama_token_to_piece(self._vocab, tok, buf, 256, 0, True)
        if n < 0:
            raise RuntimeError(f"llama_token_to_piece failed ({n})")
        return buf.raw[:n]


# ---------------------------------------------------------------------------
# Sections 9a / 9b -- batch composition by latency budget, and the closed loop
# ---------------------------------------------------------------------------

ROUND_LATENCY_BUDGET_MS = 50      # the bound §7 already commits to elsewhere
MAX_CONSECUTIVE_EXCLUSIONS = 5    # aging bound; no measured value yet
PREFILL_CHUNK_MIN = 32            # below this, per-call overhead dominates
MAX_PREFILL_STALLS = 5            # see build_batch -- not in §9b, see note there
MAX_CORRECTION = 8.0              # §9b leaves this unnamed; see observe_round


class CostCurve:
    """Per-position cost estimates, with a closed loop on top (§9b part 2).

    Two statistics from the same data, for two different purposes -- §11a's own
    rule. The MEDIAN feeds representative things like the output-cap formula.
    The WORST-OBSERVED feeds hard cutoffs like batch admission, because the
    measured trial-to-trial spread is real (up to ~19% at short context) and a
    round estimated from the median could quietly run 20% over the bound it
    exists to enforce.
    """

    def __init__(self, decode_table=None, prefill_table=None):
        # decode_table: [(pos, tps_median, tps_worst)]
        # prefill_table: [(pos, tok_per_s_worst)]
        self._decode = sorted(decode_table) if decode_table else None
        self._prefill = sorted(prefill_table) if prefill_table else None
        self.correction = 1.0

    # -- interpolation -----------------------------------------------------

    @staticmethod
    def _lookup(table, pos, col):
        lo = table[0]
        for row in table:
            if row[0] <= pos:
                lo = row
            else:
                hi = row
                span = hi[0] - lo[0]
                if span <= 0:
                    return lo[col]
                f = (pos - lo[0]) / span
                return lo[col] + f * (hi[col] - lo[col])
        return lo[col]

    def _raw_worst(self, pos):
        if not self._decode:
            # No calibration yet. Assume the WHOLE budget per token: the most
            # conservative possible estimate, so an uncalibrated engine degrades
            # to one session per round rather than over-admitting on a guess.
            return ROUND_LATENCY_BUDGET_MS
        tps = self._lookup(self._decode, pos, 2)
        if tps <= 0:
            return ROUND_LATENCY_BUDGET_MS
        return 1000.0 / tps

    def cost_ms_worst(self, pos):
        return self._raw_worst(pos) * self.correction

    def cost_ms_median(self, pos):
        if not self._decode:
            return ROUND_LATENCY_BUDGET_MS
        tps = self._lookup(self._decode, pos, 1)
        return ROUND_LATENCY_BUDGET_MS if tps <= 0 else 1000.0 / tps

    # -- prefill (§9b Phase B-prefill) --------------------------------------

    def prefill_ms(self, n_tokens, pos=0):
        """Prefill cost is NOT decode cost; reusing the decode curve is wrong.

        Decode is memory-bandwidth-bound -- one token, but every prior K/V
        vector is read, which is why the decode curve collapses 48 -> 5 tok/s
        with depth. Prefill is compute-bound and parallel: many tokens in one
        pass amortising the same weight read, typically an order of magnitude
        faster and scaling differently with depth.
        """
        if n_tokens <= 0:
            return 0.0
        if not self._prefill:
            # Uncalibrated: refuse to pretend. Charging the full budget makes
            # exactly one minimum-size chunk affordable per round.
            return float(ROUND_LATENCY_BUDGET_MS)
        tps = self._lookup(self._prefill, pos, 1)
        if tps <= 0:
            return float(ROUND_LATENCY_BUDGET_MS)
        return (n_tokens * 1000.0 / tps) * self.correction

    def prefill_tokens_affordable(self, budget_ms, pos=0):
        if budget_ms <= 0:
            return 0
        if not self._prefill:
            # Uncalibrated. Returning 0 here would mean a PENDING session is
            # NEVER prefilled -- the engine would sit with an unanswerable
            # prompt, rescued only after MAX_PREFILL_STALLS rounds by the aging
            # path. One minimum chunk per round is the honest conservative
            # behaviour, and matches what prefill_ms() charges.
            return PREFILL_CHUNK_MIN
        tps = self._lookup(self._prefill, pos, 1)
        if tps <= 0:
            return 0
        return int((budget_ms / 1000.0) * tps / max(self.correction, 1e-9))

    # -- the closed loop ----------------------------------------------------

    def observe_round(self, estimated_ms, observed_ms):
        """Feed the real round time back, so a wrong model degrades not lies.

        This project has already been burned by this exact class of assumption:
        Phase C's naive extrapolation predicted ~41 tok/s where the real joint
        measurement gave 12.5 -- a 3x error in the OPTIMISTIC direction. Nothing
        else stops that recurring here, and it would fail silently, with rounds
        simply running long while the code believed they fit.
        """
        if estimated_ms <= 0 or observed_ms < 0:
            return
        ratio = observed_ms / estimated_ms
        # EWMA so one slow round -- a GC pause, a cpu.max throttle window --
        # cannot swing admission, but sustained divergence does.
        self.correction = 0.9 * self.correction + 0.1 * ratio
        # Floor of 1.0 is deliberate: the loop may only make estimates MORE
        # conservative. An optimistic correction would let a lucky run of fast
        # rounds widen batches until the bound breaks -- the exact failure this
        # defends against. The ceiling stops one pathological measurement from
        # wedging the engine into admitting nobody, ever.
        self.correction = max(1.0, min(self.correction, MAX_CORRECTION))


def build_batch(sessions, cost_curve, foreground_tab_id,
                budget_ms=ROUND_LATENCY_BUDGET_MS):
    """Compose one round by SPENDING A LATENCY BUDGET, not by filling slots.

    Returns (decode_picks, prefill_chunks, estimated_ms).

    §7's original slot-count split could never bind: BATCH_SIZE defaulted to
    n_seq_max and admission is already capped at n_seq_max, so every runnable
    session always fit in one batch. Worse, there is normally exactly ONE
    foreground session, so a 50/50 split left foreground's spare slots to be
    donated to background -- the reverse of the intended priority.

    Batching also does not make mixed-depth sessions cheaper to share a round:
    one batched llama_decode is a single synchronous pass whose cost is the SUM
    of every member's KV-read, not the max. Measured: 4 sessions at depth 1500
    gave 12.5 tok/s aggregate, 3.1 each -- worse than one session's solo 25.1
    at that depth. A shallow foreground session sharing a round with a deep
    background one inherits the deep one's cost.
    """
    runnable = [s for s in sessions
                if s.state == SessionState.GENERATING and not s.cancelled]
    pending = [s for s in sessions
               if s.state == SessionState.PENDING and not s.cancelled]

    # §5f: foreground is ONE tab id, re-read fresh every round, never cached.
    # At most one TAB can be foreground by construction; >1 session only if two
    # extensions share that tab.
    fg = [s for s in runnable if s.key[1] == foreground_tab_id]
    bg = [s for s in runnable if s.key[1] != foreground_tab_id]

    picks, budget = [], float(budget_ms)
    spent = 0.0

    # Foreground first and UNCONDITIONALLY: the first one is admitted even if it
    # alone exceeds the budget, or a deep foreground session could never run.
    for s in fg:
        c = cost_curve.cost_ms_worst(s.pos)
        if not picks or c <= budget:
            picks.append(s); budget -= c; spent += c

    # AGING PASS -- before cheapest-first gets to pass over the same sessions
    # again. Without it a moderately expensive bg session loses to the same
    # cheaper ones every round forever: real starvation, merely moved from
    # fg-vs-bg to bg-vs-bg.
    picked = set(id(s) for s in picks)
    aged = sorted([s for s in bg if s.rounds_excluded >= MAX_CONSECUTIVE_EXCLUSIONS],
                  key=lambda s: -s.rounds_excluded)
    for s in aged:
        c = cost_curve.cost_ms_worst(s.pos)
        # "or not picks" is a CORRECTION to §9a, not a restatement of it. §9a's
        # prose says a deep session "simply runs alone in its own round,
        # spending the whole budget on itself, which harms nobody else" -- but
        # its code gives that unconditional-first-pick escape ONLY to
        # foreground. Measured consequence: a background session costing more
        # than the entire budget (100ms at pos 6000 vs a 50ms round) is excluded
        # EVERY round forever, aging included, because c <= budget can never
        # hold. 60/60 rounds excluded with no foreground even present.
        #
        # Granting aged background the same escape makes the prose true and
        # bounds the overrun to one session's cost in a round where it runs
        # alone -- which is exactly what §9a says should happen.
        if c <= budget or not picks:
            picks.append(s); picked.add(id(s)); budget -= c; spent += c

    # NORMAL FILL -- cheapest-first. Arrival order (FIFO) would let whichever
    # session asked first consume the whole budget alone, stranding several
    # cheap sessions that would collectively have fit.
    remaining = sorted([s for s in bg if id(s) not in picked],
                       key=lambda s: cost_curve.cost_ms_worst(s.pos))
    for s in remaining:
        c = cost_curve.cost_ms_worst(s.pos)
        if c <= budget:
            picks.append(s); picked.add(id(s)); budget -= c; spent += c

    for s in bg:
        s.rounds_excluded = 0 if id(s) in picked else s.rounds_excluded + 1
    for s in fg:
        s.rounds_excluded = 0

    # -- §9b: prefill fills whatever budget is LEFT ------------------------
    # Decode is admitted first on purpose: a token owed to a session already
    # mid-response is more latency-sensitive than starting a new one, and it
    # keeps §9a's foreground guarantee untouched.
    prefill_chunks = []
    pending.sort(key=lambda s: s.key[1] != foreground_tab_id)
    for s in pending:
        remaining_toks = len(s.inbox_tokens) - s.prefill_offset
        if remaining_toks <= 0:
            continue
        affordable = cost_curve.prefill_tokens_affordable(budget, s.pos)
        n = min(remaining_toks, affordable)

        # DIVERGENCE FROM §9b -- it says "if n < PREFILL_CHUNK_MIN and
        # n < remaining: continue  # wait for a rounder budget". Taken
        # literally that STARVES: if decode picks keep leaving less than a
        # minimum chunk's worth of budget, the condition holds every round and
        # the prompt is never prefilled at all. §9b bounds decode starvation
        # with an aging pass but leaves prefill unbounded. Same remedy: after
        # MAX_PREFILL_STALLS skipped rounds, force one minimum-size chunk
        # through even though the budget does not cover it. That overruns the
        # bound by at most one minimum chunk, which is finite and bounded --
        # unlike never answering the user at all.
        if n < PREFILL_CHUNK_MIN and n < remaining_toks:
            s.prefill_stalls += 1
            if s.prefill_stalls < MAX_PREFILL_STALLS:
                continue
            n = min(PREFILL_CHUNK_MIN, remaining_toks)
        if n <= 0:
            continue
        s.prefill_stalls = 0
        chunk = s.inbox_tokens[s.prefill_offset:s.prefill_offset + n]
        prefill_chunks.append((s, chunk))
        c = cost_curve.prefill_ms(n, s.pos)
        budget -= c; spent += c

    return picks, prefill_chunks, spent
