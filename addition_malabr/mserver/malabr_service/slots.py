"""KV slot allocation with wipe-on-acquire -- design section 7.

See design_doc/implementation_notes.md, 'Code rationale: slots'."""

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
