"""Per-session state, output cap and turn bookkeeping -- sections 6, 6b, 8.

See design_doc/implementation_notes.md, 'Code rationale: session'."""

import queue

from .formatter import Utf8Streamer
from .protocol import FRAME_COMPLETE, FRAME_ERROR, FRAME_TOKEN


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

# Exactly one TERMINAL frame (COMPLETE or ERROR) ends every request -- see
# terminate() for why that is unconditional rather than best-effort.


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

    def __init__(self, key, slot, formatter, output_cap, session_budget=2048):
        self.key = key
        self.slot = slot
        self.formatter = formatter
        self.output_cap = output_cap

        self.state = SessionState.IDLE
        self.pos = 0                      # absolute KV position
        self.produced = 0                 # tokens emitted this response
        self.inbox_tokens = []            # prompt tokens awaiting prefill
        self.prefill_offset = 0
        # PER-REQUEST, not per-session -- a correction to §7, which calls this
        # "the session's outbox". Under §6b's single-flight replace TWO handlers
        # are briefly alive: the superseded one waiting for its terminal frame,
        # and the new one waiting for tokens. One shared queue means whichever
        # polls first steals the other's frames -- the superseded client could
        # receive the new response, or hang until its 60s read timeout. Each
        # request gets its own queue; the engine writes to whichever is current.
        self.outbox = queue.Queue(maxsize=MAX_OUTBOX_TOKENS)
        self.pending_outbox = None
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
        # Partitioned KV: the hard slice n_ctx/n_seq_max. Shared KV: a SOFTER
        # cap (n_ctx/2 by default) -- the engine's aggregate guard is the real
        # bound. Passed in from config so it tracks n_ctx / n_seq_max instead of
        # a constant that silently goes stale when either changes.
        self.budget = session_budget

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
        # Set by stop_generation(). Unlike `cancelled` this ends the TURN, not
        # the session.
        self.stop_requested = None

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
