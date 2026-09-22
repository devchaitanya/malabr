"""MALABR inference engine -- the single engine thread. Sections 7-9b.

The engine is split by the design's own build order:
  slots.py      SlotAllocator + wipe-on-acquire        <- isolation
  formatter.py  chat template application, UTF-8 streaming
  session.py    Session, Turn, OutputCap
  scheduler.py  CostCurve, build_batch
  engine.py     Engine: the loop, compaction, the aggregate guard

Every name is re-exported here so `from malabr_service import engine as eng`
still reaches all of them. See design_doc/implementation_notes.md, 'Code
rationale: engine'.
"""

import queue
import sys
import threading
import time

import llama_cpp.llama_cpp as C

from .formatter import (ChatFormatter, TemplateError, Utf8Streamer,  # noqa: F401
                        is_stop_token)
from .protocol import FRAME_COMPLETE, FRAME_ERROR, FRAME_TOKEN
from .scheduler import (CostCurve, MAX_CONSECUTIVE_EXCLUSIONS,  # noqa: F401
                        MAX_CONSECUTIVE_ROUND_ERRORS, MAX_CORRECTION,
                        MAX_PREFILL_STALLS, PREFILL_CHUNK_MIN,
                        ROUND_LATENCY_BUDGET_MS, STALE_VISIBILITY_TIMEOUT,
                        build_batch)
from .session import (ABSOLUTE_CEILING, MAX_OUTBOX_TOKENS, MIN_CAP,  # noqa: F401
                      OutboxFull, OutputCap, RESERVED_FOR_RESPONSE, Session,
                      SessionState, Turn)
from .slots import SlotAllocator, SlotWipeError  # noqa: F401


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

    def __init__(self, ctx, model, vocab, allocator, sampling=None, seed=0,
                 n_ctx=None, n_seq_max=None, shared_kv=False, session_budget=2048,
                 cpu_duty=1.0):
        self._ctx = ctx
        self._model = model
        self._vocab = vocab
        self._alloc = allocator
        self._sampling = sampling if sampling is not None else SAMPLING_CHAT
        self._seed = seed

        # Shared-KV aggregate governance (§8). Off by default: behaviour is
        # exactly the partitioned model. On: one shared pool of n_ctx cells,
        # sessions get `session_budget` as a soft cap, and _relieve_aggregate
        # keeps Sigma(pos) < n_ctx by compacting the largest over-fair-share
        # session first.
        self._shared_kv = shared_kv
        self._n_ctx = n_ctx or (session_budget * (n_seq_max or 8))
        self._n_seq_max = n_seq_max or 8
        self._session_budget = session_budget
        self._fair_share = max(1, self._n_ctx // self._n_seq_max)
        self._agg_margin = 64               # leave a little slack under n_ctx

        # Cooperative CPU throttle (§11): fraction of wall time the engine may
        # spend computing. 1.0 = flat out. See _run.
        self._cpu_duty = min(1.0, max(0.05, cpu_duty))

        self._sessions = {}                 # key -> Session
        self._reg_lock = threading.Lock()   # guards _sessions only
        self._running = False
        self._thread = None
        self.foreground_tab_id = -1         # section 5f: ONE key, not a per-session flag
        # Section 5f degraded mode. If the control connection goes down, the
        # last foreground value is FROZEN, and a frozen key naming a
        # now-hidden tab would grant that tab §9a's unconditional admission
        # every round, forever. After the timeout we degrade to "no
        # foreground" -- every session treated uniformly -- which is the safe
        # direction: it costs priority, it cannot break the latency bound.
        self.control_connected = False
        self.control_lost_at = time.time()
        self.round_errors = 0
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
            s = Session(key, slot, formatter_factory(), output_cap,
                        session_budget=self._session_budget)
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

    def effective_foreground_tab_id(self):
        """The foreground key the scheduler should actually use.

        Read fresh every round, never cached. Returns -1 (no foreground) when
        the control connection has been down longer than
        STALE_VISIBILITY_TIMEOUT, because a stale key is worse than none.
        """
        if not self.control_connected and \
                (time.time() - self.control_lost_at) > STALE_VISIBILITY_TIMEOUT:
            return -1
        return self.foreground_tab_id

    def set_control_connected(self, connected):
        if connected:
            self.control_connected = True
        elif self.control_connected:
            self.control_connected = False
            self.control_lost_at = time.time()

    def all_keys(self):
        with self._reg_lock:
            return [k for k, s in self._sessions.items()
                    if s.state != SessionState.DEAD]

    def keys_for_tab(self, tab_id):
        with self._reg_lock:
            return [k for k, s in self._sessions.items()
                    if k[1] == tab_id and s.state != SessionState.DEAD]

    def keys_for_extension(self, extension_id):
        """Section 10's sweep: tear down EVERY session for one extension.

        Tab-close handles one session at a time; without this, an uninstalled
        extension's sessions would sit holding slots until their tabs happen to
        close, which may be never.
        """
        with self._reg_lock:
            return [k for k, s in self._sessions.items()
                    if k[0] == extension_id and s.state != SessionState.DEAD]

    def wait_for_teardown(self, keys, timeout=2.0):
        """Block until these sessions are actually gone, or the timeout expires.

        Needed because eviction is deliberately ASYNCHRONOUS: cancel() only
        sets a flag, and the engine thread does the KV work, because a
        connection thread must never touch the model (§7). But admission runs
        immediately on the connection thread, so without this the sequence
            evict_other_origins(...) ; get_or_create(...)
        rejects a cross-origin navigation with "no free session slots" while
        the slot it needs is the condemned session's own, one round from being
        released. Measured: with every slot occupied, the new-origin session
        was refused and the slot appeared one round later.

        Blocking a connection thread is fine here -- §7 calls the client pool a
        waiting room, not compute, and it is sized well above n_seq_max.
        """
        if not keys:
            return True
        deadline = time.time() + timeout
        pending = set(keys)
        while time.time() < deadline:
            with self._reg_lock:
                alive = {k for k in pending
                         if k in self._sessions
                         and self._sessions[k].state != SessionState.DEAD}
            if not alive:
                return True
            pending = alive
            time.sleep(0.002)
        return False

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

    def stop_generation(self, key, reason="stopped"):
        """User-initiated stop: end the turn, KEEP the session.

        Distinct from cancel(), which tears the session down for tab close or
        eviction. A stop must leave the conversation intact and reusable -- the
        user wants this answer to end, not the chat to disappear.

        Rolls the partial turn back for the reason section 6b already gives:
        a turn enters permanent context ONLY by reaching EOS or the output cap.
        A stop that left a half-finished assistant turn in the cache would
        desync the model from what the user can see, which is exactly what that
        rule exists to prevent.

        Flag only -- the rollback itself is engine-thread work.
        """
        with self._reg_lock:
            s = self._sessions.get(key)
        if s is None or s.state == SessionState.DEAD:
            return False                    # defined no-op
        if s.state not in (SessionState.PENDING, SessionState.GENERATING):
            return False                    # nothing in flight
        s.stop_requested = reason
        return True

    def submit(self, key, text):
        """Queue a new prompt and return THIS request's outbox.

        Section 6b: replaces any generation in flight. Returns None if there is
        no such session.
        """
        with self._reg_lock:
            s = self._sessions.get(key)
        if s is None or s.state == SessionState.DEAD:
            return None
        # The caller needs its queue reference immediately -- it will start
        # blocking on it before the engine thread reaches _begin_turn.
        outbox = queue.Queue(maxsize=MAX_OUTBOX_TOKENS)
        s.pending_outbox = outbox
        # Do NOT touch pos or KV here -- that is engine-thread work. Only the
        # flag is set; the loop performs the rollback at its next top.
        s.pending_replace = text
        return outbox

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
            t_start = time.perf_counter()
            try:
                did_work = self._step()
            except Exception as exc:
                # An unhandled exception here used to kill the engine thread
                # silently. The socket stays open and the server keeps
                # ACCEPTING, so it looks healthy while doing nothing, and every
                # session hangs until the browser's 60s read timeout. Failing
                # loudly and continuing is strictly better: the round that blew
                # up is lost, the rest of the engine keeps serving.
                self.round_errors += 1
                print(f"malabr: engine round failed ({type(exc).__name__}: {exc})",
                      file=sys.stderr, flush=True)
                if self.round_errors >= MAX_CONSECUTIVE_ROUND_ERRORS:
                    # Something is systematically broken, not transient. Stop
                    # rather than spin at full speed printing forever.
                    print("malabr: too many consecutive engine failures, stopping",
                          file=sys.stderr, flush=True)
                    self._running = False
                    return
                time.sleep(0.01)
                continue
            self.round_errors = 0
            if not did_work:
                time.sleep(0.001)
            elif self._cpu_duty < 1.0:
                # Cooperative CPU throttle. llama_decode pegs n_threads cores for
                # the round's compute; sleeping proportionally afterwards holds
                # the average near n_threads * cpu_duty, smoothly -- no cgroup,
                # no launch-path plumbing, and per-token latency scales by
                # 1/cpu_duty predictably instead of the period-freeze a
                # kernel quota causes.
                busy = time.perf_counter() - t_start
                time.sleep(busy * (1.0 / self._cpu_duty - 1.0))

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

        # Shared-KV aggregate guard, same "before composition" reason. With one
        # shared pool a single session's soft cap can be n_ctx/2, so the sum
        # across sessions can exceed n_ctx -- and then llama_decode fails with
        # "no memory slot". Keep Sigma(pos) under n_ctx by compacting.
        if self._shared_kv:
            self._relieve_aggregate_pressure(sessions)

        decode_picks, prefill_chunks, estimated_ms = build_batch(
            sessions, self.cost_curve, self.effective_foreground_tab_id())
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
            if s.stop_requested is not None:
                reason, s.stop_requested = s.stop_requested, None
                # Same rollback as a supersede, then the session goes idle and
                # stays available -- no teardown, no slot release.
                self._roll_back_partial(s)
                s.terminate(FRAME_ERROR, reason)
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
                try:
                    self._begin_turn(s, text)
                except ValueError as exc:
                    # Gate 2 (the input cap in _begin_turn) fired. It raises on
                    # purpose -- a loud "something upstream is broken" signal --
                    # but the client must still get ONE terminal frame, or the
                    # browser sits until its 60s read timeout and surfaces only
                    # "read failed ... result -7". Convert the raise into that
                    # frame here; the log line above/below keeps the signal. The
                    # session stays alive -- the input was bad, not the session.
                    print(f"malabr: {exc}", file=sys.stderr, flush=True)
                    s.terminate(FRAME_ERROR, str(exc))
                    s.state = SessionState.IDLE
                except TemplateError as exc:
                    # The formatter desynced from the KV (its _rendered is no
                    # longer a prefix of the freshly rendered conversation).
                    # Retrying cannot fix this -- every subsequent turn hits the
                    # same wall -- so end the session cleanly instead of failing
                    # the round forever.
                    print(f"malabr: formatter desync, ending session: {exc}",
                          file=sys.stderr, flush=True)
                    s.terminate(FRAME_ERROR,
                                "conversation state was lost -- start a new chat")
                    self._teardown(s, "formatter desync")

    def _begin_turn(self, s, text):
        # Swap in the queue this request's handler is already holding. Done here
        # on the engine thread, AFTER any superseded terminal frame has been
        # written to the OLD queue, so the two never mix.
        if s.pending_outbox is not None:
            s.outbox = s.pending_outbox
            s.pending_outbox = None
        s.pos_before_request = s.pos            # snapshot 1 (section 6b)
        # Snapshot 2 is set for real when prefill completes (see _run_round). Pin
        # it to the request boundary until then: left holding the PREVIOUS turn's
        # value it points inside an already-closed exchange, and if the shared-KV
        # aggregate guard's compact(keep_recent=False) drops that exchange while
        # this turn is still PENDING, the stale position lands inside the dropped
        # range and trips compact()'s corruption assert -- wedging the round.
        s.pos_before_generation = s.pos
        s.formatter_cp_before_request = s.formatter.checkpoint()
        s.inbox_tokens = s.formatter.user_turn(text)
        # Section 8 gate 2: defense in depth at the point of no return. Gate 1
        # lives upstream at admission; this one is structurally unskippable,
        # because no turn can start without passing through here. It RAISES
        # rather than tolerating the oversized input -- a loud "something
        # upstream is broken" signal. _apply_control catches the raise and
        # turns it into the client's terminal frame, so the browser still gets
        # a clean "prompt is too long" instead of a 60s read timeout.
        max_input = s.budget - RESERVED_FOR_RESPONSE
        if self._shared_kv:
            # Shared-KV: s.budget is a soft cap. The real limit is the shared
            # pool minus what the OTHER sessions are entitled to keep -- their
            # fair share -- NOT their current size: _relieve_aggregate_pressure
            # will compact any over-fair-share session down when this turn runs.
            # Reserving their live pos instead starved a 3rd tab the moment two
            # others got deep.
            with self._reg_lock:
                reserved = sum(min(o.pos, self._fair_share)
                               for o in self._sessions.values()
                               if o is not s and o.state != SessionState.DEAD)
            room = self._n_ctx - reserved - self._agg_margin
            max_input = min(max_input, room - RESERVED_FOR_RESPONSE)
        if len(s.inbox_tokens) > max_input:
            n_tokens = len(s.inbox_tokens)
            s.formatter.restore(s.formatter_cp_before_request)
            s.formatter_cp_before_request = None
            s.state = SessionState.IDLE
            s.inbox_tokens = []
            raise ValueError(
                f"gate 1 bypassed: prompt is too long -- {n_tokens} tokens, "
                f"but this conversation can accept at most {max(0, max_input)}")
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
        # No turn is in flight now -- keep pos_before_generation pinned to the
        # request boundary so it never lingers inside a compactable exchange
        # (see _finish_turn for the same reasoning).
        s.pos_before_generation = s.pos_before_request
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
        reply = bytes(s._reply_bytes).decode("utf-8", "replace")

        # Chat templates that trim message content (gemma-3 does: `content |
        # trim`) drop trailing whitespace the model emitted -- most often when
        # the OUTPUT CAP cuts a reply mid-flow right after a space or newline.
        # Left in the KV, those tokens make the formatter's _rendered (trimmed
        # to match the template) and the KV disagree, and the NEXT turn dies on
        # the prefix check. Remove them so KV == _rendered.
        stripped = reply.rstrip()
        if stripped != reply:
            # Only the tokens generated THIS turn are candidates -- never the
            # user message or the generation prompt ahead of them.
            reply_span = s.pos - s.pos_before_generation
            if not stripped:
                # The reply is ALL whitespace: the template renders it as an
                # empty assistant message, so every generated token has to leave
                # the KV. The old `stripped and` guard skipped this case, and
                # the desync it left killed the session one turn later.
                n_trim = reply_span
            else:
                try:
                    full_toks = s.formatter._tokenize(reply, parse_special=False)
                    keep_toks = s.formatter._tokenize(stripped, parse_special=False)
                except Exception:
                    full_toks = keep_toks = None
                # Re-tokenising the concatenated bytes can disagree with the
                # tokens the model actually sampled where the last real token
                # touches the whitespace. Trust the diff ONLY when the stripped
                # tokenisation is a clean prefix of the full one; otherwise a
                # count would cut into content, so leave the KV alone.
                if full_toks is not None and full_toks[:len(keep_toks)] == keep_toks:
                    n_trim = len(full_toks) - len(keep_toks)
                else:
                    n_trim = 0
            if 0 < n_trim <= reply_span:
                C.llama_memory_seq_rm(self._alloc._mem, s.slot,
                                      s.pos - n_trim, s.pos)
                s.pos -= n_trim
                reply = stripped

        # BOOKKEEPING FIRST -- before any emit that can raise OutboxFull on a
        # stalled reader. If the formatter update below is skipped by such a
        # raise, the NEXT turn fails the prefix check one turn later, silently.
        # The Turn's msg_index must be read before assistant_generated() appends
        # the assistant message.
        s.turn_boundaries.append(
            Turn(s.pos_before_request, s.pos, "exchange",
                 len(s.formatter._messages)))   # index of the user msg
        s.formatter.assistant_generated(reply)

        # Now the fallible part. A full outbox here must NOT prevent the
        # terminal frame -- terminate() bypasses the bound for exactly that.
        try:
            tail = s.streamer.flush()
            if tail:
                s.emit(FRAME_TOKEN, tail)
            # The C++ discards the FRAME_COMPLETE payload, so "why did it stop"
            # is invisible to the client. On the output cap (not EOS) send a
            # marker token first -- the panel strips it and shows a "hit the
            # length limit" note instead of a reply that just trails off.
            if frame_type == FRAME_COMPLETE and reason == "cap":
                s.emit(FRAME_TOKEN, "\x00MALABR:cap")
        except OutboxFull:
            pass
        print(f"malabr:   turn done ({reason}) produced~{len(s._reply_bytes)}B "
              f"pos={s.pos}", file=sys.stderr, flush=True)
        s.terminate(frame_type, reason)
        s.state = SessionState.IDLE
        s.produced = 0
        # The turn is committed; pos_before_generation is no longer an in-flight
        # marker. Left holding this turn's generation-start position it points
        # INSIDE the exchange just recorded in turn_boundaries -- and the
        # shared-KV aggregate guard compacts IDLE sessions too, so its
        # compact(keep_recent=False) last resort would drop that exchange and
        # trip compact()'s corruption assert. Pin it back to the request
        # boundary; _begin_turn / prefill completion set it afresh next turn.
        s.pos_before_generation = s.pos_before_request

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

    def compact(self, s, keep_recent=True):
        """Drop whole oldest exchanges, shifting EVERY live absolute position.

        Returns tokens freed. False-y result means nothing was droppable, which
        section 8 handles upstream by rejecting oversized input rather than
        letting compaction fail and deciding afterwards.

        keep_recent=False is the shared-KV aggregate guard's last resort: drop
        every exchange except the anchor, so a pool that would otherwise fail a
        decode ("no memory slot") still makes room. Costs the most context.
        """
        # Keep the anchor (index 0); normally keep the most recent exchange too.
        # The anchor carries the template preamble and the first ~32 tokens act
        # as attention anchors whose loss degrades output sharply.
        droppable = s.turn_boundaries[1:-1] if keep_recent else s.turn_boundaries[1:]
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
            # The live rollback checkpoint is ALSO a reference into the message
            # list, and it goes stale here exactly like a position does. Missing
            # this made a replace-after-compaction restore to a message count
            # that no longer existed, and the formatter's own prefix check
            # caught it as "template broke the prefix property".
            if s.formatter_cp_before_request is not None \
                    and s.formatter_cp_before_request > turn.msg_index:
                s.formatter_cp_before_request -= 2

            s.turn_boundaries.remove(turn)
            freed += n
        return freed

    def _relieve_aggregate_pressure(self, sessions):
        """Shared-KV: keep Sigma(live pos) under n_ctx by compacting.

        Victim = the largest session ABOVE its fair share (n_ctx/n_seq_max), so
        a small or foreground session is never shrunk to feed a greedy one. The
        fair shares sum to exactly n_ctx, so an over-limit total always has such
        a session -- unless several are stuck at the anchor+one-turn floor, in
        which case fall back to the largest droppable one so a decode still
        cannot hit "no memory slot".
        """
        live = [s for s in sessions if s.state != SessionState.DEAD]
        if not live:
            return
        limit = self._n_ctx - self._agg_margin
        if sum(s.pos for s in live) < limit:
            return
        before = sum(s.pos for s in live)
        guard = 0
        while sum(s.pos for s in live) >= limit and guard < 3 * len(live) + 4:
            guard += 1
            over = sorted((s for s in live if s.pos > self._fair_share),
                          key=lambda s: s.pos, reverse=True)
            pool = over or sorted(live, key=lambda s: s.pos, reverse=True)
            freed = 0
            for victim in pool:
                freed = self.compact(victim)          # keep the recent turn
                if not freed:
                    freed = self.compact(victim, keep_recent=False)   # last resort
                if freed:
                    break
            if not freed:
                break                                # compaction is spent

        remaining = sum(s.pos for s in live)
        if remaining >= limit:
            # Compaction could not claw the pool back under n_ctx: the iteration
            # bound was hit, or every session is down to its anchor. Returning
            # here would leave the next llama_decode to fail "no memory slot"
            # and the round to enter the failure-retry loop. Instead reject the
            # turns that have produced NOTHING yet -- a PENDING session rolls
            # back cleanly and its client gets one honest "at capacity" frame.
            rejected = self._reject_pending_for_capacity(live, limit)
            remaining = sum(s.pos for s in live)
            if remaining >= limit:
                print("malabr: aggregate KV pressure unrelievable "
                      f"(sum pos={remaining}, n_ctx={self._n_ctx}, "
                      f"rejected {rejected} pending)",
                      file=sys.stderr, flush=True)
                return
        print(f"malabr: aggregate guard compacted {before} -> {remaining} "
              f"(limit {limit})", file=sys.stderr, flush=True)

    def _reject_pending_for_capacity(self, live, limit):
        """Roll back PENDING turns, largest first, until the shared pool is back
        under n_ctx. They have generated nothing, so a rollback plus one
        FRAME_ERROR is clean and recoverable -- letting the decode fail is not.
        GENERATING sessions are left alone: killing a stream the user is
        watching to make room for someone else's turn is the worse trade.
        """
        pending = sorted((s for s in live if s.state == SessionState.PENDING),
                         key=lambda s: s.pos, reverse=True)
        rejected = 0
        for s in pending:
            if sum(o.pos for o in live) < limit:
                break
            self._roll_back_partial(s)
            s.terminate(FRAME_ERROR, "server is at capacity -- try again")
            rejected += 1
        return rejected

    def _token_bytes(self, tok):
        import ctypes
        buf = (ctypes.c_char * 256)()
        n = C.llama_token_to_piece(self._vocab, tok, buf, 256, 0, True)
        if n < 0:
            raise RuntimeError(f"llama_token_to_piece failed ({n})")
        return buf.raw[:n]
