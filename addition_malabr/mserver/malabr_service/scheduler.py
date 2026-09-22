"""Batch composition by latency budget, and the closed loop -- sections 9a, 9b.

See design_doc/implementation_notes.md, 'Code rationale: scheduler'."""

from .session import SessionState


ROUND_LATENCY_BUDGET_MS = 50      # the bound §7 already commits to elsewhere
MAX_CONSECUTIVE_EXCLUSIONS = 5    # aging bound; no measured value yet
PREFILL_CHUNK_MIN = 32            # below this, per-call overhead dominates
MAX_PREFILL_STALLS = 5            # see build_batch -- not in §9b, see note there
MAX_CORRECTION = 8.0              # §9b leaves this unnamed; see observe_round

# §5f degraded mode: how long a dead control connection may go before the
# foreground key is treated as stale and the engine falls back to uniform
# treatment.
STALE_VISIBILITY_TIMEOUT = 10.0

# Consecutive failed engine rounds before giving up. Transient failures are
# survivable; a systematic one should stop rather than spin.
MAX_CONSECUTIVE_ROUND_ERRORS = 20


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
    #
    # Ordered by rounds_excluded, for the two-extensions-on-one-tab case where
    # fg has two members. Without this the SAME one always took the
    # unconditional pick, and if it alone blew the budget the other could never
    # satisfy c <= budget -- and was never aged either, because the aging pass
    # below only looked at bg and rounds_excluded was reset for every fg
    # session whether it ran or not. Permanent starvation, the same class §9a's
    # aging fixes for bg-vs-bg. Giving the most-passed-over fg session the
    # unconditional pick makes the two alternate instead.
    for s in sorted(fg, key=lambda s: -s.rounds_excluded):
        c = cost_curve.cost_ms_worst(s.pos)
        if not picks or c <= budget:
            picks.append(s); budget -= c; spent += c

    # AGING PASS -- before cheapest-first gets to pass over the same sessions
    # again. Without it a moderately expensive bg session loses to the same
    # cheaper ones every round forever: real starvation, merely moved from
    # fg-vs-bg to bg-vs-bg. Drawn from every unpicked runnable session, so a
    # passed-over foreground member gets the same escape hatch.
    picked = set(id(s) for s in picks)
    aged = sorted([s for s in runnable
                   if id(s) not in picked and s.rounds_excluded >= MAX_CONSECUTIVE_EXCLUSIONS],
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
    remaining = sorted(((cost_curve.cost_ms_worst(s.pos), s)
                        for s in runnable if id(s) not in picked),
                       key=lambda cs: cs[0])
    for c, s in remaining:
        if c <= budget:
            picks.append(s); picked.add(id(s)); budget -= c; spent += c

    # Counted for foreground too: an fg member that lost this round must age
    # like anyone else, or the alternation above never triggers.
    for s in runnable:
        s.rounds_excluded = 0 if id(s) in picked else s.rounds_excluded + 1

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
            # Admit what the budget CAN afford, not a full minimum chunk.
            # Forcing PREFILL_CHUNK_MIN through was measured to overrun by
            # 87ms against a 50ms budget, in 20% of rounds -- it broke the
            # very bound §9a/§6a exist to hold, to fix a starvation problem.
            # PREFILL_CHUNK_MIN is an EFFICIENCY floor ("below this, per-call
            # overhead dominates"), not a correctness one, so trading a little
            # efficiency to keep the latency bound is the right way round.
            n = max(1, min(affordable, remaining_toks))
        if n <= 0:
            continue
        s.prefill_stalls = 0
        chunk = s.inbox_tokens[s.prefill_offset:s.prefill_offset + n]
        prefill_chunks.append((s, chunk))
        c = cost_curve.prefill_ms(n, s.pos)
        budget -= c; spent += c

    return picks, prefill_chunks, spent
