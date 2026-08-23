"""Startup calibration -- see design_doc/phase1_design.md section 11a.

Measures this machine rather than trusting numbers measured on another one.
Every throughput figure in the design document was taken on one 8-logical /
4-physical, 14 GB machine and does not transfer.

The phases, and what each exists to defend against:
  A  thread sizing            -- more threads than the quota allows is slower
  B  position-cost curve      -- decode collapses with depth (48 -> 5 tok/s)
  B-prefill  prefill curve    -- prefill is compute-bound, NOT the decode curve
  C  joint worst case         -- per-session cost does not predict joint cost
  D  finished tables + gate   -- engine never re-derives on the hot path
  E  atomic store + fallback  -- an interrupted shutdown must not corrupt it
  F  clamps on OWN output     -- calibration succeeding but returning nonsense
  G  absolute ceiling         -- a corrupted curve must not lift the cap

ONE curve captures both memory and CPU. Every decode step attends to every
prior token; reading that token's K/V is what costs the bandwidth AND the
space. They are two costs of the same growing quantity, position -- not two
things that happen to correlate.
"""

import json
import os
import statistics
import time

import llama_cpp.llama_cpp as C

from .config import MAX_N_CTX, MIN_N_CTX, QUOTA_PCT, detect_physical_cores

CALIBRATION_VERSION = 2

# Phase D: no single response should take longer than this to generate.
TARGET_MAX_RESPONSE_SECONDS = 15

# Phase C/D gate: below this, the configuration will throttle badly under
# ordinary multi-tab use and n_seq_max or n_ctx should come down first.
MIN_ACCEPTABLE_AGGREGATE_TPS = 4.0

# Section 11a measured a real 19% trial-to-trial spread at short context even
# after taking a median of 3. Three is the floor, not a comfortable margin.
TRIALS_PER_POINT = 3


class CalibrationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# low-level helpers -- every return code checked
# ---------------------------------------------------------------------------

def _decode(ctx, tokens, pos0, seq_ids, want_last_logits=True):
    """Decode `tokens` into every sequence in seq_ids at pos0..

    Return codes are checked, not assumed. Section 11a records two separate
    occasions where an unchecked llama_decode return produced a physically
    impossible throughput number (41323 tok/s once, a negative switch cost
    another time). A silent failure here would make calibration confidently
    wrong, which is worse than calibration failing.
    """
    n = len(tokens) * len(seq_ids)
    batch = C.llama_batch_init(n, 0, 1)
    try:
        i = 0
        for seq in seq_ids:
            for j, tok in enumerate(tokens):
                batch.token[i] = tok
                batch.pos[i] = pos0 + j
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = seq
                batch.logits[i] = 1 if (want_last_logits and j == len(tokens) - 1) else 0
                i += 1
        batch.n_tokens = n
        rc = C.llama_decode(ctx, batch)
        if rc != 0:
            raise CalibrationError(f"llama_decode failed rc={rc}")
    finally:
        C.llama_batch_free(batch)


def _fill_to(ctx, vocab, seq, target_pos, chunk=256):
    """Grow one sequence to target_pos with filler tokens."""
    C.llama_memory_seq_rm(C.llama_get_memory(ctx), seq, -1, -1)
    pos = 0
    filler = [_safe_token(vocab)] * chunk
    while pos < target_pos:
        n = min(chunk, target_pos - pos)
        _decode(ctx, filler[:n], pos, [seq], want_last_logits=(pos + n >= target_pos))
        pos += n
    return pos


_SAFE_TOKEN = None


def _safe_token(vocab):
    """A single ordinary token to use as filler. Cached."""
    global _SAFE_TOKEN
    if _SAFE_TOKEN is None:
        buf = (C.llama_token * 8)()
        n = C.llama_tokenize(vocab, b" the", 4, buf, 8, False, False)
        if n <= 0:
            raise CalibrationError("could not tokenize filler")
        _SAFE_TOKEN = int(buf[0])
    return _SAFE_TOKEN


def _time_decode_tokens(ctx, vocab, seq, pos, n_tokens):
    """Time n_tokens of single-token decode at `pos`. Returns tok/s."""
    tok = _safe_token(vocab)
    t0 = time.perf_counter()
    for k in range(n_tokens):
        _decode(ctx, [tok], pos + k, [seq])
    dt = time.perf_counter() - t0
    if dt <= 0:
        raise CalibrationError("non-positive elapsed time")
    return n_tokens / dt


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_a_threads(model_path, candidates=None, probe_tokens=24, n_ctx=1024):
    """Sweep n_threads and pick the measured optimum.

    NOT the physical core count. Measured on the reference machine: 8 threads
    is SLOWER than 4 (hyperthread contention, not parallelism), and the
    deployed value must additionally match the cpu.max quota -- setting one
    without the other is the nonlinear-loss regime section 11 measured.
    """
    cores = detect_physical_cores()
    if candidates is None:
        candidates = sorted({1, 2, 4, cores})
    results = {}
    for n in candidates:
        if n < 1:
            continue
        mp = C.llama_model_default_params(); mp.n_gpu_layers = 0
        model = C.llama_model_load_from_file(model_path.encode(), mp)
        if not model:
            raise CalibrationError(f"could not load model at {model_path}")
        try:
            cp = C.llama_context_default_params()
            cp.n_ctx, cp.n_seq_max, cp.n_batch, cp.n_threads = n_ctx, 1, 512, n
            ctx = C.llama_init_from_model(model, cp)
            if not ctx:
                raise CalibrationError("context creation failed")
            try:
                vocab = C.llama_model_get_vocab(model)
                _fill_to(ctx, vocab, 0, 32)
                results[n] = _time_decode_tokens(ctx, vocab, 0, 32, probe_tokens)
            finally:
                C.llama_free(ctx)
        finally:
            C.llama_model_free(model)
    if not results:
        raise CalibrationError("no thread candidates measured")
    best = max(results, key=results.get)
    # The deployed value is capped by the quota regardless of what measured
    # fastest: an unthrottled optimum is not achievable under the runtime quota.
    quota_threads = max(1, round(cores * QUOTA_PCT / 100))
    return {"measured": {str(k): v for k, v in results.items()},
            "fastest": best,
            "chosen": min(best, quota_threads),
            "quota_threads": quota_threads,
            "physical_cores": cores}


def phase_b_position_curve(ctx, vocab, positions, trials=TRIALS_PER_POINT,
                           probe_tokens=16, seq=0):
    """Decode tok/s at several depths. Emits BOTH statistics.

    Section 11a's rule, and the reason there are two: the median is
    representative and feeds the output-cap formula; the WORST observed trial
    feeds anything that is a hard cutoff, because the spread is real (19% at
    short context, measured, not hypothetical). Using the median for admission
    means a round estimated at 45ms can genuinely take 54ms.

    Piecewise interpolation between real points, deliberately -- section 11a
    tried fitting the closed-form "fixed weight-read + linear KV-read" model
    and it does NOT hold (the slope of 1/tps is not monotonic in position).
    """
    rows = []
    mem = C.llama_get_memory(ctx)
    for pos in positions:
        _fill_to(ctx, vocab, seq, pos)
        samples = []
        for _ in range(trials):
            # Truncate back to `pos` before EVERY trial. llama.cpp requires
            # sequence positions to stay consecutive: the previous trial left
            # the cache at pos+probe_tokens, so starting the next one at `pos`
            # again fails with "it is required that the sequence positions
            # remain consecutive". Cheaper than re-filling from zero.
            C.llama_memory_seq_rm(mem, seq, pos, -1)
            samples.append(_time_decode_tokens(ctx, vocab, seq, pos, probe_tokens))
        rows.append([int(pos),
                     float(statistics.median(samples)),
                     float(min(samples))])          # worst observed
    rows.sort()
    return rows


def phase_b_prefill_curve(ctx, vocab, sizes, depths, trials=2, seq=1):
    """Prompt-processing throughput. A SEPARATE curve, not the decode one.

    Decode is memory-bandwidth-bound: one token, but every prior K/V vector is
    read. Prefill is compute-bound and parallel: many tokens in one pass
    amortising the same weight read. Reusing the decode curve for prefill
    admission would be wrong by roughly an order of magnitude.

    Worst-observed, not median -- this feeds an admission cutoff (section 9b).
    """
    tok = _safe_token(vocab)
    rows = []
    for depth in depths:
        best_tps = None
        for size in sizes:
            for _ in range(trials):
                _fill_to(ctx, vocab, seq, depth)
                t0 = time.perf_counter()
                _decode(ctx, [tok] * size, depth, [seq])
                dt = time.perf_counter() - t0
                if dt <= 0:
                    continue
                tps = size / dt
                best_tps = tps if best_tps is None else min(best_tps, tps)
        if best_tps is not None:
            rows.append([int(depth), float(best_tps)])
    rows.sort()
    return rows


def phase_c_joint_worst_case(ctx, vocab, n_seq_max, depth, probe_tokens=8):
    """Every slot full, every session at depth, batched, under the real quota.

    A REAL measurement, not an extrapolation from Phase B. Section 11a proved
    extrapolation unreliable here, not merely theoretically risky: the naive
    prediction was ~41 tok/s where the joint measurement gave 12.5 -- a 3x
    error in the OPTIMISTIC direction. Per-session cost genuinely does not
    predict joint cost, because the KV-read terms stack rather than share.
    """
    seqs = list(range(n_seq_max))
    for s in seqs:
        _fill_to(ctx, vocab, s, depth)
    tok = _safe_token(vocab)
    t0 = time.perf_counter()
    for k in range(probe_tokens):
        _decode(ctx, [tok], depth + k, seqs)
    dt = time.perf_counter() - t0
    if dt <= 0:
        raise CalibrationError("non-positive elapsed time in phase C")
    total = probe_tokens * len(seqs)
    return {"depth": int(depth), "n_seq_max": int(n_seq_max),
            "aggregate_tps": total / dt,
            "per_session_tps": (total / dt) / len(seqs)}


# ---------------------------------------------------------------------------
# Phase D -- finished tables and the validation gate
# ---------------------------------------------------------------------------

def _interpolate(rows, pos, col):
    lo = rows[0]
    for row in rows:
        if row[0] <= pos:
            lo = row
        else:
            span = row[0] - lo[0]
            if span <= 0:
                return lo[col]
            f = (pos - lo[0]) / span
            return lo[col] + f * (row[col] - lo[col])
    return lo[col]


def derive_output_cap_table(decode_rows, min_cap, absolute_ceiling,
                            breakpoints=None):
    """FINISHED lookup table, not raw tok/s for someone else to interpret.

    Section 12 test 31 checks exactly this: the engine reads the table and
    never recomputes a cap from raw curve data on a request path.

    Uses the MEDIAN column -- this is the representative estimate, not a
    safety cutoff.
    """
    if breakpoints is None:
        breakpoints = [r[0] for r in decode_rows]
    table = []
    for pos in breakpoints:
        tps = _interpolate(decode_rows, pos, 1)
        cap = TARGET_MAX_RESPONSE_SECONDS * tps
        # Phase G's ceiling applies HERE and again at lookup time in the
        # engine. Two gates, because a table can also be hand-edited.
        table.append([int(pos), int(max(min_cap, min(cap, absolute_ceiling)))])
    return table


def phase_d_assemble(threads, decode_rows, prefill_rows, joint, n_ctx,
                     n_seq_max, min_cap, absolute_ceiling):
    """Assemble outputs and run the pass/fail gate."""
    warnings = []
    passed = True
    if joint and joint["aggregate_tps"] < MIN_ACCEPTABLE_AGGREGATE_TPS:
        passed = False
        warnings.append(
            f"joint worst case {joint['aggregate_tps']:.1f} tok/s is below the "
            f"{MIN_ACCEPTABLE_AGGREGATE_TPS} tok/s service floor; reduce "
            f"n_seq_max or n_ctx before starting")
    if len(decode_rows) < 2:
        warnings.append("fewer than 2 curve points; interpolation is degenerate")

    return {
        "version": CALIBRATION_VERSION,
        "measured_at": time.time(),
        "passed": passed,
        "warnings": warnings,
        "n_threads": threads["chosen"],
        "n_threads_detail": threads,
        "n_ctx": int(n_ctx),
        "n_seq_max": int(n_seq_max),
        "quota_pct": QUOTA_PCT,
        # [pos, tps_median, tps_worst] -- exactly the shape CostCurve wants
        "decode_curve": decode_rows,
        # [pos, tok_per_s_worst]
        "prefill_curve": prefill_rows,
        "joint_worst_case": joint,
        "output_cap_table": derive_output_cap_table(
            decode_rows, min_cap, absolute_ceiling),
    }


# ---------------------------------------------------------------------------
# Phase F -- clamps on calibration's OWN output
# ---------------------------------------------------------------------------

def phase_f_clamp(result, min_cap, absolute_ceiling):
    """Distinct from Phase E, which covers calibration ERRORING.

    This covers calibration SUCCEEDING and returning nonsense from an internal
    bug -- n_threads=0, an absurd n_ctx, a negative tok/s. Different failure
    class, different defense. Converts "trust calibration got it right" into
    "trust calibration to optimise within bounds that hold even when it is
    wrong", which is the property actually wanted.
    """
    cores = detect_physical_cores()
    result["n_threads"] = max(1, min(int(result.get("n_threads") or 1), cores))
    n_ctx = int(result.get("n_ctx") or MIN_N_CTX)
    result["n_ctx"] = max(MIN_N_CTX, min(n_ctx, MAX_N_CTX))
    result["n_seq_max"] = max(1, int(result.get("n_seq_max") or 1))

    clean = []
    for row in result.get("decode_curve") or []:
        try:
            pos, med, worst = int(row[0]), float(row[1]), float(row[2])
        except (TypeError, ValueError, IndexError):
            continue
        if pos < 0 or med <= 0 or worst <= 0:
            continue
        # Worst can never be better than median; if it is, the two were
        # swapped or miscomputed somewhere upstream.
        clean.append([pos, med, min(worst, med)])
    result["decode_curve"] = sorted(clean)

    pre = []
    for row in result.get("prefill_curve") or []:
        try:
            pos, tps = int(row[0]), float(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        if pos >= 0 and tps > 0:
            pre.append([pos, tps])
    result["prefill_curve"] = sorted(pre)

    capped = []
    for row in result.get("output_cap_table") or []:
        try:
            pos, cap = int(row[0]), int(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        # Phase G, applied again: the ceiling holds whatever the table says.
        capped.append([max(0, pos), max(min_cap, min(cap, absolute_ceiling))])
    result["output_cap_table"] = sorted(capped)
    return result


# ---------------------------------------------------------------------------
# Phase E -- atomic store, and a fallback that treats corrupt as missing
# ---------------------------------------------------------------------------

def save(result, path):
    """Write via temp file + rename. Atomic on the same filesystem.

    Section 11a traced the real shutdown path: MalabrManager::StopMLServer
    calls Terminate(0, false) -- SIGTERM, with wait=false, so the browser does
    NOT confirm our handler finished. If the OS force-kills during a fast
    logout mid-write, a plain write leaves a truncated file. This is the only
    thing the design persists at all, so it is exactly the operation that race
    can catch.
    """
    tmp = f"{path}.{os.getpid()}.tmp"
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())          # rename is atomic; the CONTENT still
                                       # has to be on disk before it happens
    os.replace(tmp, path)
    return path


def load(path):
    """Return a stored result, or None.

    "Unparseable" is treated exactly like "missing" -- not as an error. A
    truncated file from an interrupted shutdown must fall back to fresh
    calibration, not crash the server on startup.
    """
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("version") != CALIBRATION_VERSION:
        return None
    if not data.get("decode_curve"):
        return None
    return data


def conservative_fallback(n_ctx, n_seq_max, min_cap, absolute_ceiling):
    """Phase E: what to use when calibration itself fails.

    Deliberately pessimistic rather than a guess at typical hardware. Being
    too conservative costs throughput; being optimistic breaks the latency
    bound the whole scheduler exists to hold.
    """
    return {
        "version": CALIBRATION_VERSION,
        "measured_at": time.time(),
        "passed": False,
        "warnings": ["calibration failed; conservative fallback in use"],
        "n_threads": 1,
        # Same SHAPE as a measured result, deliberately. A fallback that omits
        # keys makes every consumer crash on the one path where things are
        # already going wrong.
        "n_threads_detail": {"measured": {}, "fastest": 1, "chosen": 1,
                             "quota_threads": 1,
                             "physical_cores": detect_physical_cores()},
        "n_ctx": int(n_ctx),
        "n_seq_max": int(n_seq_max),
        "quota_pct": QUOTA_PCT,
        "decode_curve": [[0, 10.0, 5.0], [2048, 5.0, 2.5]],
        "prefill_curve": [[0, 200.0]],
        "joint_worst_case": None,
        "output_cap_table": [[0, min_cap]],
        "fallback": True,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(cfg, min_cap, absolute_ceiling, quick=False, ctx=None, model=None):
    """Run every phase and return a clamped, ready-to-use result.

    Never raises: a calibration failure falls back (Phase E) rather than
    stopping the server. Section 11c's audit is explicit that isolation and
    browser-protection do NOT depend on calibration -- only conversation
    length and speed do -- so a fallback degrades quality, not safety.
    """
    budget = cfg.n_ctx_per_session
    if quick:
        positions = [32, min(256, budget - 1)]
        prefill_sizes, prefill_depths = [32], [32]
        joint_depth = 64
        trials = 1
    else:
        # Span the intended per-session range, 5-6 points, per section 11a.
        positions = sorted({32, budget // 8, budget // 4, budget // 2,
                            int(budget * 0.75), budget - 64})
        positions = [p for p in positions if 0 < p < budget]
        prefill_sizes, prefill_depths = [64, 256], [32, budget // 2]
        joint_depth = min(budget // 2, 1500)
        trials = TRIALS_PER_POINT

    owns = ctx is None
    try:
        threads = (phase_a_threads(cfg.model_path)
                   if not quick else
                   {"measured": {}, "fastest": cfg.n_threads,
                    "chosen": cfg.n_threads,
                    "quota_threads": cfg.n_threads,
                    "physical_cores": detect_physical_cores()})

        if owns:
            mp = C.llama_model_default_params(); mp.n_gpu_layers = 0
            model = C.llama_model_load_from_file(cfg.model_path.encode(), mp)
            if not model:
                raise CalibrationError("model load failed")
            cp = C.llama_context_default_params()
            cp.n_ctx = cfg.n_ctx
            cp.n_seq_max = cfg.n_seq_max
            cp.n_batch = cfg.n_batch
            cp.n_threads = threads["chosen"]
            ctx = C.llama_init_from_model(model, cp)
            if not ctx:
                raise CalibrationError("context creation failed")
        vocab = C.llama_model_get_vocab(model)

        decode_rows = phase_b_position_curve(ctx, vocab, positions, trials=trials)
        prefill_rows = phase_b_prefill_curve(ctx, vocab, prefill_sizes,
                                             prefill_depths, trials=1)
        joint = phase_c_joint_worst_case(ctx, vocab, cfg.n_seq_max, joint_depth)

        result = phase_d_assemble(threads, decode_rows, prefill_rows, joint,
                                  cfg.n_ctx, cfg.n_seq_max, min_cap,
                                  absolute_ceiling)
    except Exception as exc:                      # Phase E
        result = conservative_fallback(cfg.n_ctx, cfg.n_seq_max, min_cap,
                                       absolute_ceiling)
        result["warnings"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if owns:
            if ctx:
                C.llama_free(ctx)
            if model:
                C.llama_model_free(model)

    # Phase F always runs, on measured AND fallback results alike.
    return phase_f_clamp(result, min_cap, absolute_ceiling)


def load_or_run(cfg, min_cap, absolute_ceiling, quick=False, force=False):
    """Use stored calibration if usable, otherwise measure and store it."""
    if not force:
        cached = load(cfg.calibration_path)
        if cached is not None:
            return phase_f_clamp(cached, min_cap, absolute_ceiling), False
    result = run(cfg, min_cap, absolute_ceiling, quick=quick)
    try:
        save(result, cfg.calibration_path)
    except OSError:
        pass                    # persisting is an optimisation, not a
                                # correctness requirement
    return result, True


def apply_to_engine(result, engine, engine_module):
    """Hand the finished tables to the engine.

    The engine holds these from startup and never re-derives them per round
    (section 12 test 31).
    """
    engine.cost_curve = engine_module.CostCurve(
        decode_table=result.get("decode_curve") or None,
        prefill_table=result.get("prefill_curve") or None)
    return engine_module.OutputCap(result.get("output_cap_table") or None)
