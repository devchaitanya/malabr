#!/usr/bin/env python3
"""MALABR inference server entry point.

Launched by MalabrManager as `python3 addition_malabr/mserver/app.py`
(chrome/browser/malabr_manager.cc:36).

Startup order matters and is not arbitrary:
  1. config       -- sizing computed from THIS machine, not baked in
  2. pid lock     -- acquired for real, first, so a duplicate start exits in
                     milliseconds instead of after a full model load
  3. CPU ceiling  -- after the lock, so a rejected start never creates a
                     systemd scope it will not use; before anything heavy
  4. model+ctx    -- one context, created once, owned by the engine thread
  5. calibration  -- measured or loaded from disk; feeds the scheduler
  6. engine       -- started before the socket exists, so the first request
                     never races an engine that is not running yet
  7. socket       -- last, because accepting a connection we cannot serve is
                     worse than making the browser retry its connect
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llama_cpp.llama_cpp as C

from malabr_service import calibration as cal
from malabr_service import engine as eng
from malabr_service import runtime as rt
from malabr_service.config import load_config


def apply_cpu_ceiling(cores):
    """HARD CPU cap (§11): move THIS process into a transient systemd scope with
    a cpu.max quota, kernel-enforced. `cores` is a float core count; <=0 = off.

    A pure backstop -- the default (config.compute_cpu_max_cores) is half the
    logical CPUs and always above n_threads, so it never bites normal
    operation. It exists only so a bug or a pathological model cannot take the
    whole machine; MALABR_CPU_DUTY is the smooth throttle that sets where usage
    actually sits.

    Why a self-move via busctl and not `systemd-run --scope python app.py`: that
    leaves systemd-run as the parent and python as a child IN the scope, and
    MalabrManager's Terminate(pid) on the parent does NOT propagate to python
    (tested). Moving our OWN pid into the scope keeps the pid MalabrManager
    tracks, has no intermediary to leak, and the empty scope is GC'd when we
    exit. cpu is not delegated to our starting scope, so writing cpu.max
    directly is impossible; systemd (the cgroup manager) enables the controller
    when a CPU property is set on the new scope.

    Best-effort: any failure (no busctl, not under a user systemd, denied) logs
    and continues unthrottled -- the duty throttle still applies.
    """
    if not cores or cores <= 0:
        return
    usec_per_sec = int(cores * 1_000_000)

    pid = os.getpid()
    try:
        # Inside the try: this function's contract is "any failure logs and
        # continues unthrottled", and an unreadable /proc (a hardened or
        # procfs-less environment) must not be the one exception that crashes
        # startup instead.
        if "malabr-cpu" in open("/proc/self/cgroup").read():
            return                  # already scoped (e.g. after a model-switch re-exec)
        r = subprocess.run(
            ["busctl", "--user", "call", "org.freedesktop.systemd1",
             "/org/freedesktop/systemd1", "org.freedesktop.systemd1.Manager",
             "StartTransientUnit", "ssa(sv)a(sa(sv))",
             f"malabr-cpu-{pid}.scope", "fail",
             "2",
             "PIDs", "au", "1", str(pid),
             "CPUQuotaPerSecUSec", "t", str(usec_per_sec),
             "0"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            print(f"malabr: CPU ceiling not applied: {r.stderr.strip()}",
                  file=sys.stderr, flush=True)
        elif _cpu_max_quota_active():
            print(f"malabr: CPU ceiling {cores:g} cores applied "
                  f"(cpu.max quota {usec_per_sec}us/s)", file=sys.stderr, flush=True)
        else:
            # busctl returned 0 but there is no cpu.max quota on our cgroup.
            # systemd does this silently when the cpu controller is not
            # delegated to the user session -- the scope exists, the ceiling
            # does not. Say so rather than log a false success.
            print("malabr: CPU ceiling scope created but no cpu.max quota is in "
                  "force -- running unthrottled (the duty throttle still applies)",
                  file=sys.stderr, flush=True)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"malabr: CPU ceiling not applied ({exc})", file=sys.stderr, flush=True)


def _cpu_max_quota_active():
    """Read back the effective cpu.max after StartTransientUnit succeeds.

    A zero return code from busctl is not proof the quota exists: systemd will
    create the scope and drop the CPU property without error when the cpu
    controller is not delegated to the user session. The move into the new
    scope can lag the call slightly, so retry briefly.
    """
    import time as _t
    for _ in range(15):
        try:
            cg = None
            for line in open("/proc/self/cgroup").read().splitlines():
                if line.startswith("0::"):
                    cg = line[3:].strip()
                    break
            if cg is not None:
                with open("/sys/fs/cgroup" + cg + "/cpu.max") as fh:
                    if fh.read().split()[0] != "max":
                        return True         # a numeric quota is enforcing
        except (OSError, IndexError):
            pass
        _t.sleep(0.02)
    return False


def build(cfg, quick_calibration=False):
    """Construct model, context, engine and the output cap. Returns all four."""
    C.llama_backend_init()

    mp = C.llama_model_default_params()
    mp.n_gpu_layers = 0                       # CPU only: this runs beside the
                                              # browser, not on a spare GPU
    model = C.llama_model_load_from_file(cfg.model_path.encode(), mp)
    if not model:
        raise RuntimeError(f"could not load model at {cfg.model_path}")

    # Calibration BEFORE the real context, because Phase A picks n_threads and
    # the context is created once with that value.
    result, measured = cal.load_or_run(cfg, eng.MIN_CAP, eng.ABSOLUTE_CEILING,
                                       quick=quick_calibration)
    # An EXPLICIT MALABR_N_THREADS must win over the cached calibration's pick --
    # otherwise the env var is silently a no-op whenever calibration.json exists,
    # which is almost always.
    if os.getenv("MALABR_N_THREADS"):
        n_threads = cfg.n_threads
    else:
        n_threads = result.get("n_threads") or cfg.n_threads

    cp = C.llama_context_default_params()
    cp.n_ctx = cfg.n_ctx
    cp.n_seq_max = cfg.n_seq_max
    cp.n_batch = cfg.n_batch
    cp.n_threads = n_threads
    cp.n_threads_batch = n_threads
    if cfg.shared_kv and hasattr(cp, "kv_unified"):
        # One shared pool of n_ctx cells instead of n_seq_max hard slices, so a
        # session can grow past n_ctx/n_seq_max. The engine's aggregate guard is
        # then what keeps the total within n_ctx.
        cp.kv_unified = True
    ctx = C.llama_init_from_model(model, cp)
    if not ctx:
        raise RuntimeError("llama context creation failed")

    vocab = C.llama_model_get_vocab(model)
    mem = C.llama_get_memory(ctx)
    allocator = eng.SlotAllocator(mem, cfg.n_seq_max)
    engine = eng.Engine(ctx, model, vocab, allocator,
                        sampling=eng.SAMPLING_CHAT,
                        n_ctx=cfg.n_ctx, n_seq_max=cfg.n_seq_max,
                        shared_kv=cfg.shared_kv,
                        session_budget=cfg.n_ctx_per_session,
                        cpu_duty=cfg.cpu_duty)
    output_cap = cal.apply_to_engine(result, engine, eng)

    print(f"malabr: model={os.path.basename(cfg.model_path)} "
          f"n_ctx={cfg.n_ctx} ({cfg.n_ctx_per_session}/session"
          f"{', shared-KV' if cfg.shared_kv else ''}) "
          f"n_seq_max={cfg.n_seq_max} n_threads={n_threads}"
          f"{f' cpu_duty={cfg.cpu_duty}' if cfg.cpu_duty < 1.0 else ''}"
          f"{f' cpu_max={cfg.cpu_max_cores:g}' if cfg.cpu_max_cores > 0 else ''}",
          flush=True)
    print(f"malabr: calibration {'measured' if measured else 'loaded'}"
          f"{' (FALLBACK)' if result.get('fallback') else ''}"
          f" passed={result.get('passed')}", flush=True)
    for w in result.get("warnings") or []:
        print(f"malabr: WARNING {w}", file=sys.stderr, flush=True)

    return model, ctx, engine, output_cap


def main():
    cfg = load_config()
    quick = os.getenv("MALABR_QUICK_CALIBRATION") == "1"
    model = ctx = engine = None
    lock = rt.PidLock(cfg.socket_path + ".pid")
    try:
        # Take the single-instance lock FOR REAL, first thing. This is spawned
        # by MalabrManager, so a duplicate start (a browser restart that did
        # not confirm the old child died) is expected, and exit 3 keeps it out
        # of the Chrome log as a crash. It must be a real acquire, not a
        # read-only probe: a probe leaves the pidfile unclaimed through the
        # ~2-minute cold build(), so two launches close together both pass it
        # and both pay for a model load and calibration before one is turned
        # away at the socket. And it must precede apply_cpu_ceiling, or the
        # rejected start still spins up a systemd scope it will never use.
        lock.acquire()
        apply_cpu_ceiling(cfg.cpu_max_cores)   # before anything heavy, calibration included
        model, ctx, engine, output_cap = build(cfg, quick_calibration=quick)
        engine.start()
        rt.run_server(config=cfg, engine=engine,
                      formatter_factory=lambda: eng.ChatFormatter(model, vocab_of(model)),
                      output_cap=output_cap, lock=lock)
        return 0
    except rt.SingleInstanceError as exc:
        print(f"malabr: {exc}; exiting", file=sys.stderr, flush=True)
        return 3
    finally:
        if engine is not None:
            engine.stop()
        if ctx:
            C.llama_free(ctx)
        if model:
            C.llama_model_free(model)
        lock.release()                          # no-op if never acquired


def vocab_of(model):
    return C.llama_model_get_vocab(model)


if __name__ == "__main__":
    sys.exit(main() or 0)
