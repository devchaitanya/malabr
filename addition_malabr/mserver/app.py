#!/usr/bin/env python3
"""MALABR inference server entry point.

Launched by MalabrManager as `python3 addition_malabr/mserver/app.py`
(chrome/browser/malabr_manager.cc:36).

Startup order matters and is not arbitrary:
  1. config      -- sizing computed from THIS machine, not baked in
  2. model+ctx   -- one context, created once, owned by the engine thread
  3. calibration -- measured or loaded from disk; feeds the scheduler
  4. engine      -- started before the socket exists, so the first request
                    never races an engine that is not running yet
  5. socket      -- last, because accepting a connection we cannot serve is
                    worse than making the browser retry its connect
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llama_cpp.llama_cpp as C

from malabr_service import calibration as cal
from malabr_service import engine as eng
from malabr_service import runtime as rt
from malabr_service.config import load_config


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
    n_threads = result.get("n_threads") or cfg.n_threads

    cp = C.llama_context_default_params()
    cp.n_ctx = cfg.n_ctx
    cp.n_seq_max = cfg.n_seq_max
    cp.n_batch = cfg.n_batch
    cp.n_threads = n_threads
    cp.n_threads_batch = n_threads
    ctx = C.llama_init_from_model(model, cp)
    if not ctx:
        raise RuntimeError("llama context creation failed")

    vocab = C.llama_model_get_vocab(model)
    mem = C.llama_get_memory(ctx)
    allocator = eng.SlotAllocator(mem, cfg.n_seq_max)
    engine = eng.Engine(ctx, model, vocab, allocator,
                        sampling=eng.SAMPLING_CHAT)
    output_cap = cal.apply_to_engine(result, engine, eng)

    print(f"malabr: model={os.path.basename(cfg.model_path)} "
          f"n_ctx={cfg.n_ctx} ({cfg.n_ctx_per_session}/session) "
          f"n_seq_max={cfg.n_seq_max} n_threads={n_threads}", flush=True)
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
    try:
        # Refuse early and QUIETLY if another instance holds the socket. This
        # is spawned by MalabrManager, so an unhandled traceback here is noise
        # in the Chrome log for a condition that is expected (a browser restart
        # that did not confirm the old child died). Exit 3 distinguishes it from
        # a real crash.
        probe = rt.PidLock(cfg.socket_path + ".pid")
        existing = probe._read()
        if existing is not None and probe._alive(existing) and existing != os.getpid():
            print(f"malabr: another server is already running (pid {existing}); "
                  f"exiting", file=sys.stderr, flush=True)
            return 3
        model, ctx, engine, output_cap = build(cfg, quick_calibration=quick)
        engine.start()
        rt.run_server(config=cfg, engine=engine,
                      formatter_factory=lambda: eng.ChatFormatter(model, vocab_of(model)),
                      output_cap=output_cap)
        return 0
    except rt.SingleInstanceError as exc:
        print(f"malabr: {exc}", file=sys.stderr, flush=True)
        return 3
    finally:
        if engine is not None:
            engine.stop()
        if ctx:
            C.llama_free(ctx)
        if model:
            C.llama_model_free(model)


def vocab_of(model):
    return C.llama_model_get_vocab(model)


if __name__ == "__main__":
    sys.exit(main() or 0)
