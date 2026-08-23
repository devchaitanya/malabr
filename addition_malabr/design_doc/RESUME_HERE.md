# MALABR — where things stand (2026-08-24)

## Build is RUNNING, detached

Started with `setsid`, so it survives closing VS Code and the terminal.

    watch it        tail -f /tmp/malabr_build.log
    still alive?    pgrep -f "siso ninja" >/dev/null && echo running || echo stopped
    restart it      \
                    setsid nohup ./third_party/siso/siso ninja -C out/Default -j 4 chrome \
                        > /tmp/malabr_build.log 2>&1 &

`-j is specified. but not supported` in the log is harmless (siso v0.2.2).
The C++ has NEVER been compiled, so compile errors on this run are expected,
not a sign something is wrong. Grep the log with:

    grep -nE "error:|FAILED|ninja: build stopped" /tmp/malabr_build.log

This first build is large because the change set touches
`extension_function_histogram_value.h` (41+ files transitively). Later builds
touching only malabr files are fast.

## Python side: COMPLETE

All 11 build-order items done. 327 tests pass:
  282 unit checks   (scratchpad suites, see below)
  45  section 12 tests  (`addition_malabr/mserver/tests/test_phase1.py`)
  20  section 12 tests declared browser-only, listed explicitly in that file

Run the section 12 suite:

    /home/chaitu/Desktop/vscode/malabr/bin/python \
        addition_malabr/mserver/tests/test_phase1.py

Smoke-test the whole server end to end:

    cd addition_malabr/mserver
    MALABR_SOCKET_PATH=/tmp/m.sck MALABR_N_SEQ_MAX=4 MALABR_N_CTX=4096 \
    MALABR_QUICK_CALIBRATION=1 \
    /home/chaitu/Desktop/vscode/malabr/bin/python app.py

Files: `engine.py` (slots, chat template, loop, compaction, scheduler),
`protocol.py`, `runtime.py`, `config.py`, `calibration.py`, `app.py`.
`supervisor.py`, `training.py`, `storage.py` were deleted (dead sklearn path).

## Git

Everything is committed and pushed. `origin` has TWO push URLs, so
`git push origin` writes to both:
  https://github.com/spheresys/malabr.git   (lab)
  https://github.com/devchaitanya/malabr.git (personal fork)

Both carry the rewritten, attribution-free history. The fork may lag by a few
commits if a push was still in flight; `git push origin chaitanya/v0` catches
it up.

Commit messages must carry NO co-author trailer and no mention of any AI tool.

## What is NOT done

1. The C++ has never compiled. This build is its first test.
2. No output-fidelity testing. Compaction bookkeeping is proven exact and the
   canary survives, but section 8's logit-divergence check (tests 4/32/40 at
   the quality level) has NOT been run. Section 8 predicts turn-alignment
   reduces the rank-4 divergence seen on gemma-3; that prediction is untested.
3. Scheduler timing validated only against synthetic curves. Tests 52/55/62/63
   check the logic; nobody has measured a real round under mixed-depth load,
   and test 23 (p99 under a real cpu.max quota) needs cgroup work.
4. 20 section 12 tests need a real browser (window focus, iframes,
   chrome.storage.session, navigation rollback, control-connection reuse).
5. Admission is still not visibility-aware: background tabs holding every slot
   reject a new foreground tab. Section 7 names this as a known limitation.
6. 21 design-document corrections are queued in `implementation_notes.md` and
   the specification itself has NOT been edited. The four biggest: section 7's
   SlotAllocator sketch contains a data race; section 9a's aging cannot rescue
   an over-budget session (its prose and code disagree); section 9b's
   chunk-minimum starves prefill; section 11's memory formula would allocate
   7.7 GB where section 7 says 1.7 GB.

## Machine notes

- Python: `/home/chaitu/Desktop/vscode/malabr/bin/python` (conda root, 3.10),
  llama-cpp-python 0.3.34. Not a venv subdirectory.
- Models: `addition_malabr/models/`.
- RAM is tight and swap fills easily. Close WhatsApp/Teams/Brave before builds.
- Pushes are slow (10-17 GB of disk reads for ~130 KB uploaded) because the
  repo has 11,459 refs, a 46.8 GiB pack, and no commit-graph. One-time fix:
  `git commit-graph write --reachable` (slow once, faster forever after).
