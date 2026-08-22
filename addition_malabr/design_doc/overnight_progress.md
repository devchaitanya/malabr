# MALABR overnight autonomous run — live state file

Started 2026-08-23. Written by the /loop session so each 2-hourly iteration can
resume after context is summarized. UPDATE THIS FILE EVERY CYCLE.

## Standing permissions (granted by Chaitanya before starting)

- Full plan INCLUDING deletions: create/edit engine.py, calibration.py,
  protocol.py, runtime.py, config.py; gut supervisor.py; DELETE training.py and
  storage.py; run standalone llama.cpp model tests.
- Git: **commit locally each cycle, NEVER push.** He reviews and pushes himself.
- Ambiguity: pick the most defensible option, document the assumption in a code
  comment AND in the OPEN ASSUMPTIONS section below, keep building, report in
  the morning. Do not halt.
- Chromium build: permitted ONLY if all coding is genuinely finished and audits
  are clean. **HARD CONSTRAINT: VS Code is an ancestor of the Claude session
  (verified: claude -> code -> code -> gnome-shell). Killing it kills the loop.**
  So the build is a TERMINAL action only, launched detached via setsid, and only
  after everything else is complete. Default: do not build.

## Environment (verified, do not re-derive)

- Python: conda env root `/home/chaitu/Desktop/vscode/malabr/bin/python` (3.10),
  llama-cpp-python 0.3.34. It is a conda ROOT, not a `venv/` subdir (the v2
  handoff says venv — that is wrong).
- Models: `addition_malabr/models/Qwen3-0.6B-Q8_0.gguf`, `gemma-3-1b-it-Q4_K_M.gguf`
- `flatbuffers` is NOT installed in that env — the legacy path cannot import.
- `malabr_service/__init__.py` eagerly imports runtime -> ML.Request ->
  flatbuffers, which breaks standalone testing of engine.py. Tests load
  engine.py by path via importlib until this is fixed.
- Machine is memory-tight: ~473MB free RAM, swap 3.8/4.0GB. Use small n_ctx in
  tests. Do not run large builds.
- Scratchpad for temp files:
  /tmp/claude-1000/-media-chaitu-chaitanya-malabr-src/bc5e2c87-55a9-4c8c-ab3e-98004848c50b/scratchpad

## Build order (from §10a / handoff v4) — status

1. [DONE] SlotAllocator + wipe-on-acquire        engine.py
2. [ ] chat template application (§6c)
3. [ ] single-threaded engine loop, EOS / output cap
4. [ ] compaction (§8) + position-shift fix (EVERY live absolute position)
5. [ ] scheduler §9a/§9b  (round budget, aging, chunked prefill, closed loop)
then:
6. [ ] protocol.py — 6-field header, frames, payload bound
7. [ ] runtime.py — control route, reader thread
8. [ ] config.py  — n_ctx, n_seq_max, calibration path
9. [ ] calibration.py — §11a phases A–G + Phase B-prefill
10.[ ] gut supervisor.py; DELETE training.py, storage.py
11.[ ] §12 harness (31 numbered tests)

## Verified facts from measurement (do not re-assert without re-measuring)

- KV leak is REAL: slot reused without wipe still reports pos_max=12 for a
  13-token conversation. Wipe-on-acquire clears to -1.
- `llama_memory_seq_rm(mem,slot,-1,-1)` returned True in every case tried
  (full, partial, empty slot) on Qwen3. Could NOT force a False. The return
  check in _wipe_and_verify is DEFENSIVE, not observed-necessary.
- Wiping one seq does not disturb another (slot 3 wiped, slot 4 kept pos_max=19).
- The SlotAllocator lock: an unlocked pop() does NOT hand out duplicate slots
  (list.pop() is atomic under the GIL — 0 duplicates measured). The real race is
  CHECK-THEN-ACT: 120 IndexErrors / 160 acquires with a 2ms widened window and no
  lock; 0 with the lock. Symptom is a crash on a request that should have been a
  clean "no capacity" reject.
- LESSON: the first race test PASSED against a deliberately broken lock-free
  allocator. Weak tests look like passing tests. Every concurrency/scheduler test
  MUST be run against a deliberately broken implementation to prove it has
  detection power. This applies directly to §12 tests 52/55/62/63.

## OPEN ASSUMPTIONS (things the design doc did not settle; decided by me)

(none yet)

## Cycle log

- cycle 0 (start): SlotAllocator written + 15 checks passing; lock rationale
  comment corrected after measurement disproved it.
