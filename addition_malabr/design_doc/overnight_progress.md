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
2. [DONE] chat template application (§6c) — ChatFormatter, model-agnostic
3. [DONE] single-threaded engine loop, EOS / output cap (+ §6d streamer, §8 cap)
4. [DONE] compaction (§8) + position-shift fix + input-cap gate 2
5. [DONE] scheduler §9a/§9b (round budget, aging, chunked prefill, closed loop)
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

1. **§6c's incremental fragment is WRONG as written; replaced with a derived
   delta.** The doc says subsequent turns append a hardcoded
   `<|im_start|>{role}\n{content}<|im_end|>\n`. That is ChatML, i.e.
   Qwen-specific, and it is also wrong for the assistant turn (turn 1 with
   add_ass=True already leaves that block OPEN, so a fragment would duplicate
   the header). ChatFormatter instead renders the full conversation with the
   model's own template and takes the suffix past what is already in KV.
   Verified on gemma-3, whose delta begins `<end_of_turn>` — the hardcoded
   ChatML version would have silently malformed every turn after the first.
   **Recommend correcting §6c itself.**
2. **SlotAllocator must NOT touch the model from a connection thread — §7 is
   self-contradictory here.** §7's sketch has `acquire()` call
   `llama_memory_seq_rm`, while §7's own concurrency note says connection
   threads create sessions concurrently, and §7 separately names
   single-threadedness a load-bearing invariant. Those cannot all hold.
   `llama.h` states "The API is thread-safe" EXACTLY ONCE, scoped to the
   Tokenization section (line 1131); the Memory section (line 711) has no such
   guarantee. Wiping KV from a connection thread while the engine thread is
   inside `llama_decode()` is an undocumented data race whose symptom is
   corrupted KV — silently wrong output, not a crash.
   FIX APPLIED: acquire()/release() are now pure bookkeeping (any thread);
   `prepare(slot)` does the wipe and is ENGINE THREAD ONLY; `assert_ready(slot)`
   guards the decode path so a missed wipe is a loud error instead of a leak.
   The release-side "belt and braces" wipe was DROPPED — it would have to run on
   a connection thread, and wipe-on-acquire exists precisely because release
   cannot be trusted. **Recommend correcting §7.**
3. **§6b's rollback target for GENERATING is not usable; rolling the WHOLE
   turn back instead.** §6b says a GENERATING session rolls back to
   `pos_before_generation`, keeping the user message and the generation prompt
   in KV. That is not a valid boundary: the KV at that point ends inside an
   OPEN assistant block (the template's trailing generation prompt), so
   appending a new user turn after it produces malformed structure — caught by
   ChatFormatter's prefix check, not by reasoning. Rolling back to
   `pos_before_request` is also what §6b's OWN stated rule requires ("a turn
   only enters permanent context if it reaches EOS or the output cap"). A turn
   is the user message AND its response, so both go. `pos_before_generation` is
   still tracked — §8 compaction must shift it, and resume/regenerate would
   need it. **Recommend correcting §6b.**
4. **§6b and §6c were never reconciled: rollback must undo the FORMATTER too.**
   §6b specifies rollback purely as KV positions; §6c later added a second
   per-session state (the rendered conversation) that must roll back in
   lockstep. Rolling one without the other desyncs the formatter from the KV
   and the next turn is built on a conversation the model never saw. Added
   `ChatFormatter.checkpoint()/restore()`, captured in `_begin_turn` at exactly
   the same moment as `pos_before_request`. **Recommend §6b naming this.**
5. **§8 compaction must drop the FORMATTER's messages too — same gap as §6b.**
   §8 specifies compaction entirely as KV positions and never mentions the
   rendered conversation. Dropping KV without dropping the matching messages
   leaves `_rendered` describing content the model no longer has. Added
   `ChatFormatter.drop_messages()`, called from `compact()`, plus `msg_index`
   on each Turn (shifted like the positions). Verified: formatter render length
   == s.pos after compaction (330 == 330). **Recommend §8 name this.**
6. **Turn boundary = [user-delta start, response end), and that choice is
   load-bearing.** §8 says "aligned to chat-template boundaries" without saying
   which. This one works because every delta BEGINS with the terminator closing
   the preceding assistant block (a consequence of §6c deriving deltas from the
   template) — so dropping a whole exchange leaves the survivor's block closed
   by the next surviving delta, well-formed with no repair step.
7. **§9a's aging pass cannot rescue a session costing more than the whole
   budget — its prose and its code disagree.** §9a says a deep session "simply
   runs alone in its own round, spending the whole budget on itself, which
   harms nobody else". Its code gives that unconditional-first-pick escape ONLY
   to foreground. Measured: a background session at pos 6000 (100ms vs a 50ms
   round) was excluded 60/60 rounds with NO foreground present — permanent
   starvation, not the bounded limit §9a describes. FIX: aged background gets
   the same `or not picks` escape. Streak then bounded at exactly 5.
   **Recommend correcting §9a.**
8. **§9b's prefill chunk-minimum rule starves, taken literally.** §9b says
   "if n < PREFILL_CHUNK_MIN and n < remaining: continue  # wait for a rounder
   budget". If decode picks keep leaving less than a minimum chunk's budget,
   that holds EVERY round and the prompt is never prefilled. Measured: 200
   rounds, zero chunks. §9b bounds decode starvation with aging but leaves
   prefill unbounded. FIX: MAX_PREFILL_STALLS, then force one minimum chunk —
   overruns the bound by at most one chunk, which is finite, unlike never
   answering. **Recommend correcting §9b.**
9. **Stop condition uses `llama_vocab_is_eog`, not `== llama_vocab_eos`.**
   §5b describes the EOS check as verified-correct, but eos() returns ONE token
   while Qwen3 flags SIX as end-of-generation (gemma-3 flags 3). Comparing
   against eos alone would miss the rest and run to the output cap.
   **Recommend correcting §5b.**

## Cycle log

- cycle 0 (start): SlotAllocator written + 15 checks passing; lock rationale
  comment corrected after measurement disproved it.
- cycle 5: §9a/§9b scheduler + batched engine round. CostCurve (median vs
  worst-observed), build_batch (fg unconditional, aging, cheapest-first),
  chunked prefill, closed loop (EWMA, floor 1.0, ceiling 8.0). Engine now runs
  ONE batched llama_decode carrying prefill chunks and decode tokens together.
  THREE real bugs found: (a) §9a aging cannot rescue an over-budget session —
  prose and code disagree; (b) §9b's chunk-min rule starves prefill outright;
  (c) llama_get_logits_ith takes the BATCH index, not the ordinal among tokens
  requesting logits — crashed with GGML_ASSERT(logits != nullptr). That last
  one was latent while sampling used index -1; batching made it real.
  Also fixed my own inconsistency: prefill_tokens_affordable returned 0 when
  uncalibrated while the comment claimed one min chunk per round — an
  uncalibrated engine would never prefill.
  §12 tests 52/55/62 now have implementations + negative controls.
- cycle 4: compaction (§8). Verified seq_rm/seq_add semantics against the real
  model first (removes [p0,p1); seq_add(p0,-1,d) shifts everything >=p0).
  Position-shift fix applied to turn boundaries, BOTH §6b snapshots, s.pos, and
  formatter msg_index. Added the assertion §8 names but never wrote: nothing
  live may sit strictly inside a dropped range. Input-cap gate 2 added.
  Negative control proves the shift test has power — without the shift,
  boundaries go non-contiguous AND the last one exceeds s.pos.
  Found: §8 never mentions the formatter (same gap §6b had).
- cycle 3: engine loop (Session/Engine), §6d Utf8Streamer, §8/PhaseD-G OutputCap,
  §5d two sampling configs, per-session sampler (shared samplers would couple
  sessions' RNG/penalty state — same bleed class as the KV slot). 29 engine
  checks + regressions all green. TWO real holes found by tests, both caught by
  guards written in earlier cycles: (a) Engine never called prepare(), so
  assert_ready refused every first request — without that guard this was a
  silent cross-session KV leak; (b) §6b/§6c rollback desync, caught by the
  formatter prefix check. Also fixed verify_against_full()'s unstated
  precondition (it compared against the wrong render after a completed
  assistant turn and reported a false divergence).
- cycle 2 (audit): found a REAL hole in both my code and §7 — the slot wipe ran
  on connection threads, racing llama_decode on the engine thread. llama.h
  guarantees thread-safety only for tokenization. Restructured into
  acquire/release (bookkeeping, any thread) + prepare (engine thread) +
  assert_ready (decode guard). 17 checks pass incl. a negative control proving
  the guard test detects a stripped guard.
- cycle 1: ChatFormatter + is_stop_token. Closed §6c's own stated-unverified
  question: incremental == full render, and verified in TOKENS not just strings
  (§6c only compared strings; BPE merges across boundaries — measured
  tok("hell")+tok("o")=[56095,78] vs tok("hello")=[14990]). The turn split is
  safe structurally because every delta begins on a special token, and special
  tokens are hard BPE boundaries. Passes on Qwen3 AND gemma-3. Both negative
  controls (corrupt _rendered, broken prefix) fire correctly.
