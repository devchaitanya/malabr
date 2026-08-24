# MALABR Phase 1 — implementation notes

Working notes kept alongside `phase1_design.md` while the Python side was
built. Records (a) how far the build order has got, (b) facts established by
measurement that should not be re-asserted without re-measuring, and (c) points
where the design document turned out to be wrong or underspecified and an
implementation decision had to be taken.

## Environment

- Python 3.10 conda environment; `llama-cpp-python` 0.3.34.
- Test models: `addition_malabr/models/Qwen3-0.6B-Q8_0.gguf`,
  `gemma-3-1b-it-Q4_K_M.gguf`.
- `malabr_service/__init__.py` now imports nothing at package import time.
  It previously did `from .runtime import run_server`, dragging every importer
  through `ML.Request` -> `flatbuffers` (not installed), which made `engine.py`
  impossible to import standalone.
- Development machine is memory-constrained; use a small `n_ctx` in tests.

## Build order — status

1. [DONE] SlotAllocator + wipe-on-acquire        engine.py
2. [DONE] chat template application (§6c) — ChatFormatter, model-agnostic
3. [DONE] single-threaded engine loop, EOS / output cap (+ §6d streamer, §8 cap)
4. [DONE] compaction (§8) + position-shift fix + input-cap gate 2
5. [DONE] scheduler §9a/§9b (round budget, aging, chunked prefill, closed loop)
then:
6. [DONE] protocol.py — 6-field header, frames, payload bound, control msgs
7. [DONE] runtime.py — control route, reader thread, PID lock
8. [DONE] config.py — startup-computed n_ctx/n_threads, calibration path
9. [DONE] calibration.py — §11a phases A–G + Phase B-prefill
10.[DONE] deleted supervisor.py/training.py/storage.py; wrote real app.py
11.[DONE] §12 harness — 45 executed, 20 declared browser-only skips

## Verified facts from measurement

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
- Concurrency and scheduler tests must be run against a deliberately broken
  implementation to confirm they can fail at all. A first attempt at the slot
  race test PASSED against a lock-free allocator, i.e. it had no detection
  power. This applies directly to §12 tests 52/55/62/63.

## Design-document corrections and open decisions

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
9. **Where protocol.py actually broke is NOT what the handoff says.** The
   handoff and §10a state it "splits the header into 3 fields and raises
   otherwise". It used `split(",", 2)` — maxsplit=2 — so a 6-field header
   yields exactly THREE parts, the count check PASSES, and it dies later on
   `int()` of the merged remainder with "invalid payload size". Same outcome
   (everything rejected), different mechanism. Minor, but the handoff's
   description is wrong. Also: `mserver_uds.cc`'s comment claims origin may be
   the literal "null" when opaque; `malabr_api.cc:77` rejects opaque origins
   before that point, so the comment is stale.
10. **Stop condition uses `llama_vocab_is_eog`, not `== llama_vocab_eos`.**
   §5b describes the EOS check as verified-correct, but eos() returns ONE token
   while Qwen3 flags SIX as end-of-generation (gemma-3 flags 3). Comparing
   against eos alone would miss the rest and run to the output cap.
   **Recommend correcting §5b.**

### Further corrections found while building runtime.py

11. **The outbox must be per-REQUEST, not per-session.** §7 calls it "the
    session's outbox". Under §6b's single-flight replace TWO handlers are
    briefly alive: the superseded one waiting for its terminal frame and the
    new one waiting for tokens. One shared queue means whichever polls first
    steals the other's frames — the superseded client could receive the new
    response, or hang until its 60s read timeout. Each request now owns a
    queue; the engine writes to whichever is current, and the swap happens on
    the engine thread AFTER the superseded terminal frame is written to the old
    one.

12. **`LIVE_TABS` must exempt `tab_id == -1`.** -1 is the documented
    "not tab-scoped" value and can never appear in a list of live tabs, so a
    naive "reap anything not in the list" reaps those sessions instantly, on
    the very first reconciliation. Not mentioned in §5f.

13. **Single-instance enforcement must check a PID before touching the
    socket.** §10 already flags that the inherited delete-stale-socket pattern
    is dangerous; the ordering is the specific part that matters. The socket is
    only unlinked AFTER the pid lock confirms no live owner, otherwise a second
    process can delete a live instance's socket and steal the path.

14. **§11's memory formula and §7's stated `n_ctx` disagree by 4.6x, and the
    formula is the wrong one.** §11 gives
    `n_ctx = MemAvailable * 0.8 / BYTES_PER_TOKEN`; §7 states `n_ctx = 16384`
    outright. On the development machine the formula yields **75,340 tokens
    (7.7 GB of KV)** against §7's 16,384 (1.7 GB). The formula overshoots for
    three reasons: `MemAvailable` counts reclaimable page cache as free, which
    is true for transient allocations and false for a permanent one-shot
    reservation (here almost all of that 9 GB is cache, and swap is already
    full); it never subtracts the model weights, which are also resident; and
    it leaves nothing for the browser — whose protection is the entire point of
    the design. Corrected to `0.5 * (MemAvailable - browser_reserve -
    model_bytes)`, capped at §7's 16384. That lands on exactly 2048
    tokens/session on reference hardware and degrades sensibly on smaller
    machines (2 GB -> 4096 floor, 4 GB -> 6880). **Recommend correcting §11.**

## First real measurements of the assembled engine

Qwen3-0.6B-Q8_0, n_ctx 4096, n_seq_max 4, n_threads 3 (quota-matched), 4
physical cores. Full calibration takes ~43s; it is cached to disk afterwards.

    Phase A  threads   1 -> 18.3, 2 -> 33.2, 4 -> 46.8 tok/s
                       fastest is 4, quota caps at 3, so 3 is chosen
    Phase B  decode    pos   32 -> 46.1 tok/s (median), 45.1 (worst)
                       pos  256 -> 37.1 / 31.5   <- 15% spread
                       pos  960 -> 29.0 / 27.0
    Phase B-prefill    depth  32 -> 279 tok/s
                       depth 512 -> 195 tok/s
    Phase C  joint     4 sessions @ depth 512, batched:
                       64.4 tok/s aggregate, 16.1 each

Three things worth recording:

- **Prefill is ~6x decode, not the order of magnitude §9b assumed.** 279 vs
  46 tok/s at depth 32. §14 recorded prefill as entirely unmeasured; this is
  the first number. Same direction as predicted, smaller factor.
- **Phase C does not contradict the document's 12.5 tok/s figure, it
  complements it.** At depth 512 batching gives 64.4 aggregate against 31.6
  solo — a 2x benefit. The document's measurement at depth 1500 gave 12.5
  aggregate against 25.1 solo, i.e. batching *hurt*. Both fit "batching stops
  paying once sessions are deep"; this is the shallow end of that curve.
- **The measured trial spread (1–15%) justifies the median/worst split.** At
  pos 256 the spread was 15%, so a round admitted on the median estimate there
  really could overrun its budget.

### Corrections found by running the §12 suite

15. **§9b's forced prefill chunk broke the very bound it sits inside.** My
    anti-starvation fix (force `PREFILL_CHUNK_MIN` through after N stalls) was
    measured to overrun the round budget by **87 ms against 50 ms, in 20% of
    rounds**. That trades "prefill never runs" for "the ~50ms preemption bound
    breaks regularly", which is the wrong way round — the bound is the
    headline claim. Fixed to admit `max(1, affordable)` instead:
    `PREFILL_CHUNK_MIN` is an EFFICIENCY floor ("below this, per-call overhead
    dominates"), not a correctness one. Result: 0 over-budget rounds, prompt
    still progresses.

16. **The formatter rollback checkpoint goes stale across compaction —
    §8's own warning, applied to a reference §8 does not know exists.** §8
    lists the live absolute positions compaction must shift. The formatter
    checkpoint added for correction 4 is another one, and missing it made a
    replace-after-compaction restore to a message count that no longer
    existed. Caught by the formatter's own prefix check (§12 test 53), not by
    review. The checkpoint is now a message COUNT, shifted in `compact()` like
    any position, rather than a `(count, rendered_length)` pair.

17. **An empty conversation must render as the empty string.** The chat
    template emits a bare generation prompt for zero messages
    (`'<|im_start|>assistant\n'` on Qwen3). Returning that after rolling a
    session's FIRST turn back claims KV content the cache does not hold, and
    breaks the prefix check on the next turn.

### §12 coverage

42 tests execute here. 21 are declared SKIP because they need a real browser
(window focus, iframes, `chrome.storage.session`, content-script injection,
`cpu.max` quota timing) — listed explicitly in `tests/test_phase1.py` rather
than silently omitted, so the gap between "Phase 1 done" and "Phase 1 tested
here" stays visible.

### Architecture audit findings (whole-system pass, after all 11 items)

18. **Cross-origin eviction was spuriously rejected at full capacity — §5g's
    mechanism broke exactly under load.** Eviction is deliberately
    ASYNCHRONOUS (a connection thread must not touch the model, §7), but
    admission ran immediately on the connection thread. Measured with every
    slot occupied: the new-origin session was refused "no free session slots"
    while the slot it needed belonged to the session just condemned, released
    one round later. Self-inflicted, and worst in the case §5g exists for.
    Fixed with `wait_for_teardown()`, bounded and blocking only a client-pool
    thread, which §7 calls a waiting room rather than compute. Covered by new
    test 64.

19. **The engine thread had no crash protection.** An unhandled exception in
    a round killed it silently. The socket stays open and the server keeps
    ACCEPTING, so it looks healthy while serving nothing, and every session
    hangs until the browser's 60s read timeout. Now logged and survived, with
    a consecutive-failure cap so a systematic fault stops rather than spins.
    Covered by new test 65.

20. **`foreground_tab_id` had no staleness timeout (§5f, test 58).** If the
    control connection died, the last value was frozen forever — and a frozen
    key naming a now-hidden tab grants that tab §9a's UNCONDITIONAL admission
    every round, permanently. The engine now degrades to "no foreground" after
    `STALE_VISIBILITY_TIMEOUT`, which is the safe direction: it costs priority,
    it cannot break the latency bound. The raw key is never mutated, so a
    reconnect restores it immediately. Test 58 moves from browser-only to
    executed.

21. **One documented exception to "no model calls off the engine thread":**
    `ChatFormatter.__init__` calls `llama_model_chat_template` from a
    connection thread. Checked rather than assumed — it takes a `const
    llama_model *` and returns `const char *`, a pure read of immutable model
    metadata with no context involved, so concurrent reads are safe. Every
    other model call is on the engine thread.

## First compile of the browser-side C++

The C++ had never been through a compiler. The first full build stopped after
194 of 22,516 steps on a single failing target; two real defects came out of
it, both fixed and each verified by recompiling the affected object.

22. **`malabr_feature` in `chrome/browser/BUILD.gn` declared only `//base`.**
    `malabr_manager.cc` includes browser-UI, extensions, content, sessions and
    net headers, and the UI headers reach Skia transitively — so the target
    failed with `'include/core/SkAlphaType.h' file not found`, which looks like
    a Skia problem and is actually a missing dependency. Added `//chrome/common`,
    `//components/sessions`, `//content/public/browser`, `//extensions/browser`,
    `//extensions/common`, `//net` and `//skia`. `gn gen` confirms no circular
    dependency, and the object now compiles. Sibling targets in the same file
    (`display_file_feature`) already follow this pattern; this one had been left
    behind.

23. **§5a's visibility snippet does not compile on this Chromium revision.**
    §5a shows
    `rfh->GetVisibilityState() == content::PageVisibilityState::kVisible`.
    There is no `content::PageVisibilityState`:
    `RenderFrameHost::GetVisibilityState()` returns
    `blink::mojom::PageVisibilityState`, and `render_frame_host.h` includes only
    its `-forward.h`, so the type is incomplete at the use site. Using it would
    need a blink mojom dependency `extensions/browser` should not take on.
    Replaced with `web_contents->GetVisibility() == content::Visibility::VISIBLE`
    — which is what §5a's own RESOLVED rule names, needs no new dependency, and
    reuses the `WebContents` already fetched and null-checked a few lines above
    for the tab id. **Recommend correcting §5a's code snippet.**

All four malabr objects (`malabr_manager`, `malabr_api`, `mserver_uds`,
`msocket_uds`) now compile. The remaining ~22,300 steps of the full build are
untouched Chromium and have not been run to completion.

24. **Cold start is ~130s, not the ~13s §10 assumes, and nothing retried the
    connect.** Measured by starting `app.py` with the SHIPPED DEFAULTS
    (n_ctx=16384, n_seq_max=8, no cached calibration) rather than the reduced
    settings every earlier test used: the socket appears after **130 seconds**.
    §10 estimates "~13s (model load + calibration)" — a 10x underestimate, with
    calibration dominating. Once its result is cached on disk later starts are
    quick, but a fresh profile pays the full cost.

    Worse, §10 also requires "a short retry-with-backoff inside `Send()`'s
    `Connect()`" and that was **never implemented**: `MServerUDS::Send` called
    `Connect()` exactly once, so every `generate()` issued during those 130
    seconds failed outright. The jittered backoff that does exist covers only
    the control connection, not request sockets. Added a 180s jittered retry
    budget sized from the measurement rather than from the estimate.
    **Recommend correcting §10's cold-start figure.**

    Worth revisiting later: §11c's own audit says isolation and
    browser-protection do NOT depend on calibration, only conversation length
    and speed do. So the server could bind immediately with the conservative
    fallback curve and calibrate in the background, swapping the measured curve
    in when ready. Not done here because calibration and the engine would then
    share the model context concurrently, which is exactly the thread-safety
    problem correction 2 exists to prevent.

25. **The aggregate CPU ceiling is never applied — the easy half of the two-layer
    cap is the missing one.** cgroups are process-granular; they bound what the
    whole inference process may take and cannot see inside it, because a
    `seq_id` is not a process. That asymmetry is the project's premise: the OS
    supplies the outer ceiling, MALABR divides what sits inside it.

    §11 requires BOTH: "a real cgroup cpu.max quota on the process, AND set
    n_threads to match... setting one without the other is the nonlinear-loss
    regime measured above." Only `n_threads` is implemented. `config.py`
    computes `quota_pct`, calibration records it, comments throughout refer to
    "the cgroup limit" — but nothing calls `systemd-run --scope -p CPUQuota=`
    or writes `cpu.max`, and `run_malabr.sh` does not either. So CPU is bounded
    today only by `n_threads=3`, a soft limit: three threads can saturate three
    of four cores and the kernel will not intervene.

    The mechanism itself is verified working on this machine, no root needed:
    an unthrottled busy loop uses 99% CPU; the same loop under
    `systemd-run --user --scope -p CPUQuota=25%` uses 25%, with wall time
    unchanged. Note `cgroup.controllers` in our own scope lists only
    `memory pids`, so writing `cpu.max` directly is not possible — the
    `cpu` controller is not delegated. systemd-run is the working path.

    NOT implemented here because the fix touches process lifecycle:
    `MalabrManager` spawns `python3 app.py` and later calls `Terminate(pid)` on
    that handle, so re-execing under a transient scope changes the PID and the
    handle may no longer reach the real process. Needs a termination test
    alongside it. **Recommend §11 note that only half the cap is built.**

26. **Memory needs no cgroup, and that is a structural result rather than an
    omission.** §11 measured RSS flat (~1139MB) from 1 to 8 resident sessions:
    `n_ctx` is a single fixed allocation at startup with no runtime path to
    grow. A `memory.max` on top would be redundant, since the allocation cannot
    exceed itself. Memory is therefore capped exactly at BOTH granularities —
    aggregate by construction, per-session by the `n_ctx / n_seq_max` budget
    plus compaction at 95%. CPU is the asymmetric case: the per-session
    division exists, the aggregate ceiling does not.

27. **Resource monopolisation belongs in §11c's calibration-INDEPENDENT column.**
    §11c lists isolation and browser-protection as not depending on
    calibration, and conversation length and speed as depending on it.
    Monopolisation resistance belongs with the former, and this was measured
    rather than argued. One session at position 4000 against five shallow ones,
    300 rounds, cost model swept across a 1000x error range:

        accurate              deep session took  9.1% of slots, nobody starved
        10x too optimistic                      16.7%,           nobody starved
        100x too pessimistic                    16.9%,           nobody starved
        no calibration at all                   18.0%,           nobody starved

    Fair share is 16.7%. What the cost model's accuracy actually buys is
    proportional charging of expensive sessions (the accurate run gives the deep
    session LESS than fair share, correctly); as the model degrades the
    scheduler stops discriminating and converges to round-robin. It never
    permits monopolisation, because the anti-monopolisation property is
    structural: the loop yields after every single token, `ABSOLUTE_CEILING`
    bounds one response, and the aging pass counts rounds rather than
    milliseconds. None of those take a calibration input.
