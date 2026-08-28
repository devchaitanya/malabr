# MALABR — where things stand

## Run it

    EXTENSION_DIR=$PWD/addition_malabr/mserver/malabr_chat_extension \
      ./addition_malabr/scripts/run_malabr.sh

`chrome-sandbox` (hyphen, setuid root) is already in place. `NO_SANDBOX=1` for a
local run without it. `run_malabr.sh` now exports `MALABR_SHARED_KV=1` (see
below); set it to `0` there to compare against the partitioned model.

Server logs go to the launching terminal. To run it by hand with a file log:

    MALABR_SHARED_KV=1 setsid $MALABR_PYTHON -u addition_malabr/mserver/app.py \
      > /tmp/malabr_server.log 2>&1 < /dev/null &

The browser connects by socket path, so a hand-run server and the browser's
own both work; only one can hold `/tmp/malabr_v3.sck` at a time.

## Verified this pass (wire tests + the §12 suite, 46/46)

- **Streamed generate()**, multi-turn context, stop(), oversized-prompt reject,
  model switch, cap notice — all pass over the wire against a live server.
- **BOS token fix.** `llama_chat_apply_template` never emits the BOS byte, and
  the formatter tokenised with `add_special=False`, so gemma-3 / Llama / Mistral
  ran with no `<bos>` and lost track of who was speaking. Fixed: the first
  delta of a conversation tokenises with `add_special=True`. No-op for Qwen3
  (proved byte-identical). gemma recall went 0/5 → reliable.
- **Formatter/KV desync on the output cap.** A trimming template (gemma) drops
  trailing whitespace the cap left mid-reply; the KV kept those tokens, so the
  next turn died on the prefix check and every retry after it. Fixed:
  `_finish_turn` removes the trailing-whitespace tokens from the KV so it
  matches `_rendered`, and `_apply_control` now catches `TemplateError` and
  ends the session cleanly instead of failing forever. Verified: 8+ consecutive
  cap-terminated gemma turns, no desync.
- **Shared-KV memory model** (`MALABR_SHARED_KV=1`, default in `run_malabr.sh`):
  `kv_unified=true`, one shared pool of `n_ctx` cells, per-session soft cap of
  `n_ctx / MALABR_SESSION_SOFT_DIV` (default 2 → 8192 at n_ctx=16384), aggregate
  guard compacts the largest over-fair-share session (falls back to
  anchor-only) to keep Σ(pos) < n_ctx. Verified:
  - test 40b: guard compacts the greedy session, leaves a small one alone
  - adversarial config (n_ctx=3072, soft_div=1, 3 concurrent sessions, 21
    turns): guard fired 13×, 0 unrelievable, 0 "no memory slot", 0 round
    failures, all turns clean
  - RSS flat under load at 1× and 3× n_ctx (+1..+19 MiB over 4 sessions)
- **Foreground priority**: 1 fg + 3 bg all generating long → fg ~25-38 tok/s,
  bg ~0 during contention (200-500×). It is near-absolute (§9a), not a bias.
- **CPU**: `n_threads` alone pins ~3.0 cores (soft). A cgroup `CPUQuota=150%`
  via `systemd-run --user --scope` clamps to ~1.7 cores even with n_threads=3
  (hard). `MALABR_N_THREADS` now actually takes effect (it was silently
  overridden by cached calibration).
- **Shared-KV scheduler timing** (fg vs 3 DEEP bg sessions): fg inter-token
  latency p50=36ms, p95=60ms, max=74ms against a ~50ms round budget — the bound
  mostly holds, ~1.5× tail overshoot under deep contention. Worth a proper
  §9-style measurement before relying on it.

## Client changes — reload the extension, then verify in the browser

`chrome://extensions → reload`, then reload the test page. Not driven by an
automated test:

- **`bg.js`** (new) — `chrome.storage.session.setAccessLevel(...)` so the
  content script can use session storage at all. Without it, reload-persistence
  silently did nothing. Fixes F5-reload and same-origin navigation restore.
- Markdown rendering (injection-safe subset), output-cap notice, double-send
  guard (`sending` latch), model picker on its own header row, "page attached"
  chip above the bubble.
- Page text is sent **once per session** now — re-attached only if the page
  text actually changed or "New chat" was hit. Toggling the checkbox does not
  re-send.
- `malabr.stop()` button behaviour in the GUI is still unverified end to end
  (server side is proven).

## NOT done

1. **Aggregate CPU ceiling not wired into the launch path** (§11 / old item 3).
   `systemd-run --scope` works for a hand-run server, but MalabrManager does
   `Terminate(pid)` and a plain `kill` on the scope's main pid does NOT
   propagate to the python child (tested: `KillMode=control-group` only applies
   on `systemctl stop`, not a signal). Wiring it properly needs MalabrManager to
   stop a transient *service* instead of killing a pid — a C++ change + a
   browser-teardown test. `MALABR_N_THREADS` is the working soft cap meanwhile.
2. **Output FIDELITY** — §8's logit-divergence check has still never been run.
3. **Real scheduler TIMING** — the p95/max overshoot above needs a proper rig;
   test 23 (p99 under a real cpu.max quota) still needs cgroup work.
4. **20 of the §12 tests need a real browser.**
5. **Admission is not visibility-aware** — background tabs holding every slot
   reject a new foreground tab (§7 known limitation).
6. **33 design-doc corrections** still queued in `implementation_notes.md`;
   `phase1_design.md` itself still not edited.

## Env vars added this pass

    MALABR_SHARED_KV=1            one shared KV pool + aggregate cap (default via run_malabr.sh)
    MALABR_SESSION_SOFT_DIV=2     per-session soft cap = n_ctx / this
    MALABR_N_THREADS=N            now actually overrides calibration's pick
    MALABR_MODEL_PATH / _DIR      already existed; the panel's model switcher uses them

## Machine notes

- Python: `/home/chaitu/Desktop/vscode/malabr/bin/python` (3.10),
  llama-cpp-python 0.3.34. `MALABR_PYTHON` tells the browser to use it.
- RAM: `free` shows ~6 GB "used" but ~13/14 GB committed once page cache is
  counted, and **swap is 100% full**. `n_ctx=16384` (RSS ~2.5 GB) is about
  right; `24576` is the safe stretch; `49152` (RSS 6.2 GB) leaves the browser
  nothing. Lowering `n_seq_max` buys per-session size at zero RAM cost.
- `core.fsmonitor` disabled for this repo (watchman re-crawl off the USB drive
  pinned the machine). Re-enable with
  `git config core.fsmonitor .git/hooks/query-watchman`.
- `origin` has TWO push URLs (lab repo + personal fork); a push writes both.
  `git commit-graph write --reachable` once to stop pushes reading 10-17 GB.
