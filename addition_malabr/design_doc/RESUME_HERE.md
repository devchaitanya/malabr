# MALABR — where things stand

## Right now

A Chromium build is running detached (`setsid`, survives terminal/VS Code close):

    watch      tail -f /tmp/malabr_link.log
    alive?     pgrep -f "siso ninja" >/dev/null && echo running || echo stopped
    errors     grep -nE "error:|FAILED" /tmp/malabr_link.log
    restart    setsid nohup ./third_party/siso/siso ninja -C out/Default -j 3 chrome \
                   > /tmp/malabr_link.log 2>&1 &

It is NOT a quick link. `malabr.stop()` added an entry to
`extension_function_histogram_value.h`, which §10 records as touching 41+ files
transitively, and the IDL change regenerates the API bindings.

## Run the browser

    EXTENSION_DIR=$PWD/addition_malabr/mserver/malabr_chat_extension \
      ./addition_malabr/scripts/run_malabr.sh

Needs the SUID helper once (the target builds `chrome_sandbox` with an
UNDERSCORE, the runtime wants `chrome-sandbox` with a HYPHEN):

    sudo cp out/Default/chrome_sandbox out/Default/chrome-sandbox
    sudo chown root:root out/Default/chrome-sandbox
    sudo chmod 4755 out/Default/chrome-sandbox

Or `NO_SANDBOX=1` for a local run (renderers then unsandboxed).

## WORKING, verified live in the browser

- Streamed generate() end to end: content script -> browser -> UDS -> engine.
- Multi-turn context: told a name, recalled it two turns later.
- Reload persistence: same tab + same origin keeps the server session; the
  panel restores its transcript from `chrome.storage.session`.
- Cross-origin isolation: same tab, different origin, canary NOT recoverable.
- Reasoning suppressed at the prompt (read out of the model's own template).
- Page text attached for "summarise this page".

## WORKING, verified in tests but NOT yet in the browser

- `malabr.stop()`. Server side proven: a stopped turn rolls back, the session
  stays alive, the slot is retained, and a canary planted before it is still
  recallable. The C++ compiles. Only the chrome link is outstanding -- that is
  what the running build is for. The panel feature-detects it, so the button
  stays a Send button until the binary has it.

## NOT DONE — the real remaining work

1. Output FIDELITY. Compaction bookkeeping is proven exact and the canary
   survives, but §8's logit-divergence check has never been run. §8 predicts
   turn-alignment reduces the rank-4 divergence seen on gemma-3; untested.
2. Real scheduler TIMING. Tests 52/55/62/63 validate logic against synthetic
   curves. Nobody has measured an actual round against the 50ms budget under
   mixed-depth load. Test 23 (p99 under a real cpu.max quota) needs cgroup work.
3. The aggregate CPU ceiling is NEVER APPLIED. §11 requires both a cgroup
   cpu.max quota and n_threads matched to it; only n_threads exists. Verified
   the mechanism works here (99% -> 25% under CPUQuota=25%, no root needed).
   Not wired up because MalabrManager holds a PID it later calls Terminate on,
   and re-execing under a transient scope changes that PID.
4. 20 of the 63 §12 tests need a real browser (window focus, iframes,
   storage.session halves, navigation rollback, control-connection reuse).
5. Admission is still not visibility-aware: background tabs holding every slot
   reject a new foreground tab. §7 names this as a known limitation.
6. 33 design-document corrections are queued in `implementation_notes.md` and
   `phase1_design.md` itself has NOT been edited.

## Rough effort, honestly

- finish this build + verify stop in the browser: under an hour, mostly waiting
- items 1 and 2 (fidelity + timing): the real research work, days not hours
- item 3 (cgroup ceiling): half a day including a teardown test
- item 4 (browser tests): a day or two
- item 6 (fold corrections back into the spec): a day

## Machine notes

- Python: `/home/chaitu/Desktop/vscode/malabr/bin/python` (conda root, 3.10),
  llama-cpp-python 0.3.34. MALABR_PYTHON tells the browser to use it.
- `core.fsmonitor` is disabled for this repo: after a reboot watchman re-crawls
  the 400k-file tree off the USB drive and stacked `git status` queries pinned
  the machine at load 15. Re-enable with
  `git config core.fsmonitor .git/hooks/query-watchman` if wanted.
- Pushes read 10-17 GB off disk for ~130 KB uploaded (11,459 refs, 46.8 GiB
  pack, no commit-graph). `git commit-graph write --reachable` fixes it once.
- `origin` has TWO push URLs (lab repo and personal fork); a push writes both.
