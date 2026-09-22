# MALABR -- progress report

Branch `chaitanya/v0`. Covers the work since the shared-KV memory model landed:
two full audits of the inference server and chat panel, the fixes that came out
of them, and the design analysis done alongside.

## What the system is

MALABR runs one local LLM (llama.cpp, CPU only) inside a patched Chromium and
shares it across browser tabs. Each tab gets an isolated session; the server
schedules them so the visible tab stays responsive, bounds each session's
memory, and keeps the browser itself usable. The chat panel is a content-script
extension talking to a Python server over a Unix socket.

## Audit 1 -- correctness (13 suspect areas, all resolved)

| Problem found | What was wrong | How it was fixed |
|---|---|---|
| Shared-KV pool could wedge permanently | A rollback snapshot from the previous turn was left pointing inside an exchange that the memory guard then dropped; the guard's own safety check fired every round and nothing recovered | Snapshot is pinned at the start of every turn and cleared when a turn ends |
| Same wedge from idle sessions | The guard also compacts idle sessions, which carry the same stale snapshot | Same pin applied on turn completion and rollback |
| A whitespace-only reply killed the session | Chat templates trim replies; the KV kept the whitespace, so the next turn failed its consistency check and tore the session down | Trim logic rewritten: handles all-whitespace replies, refuses to cut into real text, bounded to the reply |
| Model switch could leave a dead server | `os.execv` keeps the pid; if the old pidfile survived, the new process saw its own pid as a rival and refused to start | Lock treats its own pid as stale and reclaims it |
| "New chat" only cleared the display | The server session and its context lived on, so the next message continued the old conversation | Wired to a real server-side teardown over the existing control channel |
| CPU ceiling applied before the duplicate-instance check | A rejected second launch still created a systemd scope and could stall on it | Order swapped; ceiling now also verifies the quota actually took effect |
| Memory guard could give up and let the next decode fail | When compaction ran out of room it returned silently and the engine hit "no memory slot" | Guard now rejects the not-yet-started turn with a clear error instead |
| Chat panel could start two generations at once | The send lock timed out on slow first tokens | Timeout aligned with the existing round-trip limit; a timed-out send cleans up its UI |
| Test harness used a fixed per-session budget | Did not match the production formula, so the compaction trigger in tests was off | Harness mirrors production |

Four other suspect areas (BOS handling under rollback, template-error teardown,
an unlocked read in the guard, shared-KV parity) were verified correct and
locked with regression tests. The full suite now also runs a second time in
shared-KV mode and passes identically.

## Audit 2 -- line-by-line review (16 items, all resolved)

Real gaps: an unbounded read of the wire header length (same bug class already
fixed for the payload); the single-instance lock taken only after the
two-minute model load, so two launches could both pay for it; a foreground
session starved permanently when two extensions share a tab; the benchmark
command bypassing the send lock; a startup path that could crash instead of
degrading. The rest were duplicated constants, a stale module docstring,
scattered imports, a redundant model reload during calibration, and config
documentation that could be misread. Each is one commit.

## Operational

Diagnosed the "model name not showing" launch failure: the browser resolves the
server script relative to its working directory, so launching from any
directory other than the repo root silently starts nothing. Documented the
correct launch command.

## Design analysis

Positioned the per-session memory/CPU isolation against prior art (Orca, vLLM,
VTC, Sarathi): the mechanisms are known; the novelty is the setting (CPU-only,
interactive, in-browser, across web origins) and the isolation-verification
method. Assessed alternative control signals (per-session CPU time, PSI
pressure, browser attention signals) and the page-context truncation problem
(naive prefix cut; extraction and relevance selection are the fixes, encoding
tricks are not).

## Next

- Optimization targets -- see `OPTIMIZATION_ANALYSIS.md`.
- Simplify the Python and JS for readability without changing behaviour.
- Page-context handling: readability extraction and relevance selection
  before truncation.
