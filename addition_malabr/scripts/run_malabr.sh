#!/usr/bin/env bash
#
# Launch Chromium with MALABR tunables, and record exactly which config
# produced the run.
#
# Why this exists: every tunable here is a base::FeatureParam rather than a
# C++ constant, specifically so it can be changed WITHOUT a rebuild (a full
# Chromium build on this machine costs 8-9 hours). This script is the place
# those values live, so an experiment is reproducible from one file instead
# of a remembered command line.
#
# phase1_design.md 11a requires calibration numbers be tied to the config
# that produced them -- that is what the run-log at the bottom is for.
#
# Usage:
#   ./run_malabr.sh                                  # defaults below
#   FRAME_READ_TIMEOUT_SECONDS=120 ./run_malabr.sh   # override one value
#   ./run_malabr.sh --dry-run                        # print, do not launch
#   ./run_malabr.sh -- --incognito                   # pass extra chrome args

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHROME_BIN="${CHROME_BIN:-${REPO_ROOT}/out/Default/chrome}"

# ---------------------------------------------------------------------------
# TUNABLES
#
# Add a line here when a new FeatureParam is introduced. The name on the LEFT
# must match the /*name=*/ string in extensions/common/extension_features.cc
# exactly -- a typo is silently ignored by Chromium (it just uses the default),
# which is why the verify step below prints what was actually sent.
# ---------------------------------------------------------------------------
declare -A TUNABLES=(
  # extensions/common/extension_features.cc :: kMalabrFrameReadTimeoutSeconds
  # Per-frame recv() budget in seconds. Placeholder until prefill latency is
  # measured (phase1_design.md 14).
  [frame_read_timeout_seconds]="${FRAME_READ_TIMEOUT_SECONDS:-60}"

  # --- future tunables, uncomment as the FeatureParams are added ---
  # [round_latency_budget_ms]="${ROUND_LATENCY_BUDGET_MS:-50}"
  # [max_consecutive_exclusions]="${MAX_CONSECUTIVE_EXCLUSIONS:-5}"
  # [stale_visibility_timeout_ms]="${STALE_VISIBILITY_TIMEOUT_MS:-2000}"
)

FEATURE_NAME="MalabrTunables"

# Separate profile so experiments never disturb a real browsing profile, and
# so a run can be reset by deleting one directory.
USER_DATA_DIR="${USER_DATA_DIR:-/tmp/malabr-profile}"

# Optional unpacked extension to auto-load.
EXTENSION_DIR="${EXTENSION_DIR:-}"

RUN_LOG="${RUN_LOG:-${REPO_ROOT}/addition_malabr/scripts/run_history.log}"

# ---------------------------------------------------------------------------
# Build the --enable-features string:
#   MalabrTunables:name1/value1/name2/value2
# Chromium synthesizes a field trial when params are given without a study,
# verified in base/feature_list.cc::ParseEnableFeatureString.
# ---------------------------------------------------------------------------
params=""
for key in "${!TUNABLES[@]}"; do
  params+="${key}/${TUNABLES[$key]}/"
done
params="${params%/}"   # strip trailing slash

if [[ -n "$params" ]]; then
  ENABLE_FEATURES="${FEATURE_NAME}:${params}"
else
  ENABLE_FEATURES="${FEATURE_NAME}"
fi

# ---------------------------------------------------------------------------
# Assemble argv
# ---------------------------------------------------------------------------
DRY_RUN=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --)        shift; EXTRA_ARGS+=("$@"); break ;;
    *)         EXTRA_ARGS+=("$1"); shift ;;
  esac
done

CHROME_ARGS=(
  "--enable-features=${ENABLE_FEATURES}"
  "--user-data-dir=${USER_DATA_DIR}"
  # MALABR logs at INFO; without this they are dropped.
  "--enable-logging=stderr"
  "--v=0"
)

if [[ -n "$EXTENSION_DIR" ]]; then
  CHROME_ARGS+=("--load-extension=${EXTENSION_DIR}")
fi

CHROME_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

# ---------------------------------------------------------------------------
# Report + record
# ---------------------------------------------------------------------------
echo "MALABR launch config"
echo "  chrome      : ${CHROME_BIN}"
echo "  profile     : ${USER_DATA_DIR}"
echo "  tunables    :"
for key in "${!TUNABLES[@]}"; do
  printf '      %-32s = %s\n' "$key" "${TUNABLES[$key]}"
done
echo "  --enable-features=${ENABLE_FEATURES}"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry run] would exec:"
  printf '  %q ' "$CHROME_BIN" "${CHROME_ARGS[@]}"; echo
  exit 0
fi

if [[ ! -x "$CHROME_BIN" ]]; then
  echo "ERROR: chrome binary not found or not executable:" >&2
  echo "       ${CHROME_BIN}" >&2
  echo "       Build it first, or set CHROME_BIN=/path/to/chrome" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# STALENESS GUARD
#
# A chrome binary from before your last source edit will run happily and show
# none of your changes -- an easy hour to lose. Compare the binary's mtime
# against the MALABR sources and fail loudly rather than silently testing old
# code. Override with ALLOW_STALE=1 when you genuinely mean to.
# ---------------------------------------------------------------------------
MALABR_SOURCES=(
  "${REPO_ROOT}/extensions/browser/api/malabr"
  "${REPO_ROOT}/extensions/common/api/malabr.idl"
  "${REPO_ROOT}/extensions/common/extension_features.cc"
  "${REPO_ROOT}/chrome/browser/malabr_manager.cc"
  "${REPO_ROOT}/chrome/browser/malabr_manager.h"
)
newest_src=0
for path in "${MALABR_SOURCES[@]}"; do
  [[ -e "$path" ]] || continue
  while IFS= read -r ts; do
    (( ts > newest_src )) && newest_src=$ts
  done < <(find "$path" -type f -printf '%T@\n' 2>/dev/null | cut -d. -f1)
done
chrome_ts=$(stat -c %Y "$CHROME_BIN")

if (( newest_src > chrome_ts )); then
  echo "ERROR: chrome binary is OLDER than MALABR sources -- it does not" >&2
  echo "       contain your changes." >&2
  echo "         chrome built : $(date -d @${chrome_ts} -Is)" >&2
  echo "         newest source: $(date -d @${newest_src} -Is)" >&2
  echo "       Rebuild, or set ALLOW_STALE=1 to run the old binary anyway." >&2
  [[ "${ALLOW_STALE:-0}" == "1" ]] || exit 1
  echo "       ALLOW_STALE=1 set -- continuing with the stale binary." >&2
fi

# One line per run, so a measurement can always be traced back to its config.
{
  printf '%s\t%s\t%s\n' \
    "$(date -Is)" \
    "$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)" \
    "${ENABLE_FEATURES}"
} >> "$RUN_LOG"

echo "recorded to ${RUN_LOG}"
echo "launching..."
exec "$CHROME_BIN" "${CHROME_ARGS[@]}"
