#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# Preserve caller overrides across the shared master source. The common
# loader intentionally sources the canonical env file, so restore the CAD
# namespace and direct model/data overrides explicitly afterwards.
SAVED_NAMES=()
SAVED_VALUES=()
for NAME in \
  CAD_PHASE CAD_VARIANT CAD_DATA_DIR CAD_OUTPUT_ROOT CAD_RUN_ID CAD_BUDGETS CAD_GAMMAS \
  CAD_SELECTOR CAD_TARGET_LAYER CAD_REFRESH_PERIOD CAD_SIGNAL_UPDATE_PERIOD \
  CAD_SOURCE_CHUNK_SIZE CAD_SOURCE_ANCHORS CAD_RECENT_OUTPUT CAD_PRIOR_STRENGTH \
  CAD_MIN_STATE_SUPPORT CAD_STATISTICS_DECAY CAD_HISTORY_ALPHA CAD_LATENCY_EMA_ALPHA CAD_CONTROLLER_MARGIN \
  CAD_ENTROPY_SIGNAL_TEMPERATURE CAD_TIMING_MODE CAD_COST_UPDATE_MODE \
  CAD_STATISTICS_UPDATE_MODE CAD_LENGTH_MODE CAD_FIXED_BUDGET CAD_FIXED_GAMMA \
  CAD_GAMMA_REFERENCE CAD_MAX_NEW_TOKENS MODEL_TARGET MODEL_DFLASH_DRAFT TARGET_MODEL \
  DRAFT_MODEL DATA_INPUT DATA_FILE RUN_SAMPLES RUN_MAX_NEW_TOKENS RUN_MAX_INPUT_TOKENS \
  RUN_TEMPERATURE MAX_SAMPLES MAX_INPUT_TOKENS TEMPERATURE LONG_BENCH_SEED SMOKE FULL; do
  if [[ -v "$NAME" ]]; then
    SAVED_NAMES+=("$NAME")
    SAVED_VALUES+=("${!NAME}")
  fi
done

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config dflash || exit 1
for INDEX in "${!SAVED_NAMES[@]}"; do
  printf -v "${SAVED_NAMES[$INDEX]}" '%s' "${SAVED_VALUES[$INDEX]}"
  export "${SAVED_NAMES[$INDEX]}"
done
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

if [[ "${SMOKE:-0}" == "1" && "${FULL:-0}" == "1" ]]; then
  echo "SMOKE=1 and FULL=1 cannot both be enabled" >&2
  exit 2
fi

ARGS=()
if [[ -n "${TARGET_MODEL:-${MODEL_TARGET:-}}" ]]; then
  ARGS+=(--target-model "${TARGET_MODEL:-$MODEL_TARGET}")
fi
if [[ -n "${DRAFT_MODEL:-${MODEL_DFLASH_DRAFT:-}}" ]]; then
  ARGS+=(--draft-model "${DRAFT_MODEL:-$MODEL_DFLASH_DRAFT}")
fi
if [[ -n "${CAD_DATA_DIR:-}" ]]; then
  ARGS+=(--data-dir "$CAD_DATA_DIR")
elif [[ -n "${DATA_FILE:-${DATA_INPUT:-}}" ]]; then
  ARGS+=(--data-file "${DATA_FILE:-$DATA_INPUT}")
fi
[[ -n "${CAD_OUTPUT_ROOT:-}" ]] && ARGS+=(--output-root "$CAD_OUTPUT_ROOT")
[[ -n "${CAD_RUN_ID:-}" ]] && ARGS+=(--run-id "$CAD_RUN_ID")
[[ -n "${MAX_SAMPLES:-}" && "${CAD_PHASE:-smoke}" == "smoke" ]] && ARGS+=(--max-samples "$MAX_SAMPLES")
[[ -n "${MAX_INPUT_TOKENS:-}" ]] && ARGS+=(--max-input-tokens "$MAX_INPUT_TOKENS")
[[ -n "${TEMPERATURE:-}" ]] && ARGS+=(--temperature "$TEMPERATURE")

if [[ -n "${CAD_PHASE:-}" ]]; then
  ARGS+=(--phase "$CAD_PHASE")
elif [[ "${SMOKE:-0}" == "1" ]]; then
  ARGS+=(--phase smoke)
elif [[ "${FULL:-0}" == "1" ]]; then
  ARGS+=(--phase test)
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/externals/dflash${PYTHONPATH:+:$PYTHONPATH}"
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_context_adaptive_dflash.py" "${ARGS[@]}" "$@"
