#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config syncspec || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

cd "$ROOT"
ARGS=(
  --backend "${BACKEND:-transformers}"
  --output "${OUTPUT_FILE:-outputs/syncspec.jsonl}"
  --device "${DEVICE:-${FI_DEVICE:-cuda}}"
  --dtype "${DTYPE:-bfloat16}"
  --max-samples "${MAX_SAMPLES:-1}"
  --batch-size "${BATCH_SIZE:-1}"
  --max-new-tokens "${MAX_NEW_TOKENS:-64}"
  --max-input-tokens "${MAX_INPUT_TOKENS:-0}"
)
if [[ -n "${KD:-}" ]]; then ARGS+=(--kd "$KD"); fi
if [[ -n "${KV:-}" ]]; then ARGS+=(--kv "$KV"); fi
if [[ -n "${BUDGET_PROFILES:-}" && -z "${KD:-}" && -z "${KV:-}" ]]; then
  ARGS+=(--budget-profiles "$BUDGET_PROFILES")
fi
if [[ -n "${DATA_FILE:-}" ]]; then ARGS+=(--input "$DATA_FILE"); fi
if [[ -n "${TARGET_MODEL:-}" ]]; then ARGS+=(--target-model "$TARGET_MODEL"); fi
if [[ -n "${DRAFTER_CHECKPOINT:-}" ]]; then ARGS+=(--drafter-checkpoint "$DRAFTER_CHECKPOINT"); fi
if [[ -n "${SELECTOR_CHECKPOINT:-}" ]]; then ARGS+=(--selector-checkpoint "$SELECTOR_CHECKPOINT"); fi
if [[ -n "${SURVIVAL_CHECKPOINT:-}" ]]; then ARGS+=(--survival-checkpoint "$SURVIVAL_CHECKPOINT"); fi
if [[ -n "${PROFILE:-}" ]]; then ARGS+=(--profile "$PROFILE"); fi
if [[ -n "${GATE_TABLE:-}" ]]; then ARGS+=(--gate-table "$GATE_TABLE"); fi
if [[ "${STOCHASTIC:-0}" == "1" ]]; then ARGS+=(--stochastic); fi
if [[ "${LOCAL_FILES_ONLY:-1}" == "1" ]]; then ARGS+=(--local-files-only); else ARGS+=(--no-local-files-only); fi
if [[ "${SMOKE:-0}" == "1" ]]; then ARGS+=(--smoke); fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_syncspec.py" "${ARGS[@]}" "$@"
