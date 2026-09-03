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
MAX_SAMPLES="${MAX_SAMPLES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
if (( MAX_SAMPLES < BATCH_SIZE )); then
  echo "max_samples must be >= batch size (MAX_SAMPLES=$MAX_SAMPLES BATCH_SIZE=$BATCH_SIZE)" >&2
  exit 2
fi
PREFLIGHT="${SYNCSPEC_PREFLIGHT_OUTPUT:-outputs/syncspec_b200_preflight.json}"
"$FAST_INFER_PYTHON" scripts/check_syncspec_b200.py \
  --target-model "${TARGET_MODEL:-}" \
  --drafter-checkpoint "${DRAFTER_CHECKPOINT:-}" \
  --data-file "${DATA_FILE:-}" \
  --selector-checkpoint "${SELECTOR_CHECKPOINT:-}" \
  --survival-checkpoint "${SURVIVAL_CHECKPOINT:-}" \
  --profile "${PROFILE:-}" \
  --precision "${DTYPE:-bfloat16}" \
  --batch-size "$BATCH_SIZE" \
  --output "$PREFLIGHT" --strict

ARGS=(
  --backend transformers
  --target-model "$TARGET_MODEL"
  --drafter-checkpoint "$DRAFTER_CHECKPOINT"
  --input "$DATA_FILE"
  --output "${OUTPUT_FILE:-outputs/syncspec_b200_smoke.jsonl}"
  --device "${DEVICE:-cuda}"
  --dtype "${DTYPE:-bfloat16}"
  --max-samples "$MAX_SAMPLES"
  --batch-size "${BATCH_SIZE:-1}"
  --max-new-tokens "${MAX_NEW_TOKENS:-8}"
  --max-input-tokens "${MAX_INPUT_TOKENS:-0}"
  --local-files-only
  --smoke
)
if [[ -n "${BUDGET_PROFILES:-}" && -z "${KD:-}" && -z "${KV:-}" ]]; then
  ARGS+=(--budget-profiles "$BUDGET_PROFILES")
else
  ARGS+=(--kd "${KD:-16}" --kv "${KV:-8}")
fi
if [[ "${STOCHASTIC:-0}" != "1" ]]; then ARGS+=(--check-exactness); fi
if [[ -n "${SELECTOR_CHECKPOINT:-}" ]]; then ARGS+=(--selector-checkpoint "$SELECTOR_CHECKPOINT"); fi
if [[ -n "${SURVIVAL_CHECKPOINT:-}" ]]; then ARGS+=(--survival-checkpoint "$SURVIVAL_CHECKPOINT"); fi
if [[ "${STOCHASTIC:-0}" == "1" ]]; then ARGS+=(--stochastic); fi
if [[ -n "${PROFILE:-}" ]]; then ARGS+=(--profile "$PROFILE"); fi
if [[ -n "${GATE_TABLE:-}" ]]; then ARGS+=(--gate-table "$GATE_TABLE"); fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$FAST_INFER_PYTHON" scripts/infer_syncspec.py "${ARGS[@]}" "$@"
