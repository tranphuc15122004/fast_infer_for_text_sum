#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

source "$ROOT/scripts/common/config.sh"
fast_infer_load_config domino || exit 1
source "$ROOT/scripts/common/runtime.sh" || exit 1

: "${TARGET_MODEL:?TARGET_MODEL is required}"
: "${DRAFT_MODEL:?DRAFT_MODEL/ MODEL_DOMINO is required}"
: "${DATA_FILE:?DATA_FILE is required}"
: "${OUTPUT_FILE:?OUTPUT_FILE is required}"

ARGS=(
  --target-model "$TARGET_MODEL"
  --draft-model "$DRAFT_MODEL"
  --data-file "$DATA_FILE"
  --max-samples "${MAX_SAMPLES:-5}"
  --max-new-tokens "${MAX_NEW_TOKENS:-64}"
  --max-input-tokens "${MAX_INPUT_TOKENS:-0}"
  --temperature "${TEMPERATURE:-0}"
  --attention-backend "${ATTENTION_BACKEND:-sdpa}"
  --output "$OUTPUT_FILE"
)
[[ "${SMOKE:-0}" == "1" ]] && ARGS+=(--smoke)
[[ -n "${BLOCK_SIZE:-}" ]] && ARGS+=(--block-size "$BLOCK_SIZE")

cd "$ROOT"
export PYTHONPATH="$ROOT/scripts:$ROOT/externals/Domino/code${PYTHONPATH:+:$PYTHONPATH}"
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_domino.py" "${ARGS[@]}" "$@"
