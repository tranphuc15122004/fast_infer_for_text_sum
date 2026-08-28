#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config sssd || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

: "${OUTPUT_FILE:?OUTPUT_FILE is required}"
: "${MODEL:?MODEL is required}"

cd "$ROOT"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --model "$MODEL"
  --max-samples "${MAX_SAMPLES:-1}"
  --max-new-tokens "${MAX_NEW_TOKENS:-32}"
  --datastore-path "${DATASTORE_PATH:-}"
  --num-draft-tokens "${NUM_DRAFT_TOKENS:-8}"
  --num-steps "${NUM_STEPS:-5}"
  --topk "${TOPK:-5}"
  --output "$OUTPUT_FILE"
)
[[ -n "${DATA_FILE:-}" ]] && ARGS+=(--data-file "$DATA_FILE")
[[ "${ADAPTIVE:-0}" == "1" ]] && ARGS+=(--adaptive)
[[ "${SMOKE:-0}" == "1" ]] && ARGS+=(--smoke)

exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_sssd.py" "${ARGS[@]}" "$@"
