#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$ROOT/config/minference.env"
if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG_FILE="$1"
  shift
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

: "${MODEL:?MODEL is required}"
: "${OUTPUT_FILE:?OUTPUT_FILE is required}"

cd "$ROOT"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$ROOT/externals/MInference${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --model "$MODEL"
  --attn-type "${ATTN_TYPE:-minference}"
  --max-new-tokens "${MAX_NEW_TOKENS:-32}"
  --max-model-len "${MAX_MODEL_LEN:-8192}"
  --device "${DEVICE:-cuda}"
  --attn-implementation "${ATTN_IMPLEMENTATION:-auto}"
  --output "$OUTPUT_FILE"
)
[[ -n "${DATA_FILE:-}" ]] && {
  ARGS+=(--data-file "$DATA_FILE")
  [[ -n "${MAX_SAMPLES:-}" ]] && ARGS+=(--max-samples "$MAX_SAMPLES")
}
[[ -n "${MAX_INPUT_TOKENS:-}" ]] && ARGS+=(--max-input-tokens "$MAX_INPUT_TOKENS")
[[ "${SMOKE:-1}" == "1" ]] && ARGS+=(--smoke)

exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_minference.py" "${ARGS[@]}" "$@"
