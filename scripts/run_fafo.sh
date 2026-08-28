#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config fafo || exit 1
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
  --kv-method "${KV_METHOD:-stream-llm}"
  --output "$OUTPUT_FILE"
)
[[ -n "${DATA_FILE:-}" ]] && ARGS+=(--data-file "$DATA_FILE")
if [[ "${USE_FLASH:-0}" == "1" ]]; then
  ARGS+=(--use-flash)
else
  ARGS+=(--no-use-flash)
fi
[[ "${SMOKE:-0}" == "1" ]] && ARGS+=(--smoke)

exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_fafo.py" "${ARGS[@]}" "$@"
