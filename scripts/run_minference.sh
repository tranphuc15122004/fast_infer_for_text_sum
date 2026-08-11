#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT/config/minference.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

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
  --output "$OUTPUT_FILE"
)
[[ -n "${DATA_FILE:-}" ]] && {
  ARGS+=(--data-file "$DATA_FILE")
  [[ -n "${MAX_SAMPLES:-}" ]] && ARGS+=(--max-samples "$MAX_SAMPLES")
}
[[ "${SMOKE:-1}" == "1" ]] && ARGS+=(--smoke)

exec uv run --project "$ROOT/envs/specprefill" --locked python "$ROOT/scripts/infer_minference.py" "${ARGS[@]}"
