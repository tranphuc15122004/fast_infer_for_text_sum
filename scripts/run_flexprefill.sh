#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT/config/flexprefill.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${MODEL:?MODEL is required}"
: "${OUTPUT_FILE:?OUTPUT_FILE is required}"

cd "$ROOT"
export PYTHONPATH="$ROOT/scripts:$ROOT/externals/FlexPrefill${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --model "$MODEL"
  --pattern "${PATTERN:-flex_prefill}"
  --max-new-tokens "${MAX_NEW_TOKENS:-64}"
  --output "$OUTPUT_FILE"
)
if [[ -n "${DATA_FILE:-}" ]]; then
  ARGS+=(--data-file "$DATA_FILE")
  [[ -n "${MAX_SAMPLES:-}" ]] && ARGS+=(--max-samples "$MAX_SAMPLES")
fi
[[ -n "${MAX_INPUT_TOKENS:-}" ]] && ARGS+=(--max-input-tokens "$MAX_INPUT_TOKENS")
[[ "${SKIP_NAIVE:-0}" == "1" ]] && ARGS+=(--skip-naive)
[[ "${SMOKE:-0}" == "1" ]] && ARGS+=(--smoke)

exec uv run --project "$ROOT/envs/flexprefill" --locked python \
  "$ROOT/scripts/infer_flexprefill.py" "${ARGS[@]}"
