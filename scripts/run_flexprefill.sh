#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config flexprefill || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

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

exec "$FAST_INFER_PYTHON" \
  "$ROOT/scripts/infer_flexprefill.py" "${ARGS[@]}" "$@"
