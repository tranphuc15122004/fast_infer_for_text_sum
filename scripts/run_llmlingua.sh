#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config llmlingua || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

: "${DOC_FILE:?DOC_FILE is required}"
: "${OUTPUT_FILE:?OUTPUT_FILE is required}"

cd "$ROOT"

export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$ROOT/externals/LLMLingua${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --doc-file "$DOC_FILE"
  --compression-rate "${COMPRESSION_RATE:-0.5}"
  --max-samples "${MAX_SAMPLES:-3}"
  --max-new-tokens "${MAX_NEW_TOKENS:-64}"
  --device "${DEVICE:-cuda}"
  --output "$OUTPUT_FILE"
)

if [[ -n "${COMPRESSOR_MODEL:-}" ]]; then
  ARGS+=(--compressor-model "$COMPRESSOR_MODEL")
fi

if [[ -n "${TARGET_MODEL:-}" ]]; then
  ARGS+=(--target-model "$TARGET_MODEL")
fi

if [[ -n "${MAX_INPUT_TOKENS:-}" ]]; then
  ARGS+=(--max-input-tokens "$MAX_INPUT_TOKENS")
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  ARGS+=(--smoke)
fi

exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_llmlingua.py" "${ARGS[@]}" "$@"
