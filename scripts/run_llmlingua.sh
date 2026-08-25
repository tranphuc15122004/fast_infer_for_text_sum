#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT/config/llmlingua.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

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

exec uv run --project "$ROOT" --locked python "$ROOT/scripts/infer_llmlingua.py" "${ARGS[@]}"
