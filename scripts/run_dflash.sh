#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$ROOT/config/dflash.env"
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

# Preserve the existing GSM8K entry point when callers explicitly provide one
# of the legacy DFlash configs (they contain BACKEND/DATASET instead of the
# representative JSONL fields below).
if [[ -n "${BACKEND:-}" && -n "${DATASET:-}" ]]; then
  exec bash "$ROOT/scripts/run_dflash_gsm8k.sh" "$CONFIG_FILE" "$@"
fi

: "${TARGET_MODEL:?TARGET_MODEL is required}"
: "${DRAFT_MODEL:?DRAFT_MODEL is required}"
: "${OUTPUT_FILE:?OUTPUT_FILE is required}"

ARGS=(
  --target-model "$TARGET_MODEL"
  --draft-model "$DRAFT_MODEL"
  --max-new-tokens "${MAX_NEW_TOKENS:-64}"
  --temperature "${TEMPERATURE:-0}"
  --output "$OUTPUT_FILE"
)
if [[ -n "${DATA_FILE:-}" ]]; then
  ARGS+=(--data-file "$DATA_FILE" --max-samples "${MAX_SAMPLES:-5}")
else
  ARGS+=(--prompt "${PROMPT:-The capital of France is}")
fi
[[ -n "${BLOCK_SIZE:-}" ]] && ARGS+=(--block-size "$BLOCK_SIZE")
[[ -n "${MAX_INPUT_TOKENS:-}" ]] && ARGS+=(--max-input-tokens "$MAX_INPUT_TOKENS")
[[ "${SMOKE:-0}" == "1" ]] && ARGS+=(--smoke)

cd "$ROOT"
export PYTHONPATH="$ROOT/scripts:$ROOT/externals/dflash${PYTHONPATH:+:$PYTHONPATH}"
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_dflash.py" "${ARGS[@]}" "$@"
