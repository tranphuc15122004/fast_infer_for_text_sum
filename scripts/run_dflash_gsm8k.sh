#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT/config/dflash_gsm8k.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${TARGET_MODEL:?TARGET_MODEL is required}"
: "${DRAFT_MODEL:?DRAFT_MODEL is required}"
: "${BACKEND:?BACKEND is required}"
: "${DATASET:?DATASET is required}"
: "${MAX_NEW_TOKENS:?MAX_NEW_TOKENS is required}"
: "${TEMPERATURE:?TEMPERATURE is required}"

ARGS=(
  -m dflash.benchmark
  --backend "$BACKEND"
  --model "$TARGET_MODEL"
  --draft-model "$DRAFT_MODEL"
  --dataset "$DATASET"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
)

if [[ -n "${MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi

if [[ -n "${BLOCK_SIZE:-}" ]]; then
  ARGS+=(--block-size "$BLOCK_SIZE")
fi

cd "$ROOT"
exec uv run --project "$ROOT" --locked python "${ARGS[@]}"
