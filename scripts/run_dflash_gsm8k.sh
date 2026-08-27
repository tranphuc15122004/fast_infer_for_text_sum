#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$ROOT/config/dflash_gsm8k.env"
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

SMOKE=0
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      SMOKE=1
      shift
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

: "${TARGET_MODEL:?TARGET_MODEL is required}"
: "${DRAFT_MODEL:?DRAFT_MODEL is required}"
: "${BACKEND:?BACKEND is required}"
: "${DATASET:?DATASET is required}"
: "${MAX_NEW_TOKENS:?MAX_NEW_TOKENS is required}"
: "${TEMPERATURE:?TEMPERATURE is required}"

RUN_MAX_SAMPLES="${MAX_SAMPLES:-}"
RUN_MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
if [[ "$SMOKE" == "1" ]]; then
  RUN_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-1}"
  RUN_MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-128}"
fi

ARGS=(
  -m dflash.benchmark
  --backend "$BACKEND"
  --model "$TARGET_MODEL"
  --draft-model "$DRAFT_MODEL"
  --dataset "$DATASET"
  --max-new-tokens "$RUN_MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
)

if [[ -n "$RUN_MAX_SAMPLES" ]]; then
  ARGS+=(--max-samples "$RUN_MAX_SAMPLES")
fi

if [[ -n "${BLOCK_SIZE:-}" ]]; then
  ARGS+=(--block-size "$BLOCK_SIZE")
fi

cd "$ROOT"
exec "$FAST_INFER_PYTHON" "${ARGS[@]}" "${PASSTHROUGH_ARGS[@]}"
