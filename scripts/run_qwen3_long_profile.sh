#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$ROOT/config/qwen3_long_profile.env"
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
: "${INPUT_FILE:?INPUT_FILE is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

cd "$ROOT"
read -r -a WORD_MARK_ARGS <<< "${WORD_MARKS:-256 512 1024 2048 3072}"
ARGS=(
  --model "$MODEL"
  --input "$INPUT_FILE"
  --output-dir "$OUTPUT_DIR"
  --word-marks "${WORD_MARK_ARGS[@]}"
  --max-new-tokens "${MAX_NEW_TOKENS:-128}"
  --repeats "${REPEATS:-3}"
  --warmup-runs "${WARMUP_RUNS:-1}"
  --device "${DEVICE:-cuda:0}"
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
)
[[ "${LOCAL_FILES_ONLY:-0}" == "1" ]] && ARGS+=(--local-files-only)

exec "$FAST_INFER_PYTHON" \
  "$ROOT/src/analyze/full_infer/profile_qwen3_long_summary.py" "${ARGS[@]}" "$@"
