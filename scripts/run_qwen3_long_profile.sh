#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT/config/qwen3_long_profile.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${MODEL:?MODEL is required}"
: "${INPUT_FILE:?INPUT_FILE is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

cd "$ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/fast_infer_uv_cache}"
mkdir -p "$UV_CACHE_DIR"

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

exec uv run --project "$ROOT" --locked python \
  "$ROOT/src/analyze/full_infer/profile_qwen3_long_summary.py" "${ARGS[@]}"
