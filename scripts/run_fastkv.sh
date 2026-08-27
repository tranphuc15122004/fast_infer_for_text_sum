#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config fastkv || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

: "${OUTPUT_FILE:?OUTPUT_FILE is required}"

cd "$ROOT"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$ROOT/externals/FastKV${PYTHONPATH:+:$PYTHONPATH}"
export FASTKV_DATA_ROOT="${FASTKV_DATA_ROOT:-${HF_HOME:-$HOME/.cache/huggingface}/datasets/fast_infer_text_sum/FastKV/data}"

ARGS=(
  --max-new-tokens "${MAX_NEW_TOKENS:-64}"
  --window-size "${WINDOW_SIZE:-1024}"
  --max-capacity-prompts "${MAX_CAPACITY_PROMPTS:-2048}"
  --retain-rate "${RETAIN_RATE:-0.1}"
  --eviction-mode "${EVICTION_MODE:-proportional}"
  --num-runs "${NUM_RUNS:-2}"
  --output "$OUTPUT_FILE"
)

[[ -n "${MODEL:-}" ]] && ARGS+=(--model "$MODEL")
[[ -n "${METHOD:-}" ]] && ARGS+=(--method "$METHOD")
[[ -n "${ATTN_IMPL:-}" ]] && ARGS+=(--attn-implementation "$ATTN_IMPL")
if [[ -n "${DATA_FILE:-}" ]]; then
  ARGS+=(--data-file "$DATA_FILE")
  [[ -n "${MAX_SAMPLES:-}" ]] && ARGS+=(--max-samples "$MAX_SAMPLES")
fi
[[ "${SMOKE:-0}" == "1" ]] && ARGS+=(--smoke)

exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_fastkv.py" "${ARGS[@]}" "$@"
