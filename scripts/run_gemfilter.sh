#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config gemfilter || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

: "${OUTPUT_FILE:?OUTPUT_FILE is required}"

# GemFilter is not a pip package; it imports my_utils / my_baseline from the
# repo root, so PYTHONPATH points there (and to scripts/ for common helpers).
cd "$ROOT"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$ROOT/externals/GemFilter${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --topk "${TOPK:-1024}"
  --max-gen-len "${MAX_GEN_LEN:-32}"
  --num-runs "${NUM_RUNS:-2}"
  --output "$OUTPUT_FILE"
)
[[ -n "${MODEL:-}" ]] && ARGS+=(--model "$MODEL")
if [[ -n "${DATA_FILE:-}" ]]; then
  ARGS+=(--data-file "$DATA_FILE")
  [[ -n "${MAX_SAMPLES:-}" ]] && ARGS+=(--max-samples "$MAX_SAMPLES")
fi
[[ -n "${SELECT_LAYER_IDX:-}" ]] && ARGS+=(--select-layer-idx "$SELECT_LAYER_IDX")
[[ -n "${PROMPT:-}" ]] && ARGS+=(--prompt "$PROMPT")
[[ "${SMOKE:-1}" == "1" ]] && ARGS+=(--smoke)

exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_gemfilter.py" "${ARGS[@]}" "$@"
