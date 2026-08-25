#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT/config/gemfilter.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

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

exec uv run --project "$ROOT/envs/legacy" --locked python "$ROOT/scripts/infer_gemfilter.py" "${ARGS[@]}"
