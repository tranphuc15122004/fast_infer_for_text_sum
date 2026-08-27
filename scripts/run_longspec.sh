#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config longspec || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

: "${OUTPUT_FILE:?OUTPUT_FILE is required}"

cd "$ROOT"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(--output "$OUTPUT_FILE")
if [[ -n "${DATA_FILE:-}" ]]; then
  : "${TARGET_MODEL:?TARGET_MODEL required when DATA_FILE is set}"
  : "${DRAFT_MODEL:?DRAFT_MODEL required when DATA_FILE is set}"
  ARGS+=(--data-file "$DATA_FILE"
         --max-samples "${MAX_SAMPLES:-5}"
         --model-name "${MODEL_NAME:-vicuna7b}"
         --target-model "$TARGET_MODEL"
         --draft-model "$DRAFT_MODEL"
         --max-gen-len "${MAX_GEN_LEN:-64}")
  [[ "${SMOKE:-0}" == "1" ]] && ARGS+=(--smoke)
elif [[ "${FULL:-0}" == "1" ]]; then
  ARGS+=(--full --model-name "${MODEL_NAME:-llama8b}" --method "${METHOD:-tree}"
         --task "${TASK:-gov_report}" --max-gen-len "${MAX_GEN_LEN:-1024}"
         --tree-shape "${TREE_SHAPE:-4 16 16 16 16}")
  : "${DATA_PATH_PREFIX:?DATA_PATH_PREFIX required when FULL=1}"
  ARGS+=(--data-path-prefix "$DATA_PATH_PREFIX")
else
  ARGS+=(--smoke)
fi

# LongSpec imports run from its test dir.
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_longspec.py" "${ARGS[@]}" "$@"
