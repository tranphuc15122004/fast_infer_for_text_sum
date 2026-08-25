#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT/config/longspec.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

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
exec uv run --project "$ROOT/envs/longspec" --locked python "$ROOT/scripts/infer_longspec.py" "${ARGS[@]}"
