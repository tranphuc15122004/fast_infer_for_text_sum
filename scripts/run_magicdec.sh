#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT/config/magicdec.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${OUTPUT_FILE:?OUTPUT_FILE is required}"
: "${MODEL_PTH:?MODEL_PTH is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"

cd "$ROOT"

if [[ "${PREPARE_CHECKPOINT:-0}" == "1" ]]; then
  export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
  uv run --project "$ROOT/envs/magicdec" --locked python "$ROOT/scripts/magicdec_prepare_checkpoint.py" \
    --repo-id "${REPO_ID:?REPO_ID required when PREPARE_CHECKPOINT=1}" \
    --model-key "${MODEL_KEY:?MODEL_KEY required when PREPARE_CHECKPOINT=1}" \
    --out-dir "$(dirname "$MODEL_PTH")"
fi

if [[ ! -f "$MODEL_PTH" ]]; then
  echo "model.pth not found: $MODEL_PTH (set PREPARE_CHECKPOINT=1 to build it)" >&2
  exit 1
fi

export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --model-pth "$MODEL_PTH"
  --model-name "$MODEL_NAME"
  --batch-size "${BATCH_SIZE:-1}"
  --prefix-len "${PREFIX_LEN:-2048}"
  --max-len "${MAX_LEN:-2176}"
  --num-runs "${NUM_RUNS:-1}"
  --output "$OUTPUT_FILE"
)
[[ "${SELF_SPEC:-0}" == "1" ]] && ARGS+=(--self-spec --gamma "${GAMMA:-3}" --draft-budget "${DRAFT_BUDGET:-257}")
[[ "${SMOKE:-1}" == "1" ]] && ARGS+=(--smoke)

exec uv run --project "$ROOT/envs/magicdec" --locked python "$ROOT/scripts/infer_magicdec.py" "${ARGS[@]}"
