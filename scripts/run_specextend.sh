#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT/config/specextend.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${OUTPUT_FILE:?OUTPUT_FILE is required}"

cd "$ROOT"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --script "${SCRIPT:-run_classic.py}"
  --model-name "${MODEL_NAME:-vicuna_7b}"
  --input-file "${INPUT_FILE:-data/govreport/govreport_512.jsonl}"
  --max-samples "${MAX_SAMPLES:-1}"
  --max-gen-len "${MAX_GEN_LEN:-64}"
  --output "$OUTPUT_FILE"
)
[[ -n "${BASE_MODEL:-}" ]] && ARGS+=(--base-model "$BASE_MODEL")
[[ -n "${DRAFT_MODEL:-}" ]] && ARGS+=(--draft-model "$DRAFT_MODEL")
[[ "${USE_SPECEXTEND:-1}" == "1" ]] && ARGS+=(--use-specextend)
[[ "${SMOKE:-1}" == "1" ]] && ARGS+=(--smoke)

exec uv run --project "$ROOT/envs/legacy" --locked python "$ROOT/scripts/infer_specextend.py" "${ARGS[@]}"
