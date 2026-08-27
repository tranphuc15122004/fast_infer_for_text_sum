#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$ROOT/config/specextend.env"
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

: "${OUTPUT_FILE:?OUTPUT_FILE is required}"

cd "$ROOT"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --script "${SCRIPT:-run_classic.py}"
  --model-name "${MODEL_NAME:-llama3_1_8b}"
  --input-file "${INPUT_FILE:-data/govreport/govreport_512.jsonl}"
  --max-samples "${MAX_SAMPLES:-1}"
  --max-gen-len "${MAX_GEN_LEN:-64}"
  --max-input-tokens "${MAX_INPUT_TOKENS:-0}"
  --warmup-runs "${WARMUP_RUNS:-3}"
  --output "$OUTPUT_FILE"
)
[[ -n "${BASE_MODEL:-}" ]] && ARGS+=(--base-model "$BASE_MODEL")
[[ -n "${DRAFT_MODEL:-}" ]] && ARGS+=(--draft-model "$DRAFT_MODEL")
if [[ "${USE_SPECEXTEND:-1}" == "1" ]]; then
  ARGS+=(--use-specextend)
else
  ARGS+=(--no-use-specextend)
fi
[[ "${SMOKE:-1}" == "1" ]] && ARGS+=(--smoke)

exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_specextend.py" "${ARGS[@]}" "$@"
