#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config specextend || exit 1
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
