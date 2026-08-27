#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config specprefill || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

: "${TARGET_MODEL:?TARGET_MODEL is required}"
: "${SPEC_MODEL:?SPEC_MODEL is required}"
: "${OUTPUT_FILE:?OUTPUT_FILE is required}"

# The repo's default spec config path is relative to the repo root.
REPO="$ROOT/externals/speculative_prefill"
if [[ -n "${DATA_FILE:-}" && "$DATA_FILE" != /* ]]; then
  DATA_FILE="$ROOT/$DATA_FILE"
fi
cd "$REPO"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --target-model "$TARGET_MODEL"
  --spec-model "$SPEC_MODEL"
  --spec-config "configs/${SPEC_CONFIG:-config_p1_full_lah8.yaml}"
  --max-tokens "${MAX_TOKENS:-64}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}"
  --output "$OUTPUT_FILE"
)
[[ -n "${DATA_FILE:-}" ]] && {
  ARGS+=(--data-file "$DATA_FILE")
  [[ -n "${MAX_SAMPLES:-}" ]] && ARGS+=(--max-samples "$MAX_SAMPLES")
}
[[ "${SMOKE:-1}" == "1" ]] && ARGS+=(--smoke)

exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_specprefill.py" "${ARGS[@]}" "$@"
