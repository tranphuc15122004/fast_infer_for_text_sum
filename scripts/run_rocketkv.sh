#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config rocketkv || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

: "${OUTPUT_FILE:?OUTPUT_FILE is required}"

# RocketKV's gpt-fast files do `import rocket` / `import model` from the
# gpt-fast dir, so that dir must be on PYTHONPATH (script files do not add
# cwd to sys.path). We do NOT cd away from the repo root so relative
# --output paths resolve to outputs/.
REPO="$ROOT/externals/RocketKV"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$REPO/gpt-fast${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --token-budget "${TOKEN_BUDGET:-512}"
  --seq-len "${SEQ_LEN:-2048}"
  --max-new-tokens "${MAX_NEW_TOKENS:-64}"
  --head-dim "${HEAD_DIM:-128}"
  --num-runs "${NUM_RUNS:-2}"
  --output "$OUTPUT_FILE"
)
[[ "${SMOKE:-1}" == "1" ]] && ARGS+=(--smoke)
[[ "${FULL:-0}" == "1" ]] && ARGS+=(--full)

exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_rocketkv.py" "${ARGS[@]}" "$@"
