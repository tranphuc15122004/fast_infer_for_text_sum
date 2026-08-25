#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT/config/rocketkv.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

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

exec uv run --project "$ROOT/envs/legacy" --locked python "$ROOT/scripts/infer_rocketkv.py" "${ARGS[@]}"
