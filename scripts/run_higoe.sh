#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT/config/higoe.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${OUTPUT_FILE:?OUTPUT_FILE is required}"

# HiGOE is not a package: it imports sibling modules from the repo root.
# cwd is NOT added to sys.path when running `python /abs/script.py`, so the
# HiGOE dir itself must be on PYTHONPATH too.
cd "$ROOT/externals/HiGOE"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$ROOT/externals/HiGOE${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --retriever-model "${RETRIEVER_MODEL:-facebook/contriever}"
  --num-docs "${NUM_DOCS:-3}"
  --output "$OUTPUT_FILE"
)
[[ "${SMOKE:-1}" == "1" ]] && ARGS+=(--smoke)

exec uv run --project "$ROOT/envs/legacy" --locked python "$ROOT/scripts/infer_higoe.py" "${ARGS[@]}"
