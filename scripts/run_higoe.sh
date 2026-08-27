#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config higoe || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

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

exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_higoe.py" "${ARGS[@]}" "$@"
