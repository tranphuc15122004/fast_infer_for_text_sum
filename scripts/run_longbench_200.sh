#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Optional positional/config flag override the repository's master pointer.
if [[ "${1:-}" == "--config" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "--config requires a master env path" >&2
    exit 2
  fi
  export FAST_INFER_MASTER_CONFIG="$2"
  shift 2
elif [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh" || exit 1
fast_infer_load_config longbench || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

cd "$ROOT"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/run_longbench_200.py" "$@"
