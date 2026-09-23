#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/common/config.sh" || exit 1
fast_infer_load_master || exit 1
source "$ROOT/scripts/common/runtime.sh" || exit 1

cd "$ROOT"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/run_qwen3_4b_paired.py" "$@"
