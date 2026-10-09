#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMAND="infer"

if [[ $# -gt 0 && "$1" != -* ]]; then
  case "$1" in
    preflight|prepare-data|capture|candidates|label|train-selector|train-compressor|evaluate-fixed|infer)
      COMMAND="$1"
      shift
      ;;
    *)
      export FAST_INFER_MASTER_CONFIG="$1"
      shift
      if [[ $# -gt 0 && "$1" != -* ]]; then
        COMMAND="$1"
        shift
      fi
      ;;
  esac
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config amr_dflash || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

if [[ -n "${AMR_CONFIG:-}" ]]; then
  CONFIG_PATH="$AMR_CONFIG"
else
  CONFIG_PATH="$ROOT/src/AMR_DFlash/configs/pilot.yaml"
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/externals/dflash${PYTHONPATH:+:$PYTHONPATH}"
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/amr_dflash/cli.py" \
  --config "$CONFIG_PATH" \
  --device "${AMR_DEVICE:-${FI_DEVICE:-cuda}}" \
  "$COMMAND" "$@"
