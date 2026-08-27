#!/usr/bin/env bash
# Run one-sample smoke checks with the production B200 profile.
#
# Production uses the `python3` command resolved from PATH.
# Local simulation: FAST_INFER_PYTHON="$PWD/.venv/bin/python" bash ...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINES=""
OUTPUT_DIR=""
TIMEOUT=""
PREFLIGHT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      export FAST_INFER_MASTER_CONFIG="$2"
      shift 2
      ;;
    --baselines)
      BASELINES="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --preflight-only)
      PREFLIGHT_ONLY=1
      shift
      ;;
    -h|--help)
      echo "Usage: bash scripts/run_b200_smoke.sh [--config MASTER] [--baselines LIST] [--output-dir DIR] [--timeout SEC] [--preflight-only]"
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 2
      ;;
  esac
done

# Load the single external master (or the repository pointer) once. Exported
# compatibility aliases are inherited by the coordinator and child launchers.
# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_master || exit 1

# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

[[ -n "$BASELINES" ]] || BASELINES="${B200_BASELINES:-}"
[[ -n "$TIMEOUT" ]] || TIMEOUT="${B200_TIMEOUT_SECONDS:-900}"
[[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="${B200_OUTPUT_DIR:-outputs/b200_smoke}"

if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$ROOT/$OUTPUT_DIR"
fi

runner_args=(
  --root "$ROOT"
  --baselines "$BASELINES"
  --output-dir "$OUTPUT_DIR"
  --timeout "$TIMEOUT"
)
[[ "$PREFLIGHT_ONLY" == "1" ]] && runner_args+=(--preflight-only)

exec "$FAST_INFER_PYTHON" "$ROOT/scripts/b200_smoke.py" "${runner_args[@]}"
