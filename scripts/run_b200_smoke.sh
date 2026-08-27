#!/usr/bin/env bash
# Run one-sample smoke checks with the production B200 profile.
#
# Production uses the `python3` command resolved from PATH.
# Local simulation: FAST_INFER_PYTHON="$PWD/.venv/bin/python" bash ...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$ROOT/config/b200.env"
BASELINES=""
OUTPUT_DIR="outputs/b200_smoke"
TIMEOUT=""
PREFLIGHT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="$2"
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
      echo "Usage: bash scripts/run_b200_smoke.sh [--config FILE] [--baselines LIST] [--output-dir DIR] [--timeout SEC] [--preflight-only]"
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$CONFIG_FILE" != /* ]]; then
  CONFIG_FILE="$ROOT/$CONFIG_FILE"
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

# Export profile values so the Python coordinator and every child launcher see
# the same production/simulation settings.
set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

[[ -n "$BASELINES" ]] || BASELINES="${B200_BASELINES:-}"
[[ -n "$TIMEOUT" ]] || TIMEOUT="${B200_TIMEOUT_SECONDS:-900}"

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
