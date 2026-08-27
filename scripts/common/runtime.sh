#!/usr/bin/env bash
# Shared Python runtime for every benchmark launcher.
#
# This file is meant to be sourced after ROOT has been defined. It can also
# derive ROOT from its own location when used by a standalone helper.

if [[ -z "${ROOT:-}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

# A coordinator may provide a temporary overlay after the baseline config has
# been sourced. This lets the production B200 runner keep one shared profile
# while preserving each launcher's normal config boundary.
if [[ -n "${FAST_INFER_CONFIG_OVERLAY:-}" ]]; then
  if [[ ! -f "$FAST_INFER_CONFIG_OVERLAY" ]]; then
    echo "Shared config overlay not found: $FAST_INFER_CONFIG_OVERLAY" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source "$FAST_INFER_CONFIG_OVERLAY"
fi

fast_infer_resolve_python() {
  local candidate=""

  if [[ -n "${FAST_INFER_PYTHON:-}" ]]; then
    candidate="$FAST_INFER_PYTHON"
  elif [[ -n "${FAST_INFER_VENV:-}" ]]; then
    candidate="$FAST_INFER_VENV/bin/python"
  elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    candidate="$VIRTUAL_ENV/bin/python"
  elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    candidate="$ROOT/.venv/bin/python"
  else
    # A production server may intentionally have no project venv. In that
    # case use the Python 3.12 command provided by the image/PATH.
    candidate="${FAST_INFER_SYSTEM_PYTHON:-python3}"
  fi

  # Production B200 images expose the shared interpreter as `python3` on PATH,
  # while local simulation commonly supplies an absolute .venv path. Resolve
  # command names once so every child process receives an executable path.
  if [[ "$candidate" != */* ]]; then
    candidate="$(command -v "$candidate" 2>/dev/null || true)"
  fi

  if [[ ! -x "$candidate" ]]; then
    echo "Shared Python interpreter not found or not executable: $candidate" >&2
    echo "Create it with: bash $ROOT/scripts/setup_venv.sh --offline" >&2
    return 1
  fi

  printf '%s\n' "$candidate"
}

fast_infer_require_python312() {
  local selected
  selected="$(fast_infer_resolve_python)" || return 1

  if ! "$selected" -c 'import sys; sys.exit(1) if sys.version_info[:2] != (3, 12) else None'; then
    echo "Shared runtime must use Python 3.12: $selected" >&2
    return 1
  fi

  export FAST_INFER_PYTHON="$selected"
}

fast_infer_require_python312
