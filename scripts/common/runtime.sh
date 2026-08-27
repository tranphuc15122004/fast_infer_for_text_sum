#!/usr/bin/env bash
# Shared Python runtime for every benchmark launcher.
#
# This file is meant to be sourced after ROOT has been defined. It can also
# derive ROOT from its own location when used by a standalone helper.

if [[ -z "${ROOT:-}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

fast_infer_resolve_python() {
  local candidate=""

  if [[ -n "${FAST_INFER_PYTHON:-}" ]]; then
    candidate="$FAST_INFER_PYTHON"
  elif [[ -n "${FAST_INFER_VENV:-}" ]]; then
    candidate="$FAST_INFER_VENV/bin/python"
  elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    candidate="$VIRTUAL_ENV/bin/python"
  else
    candidate="$ROOT/.venv/bin/python"
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
