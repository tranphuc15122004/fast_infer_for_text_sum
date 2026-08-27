#!/usr/bin/env bash
# Create/check the single Python environment used by this repository.
#
# The command is offline-first. Python 3.12 and every wheel referenced by
# requirements.txt must already be installed or available in uv's local cache.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${FAST_INFER_VENV:-$ROOT/.venv}"
RECREATE=0
CHECK_ONLY=0

usage() {
  echo "Usage: $0 [--offline] [--recreate] [--check]"
}

for arg in "$@"; do
  case "$arg" in
    --offline) ;;
    --recreate) RECREATE=1 ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

die() {
  echo "setup_venv: $*" >&2
  exit 1
}

if [[ -n "${FAST_INFER_UV_CACHE:-}" ]]; then
  export UV_CACHE_DIR="$FAST_INFER_UV_CACHE"
  mkdir -p "$UV_CACHE_DIR"
fi

check_python312() {
  local python_bin="$1"
  [[ -x "$python_bin" ]] || return 1
  "$python_bin" -c 'import sys; sys.exit(1) if sys.version_info[:2] != (3, 12) else None'
}

check_local_requirement_sources() {
  local requirements_file="$1"
  local line source
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    source=""
    if [[ "$line" == *"@ file://"* ]]; then
      source="${line##*@ file://}"
    elif [[ "$line" == -e\ /* ]]; then
      source="${line#-e }"
    fi
    if [[ -n "$source" && ! -e "$source" ]]; then
      die "local requirement source missing: $source"
    fi
  done < "$requirements_file"
}

check_offline_requirement_urls() {
  local requirements_file="$1"
  local line
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    if [[ "$line" == *"@ https://"* || "$line" == *"@ http://"* ]]; then
      echo "setup_venv: offline direct URL requires a matching artifact in the uv cache: $line" >&2
    fi
  done < "$requirements_file"
}

ensure_uv_lock_from_requirements() {
  echo "Refreshing uv project manifest/lock from requirements.txt (offline)"
  uv add \
    --requirements "$ROOT/requirements.txt" \
    --no-sync \
    --offline \
    --no-python-downloads \
    --python "$PYTHON312"
}

[[ -f "$ROOT/requirements.txt" ]] \
  || die "requirements.txt not found at $ROOT/requirements.txt"
check_local_requirement_sources "$ROOT/requirements.txt"
check_offline_requirement_urls "$ROOT/requirements.txt"

if [[ "$CHECK_ONLY" == "1" ]]; then
  check_python312 "$VENV_DIR/bin/python" \
    || die "shared venv is missing or is not Python 3.12: $VENV_DIR"
  echo "Shared venv OK: $VENV_DIR/bin/python"
  exit 0
fi

if [[ -n "${PYTHON312_BIN:-}" ]]; then
  PYTHON312="$PYTHON312_BIN"
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON312="$(command -v python3.12)"
else
  PYTHON312=""
  if command -v uv >/dev/null 2>&1; then
    # `uv python find` is a local lookup. It must not download a managed
    # interpreter; --offline is enforced for package installation below.
    PYTHON312="$(uv python find 3.12 2>/dev/null || true)"
  fi
fi

[[ -n "$PYTHON312" ]] || die "Python 3.12 was not found; set PYTHON312_BIN to an installed interpreter"
check_python312 "$PYTHON312" \
  || die "PYTHON312_BIN is not Python 3.12: $PYTHON312"
command -v uv >/dev/null 2>&1 \
  || die "uv is required for offline venv setup"

ensure_uv_lock_from_requirements

if [[ -e "$VENV_DIR" && "$RECREATE" != "1" ]]; then
  check_python312 "$VENV_DIR/bin/python" \
    || die "$VENV_DIR already exists but is not a Python 3.12 venv; rerun with --recreate"
else
  mkdir -p "$(dirname "$VENV_DIR")"
  UV_VENV_ARGS=(venv --python "$PYTHON312")
  [[ "$RECREATE" == "1" ]] && UV_VENV_ARGS+=(--clear)
  uv "${UV_VENV_ARGS[@]}" "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
check_python312 "$VENV_PYTHON" \
  || die "created venv is not Python 3.12: $VENV_PYTHON"

echo "Installing requirements.txt into $VENV_PYTHON (offline)"
uv pip install --offline --python "$VENV_PYTHON" -r "$ROOT/requirements.txt"
echo "Shared venv ready: $VENV_PYTHON"
