#!/usr/bin/env bash
# Create an isolated Python 3.12 environment for the B200/cu130 server.
#
# Online mode uses the indexes configured by pip on the host. Offline mode
# uses only the wheelhouse supplied through B200_WHEELHOUSE.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${FAST_INFER_B200_VENV:-$ROOT/.venv-b200}"
PYTHON_BIN="${FAST_INFER_B200_PYTHON:-python3}"
WHEELHOUSE="${B200_WHEELHOUSE:-}"
OFFLINE="${B200_OFFLINE:-0}"
SKIP_FLASH_ATTN="${FAST_INFER_SKIP_FLASH_ATTN:-0}"
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage: scripts/setup_b200_venv.sh [--check] [--offline]

Environment:
  FAST_INFER_B200_VENV   venv target (default: <repo>/.venv-b200)
  FAST_INFER_B200_PYTHON Python 3.12 executable (default: python3)
  B200_WHEELHOUSE        directory containing all offline wheels
  B200_OFFLINE=1         install with --no-index --find-links
  FAST_INFER_SKIP_FLASH_ATTN=1
                         skip the source build (not suitable for full eval)
EOF
}

die() {
  echo "setup_b200_venv: $*" >&2
  exit 1
}

for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    --offline) OFFLINE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

if [[ "$CHECK_ONLY" == "1" ]]; then
  [[ -x "$VENV_DIR/bin/python" ]] \
    || die "venv is missing: $VENV_DIR"
  "$VENV_DIR/bin/python" -c \
    'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' \
    || die "existing venv is not Python 3.12: $VENV_DIR"
  "$VENV_DIR/bin/python" -m pip --version
  echo "B200 venv OK: $VENV_DIR"
  exit 0
fi

if [[ "$PYTHON_BIN" != */* ]]; then
  PYTHON_BIN="$(command -v "$PYTHON_BIN" 2>/dev/null || true)"
fi
[[ -x "$PYTHON_BIN" ]] || die "Python executable not found: $PYTHON_BIN"

"$PYTHON_BIN" -c \
  'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' \
  || die "B200 venv requires Python 3.12: $PYTHON_BIN"

[[ ! -e "$VENV_DIR" ]] \
  || die "target already exists; choose FAST_INFER_B200_VENV or remove it explicitly: $VENV_DIR"

echo "Creating Python 3.12 venv: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
VENV_PYTHON="$VENV_DIR/bin/python"

# Some minimal Python installations create a venv without pip.
"$VENV_PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$VENV_PYTHON" -m pip --version >/dev/null \
  || die "venv does not contain pip; install python3.12-venv on the server"

REQUIREMENTS_NO_FLASH_ATTN="$(mktemp "${TMPDIR:-/tmp}/fast-infer-requirements.XXXXXX")"
cleanup() {
  rm -f "$REQUIREMENTS_NO_FLASH_ATTN"
}
trap cleanup EXIT

# flash-attn is optional and intentionally absent from requirements.txt: the
# mirrored artifact is an sdist whose setup.py must be patched for Torch
# 2.14/C++20. Install the manifest first, then use the isolated helper.
awk '!/^flash-attn==/' "$ROOT/requirements.txt" > "$REQUIREMENTS_NO_FLASH_ATTN"
PIP_ARGS=(install --prefer-binary -r "$REQUIREMENTS_NO_FLASH_ATTN")
if [[ "$OFFLINE" == "1" ]]; then
  [[ -n "$WHEELHOUSE" ]] || die "B200_WHEELHOUSE is required in offline mode"
  [[ -d "$WHEELHOUSE" ]] || die "wheelhouse directory not found: $WHEELHOUSE"
  PIP_ARGS+=(--no-index --find-links "$WHEELHOUSE")
elif [[ -n "$WHEELHOUSE" ]]; then
  PIP_ARGS+=(--find-links "$WHEELHOUSE")
fi

echo "Installing requirements manifest into $VENV_PYTHON (FlashAttention is built separately)"
PIP_DISABLE_PIP_VERSION_CHECK=1 \
  "$VENV_PYTHON" -m pip "${PIP_ARGS[@]}"

if [[ "$SKIP_FLASH_ATTN" != "1" ]]; then
  FLASH_ATTN_ARGS=(
    "$ROOT/scripts/install_flash_attn_b200.sh"
    --python "$VENV_PYTHON"
  )
  [[ "$OFFLINE" == "1" ]] && FLASH_ATTN_ARGS+=(--offline)
  B200_WHEELHOUSE="$WHEELHOUSE" \
    B200_OFFLINE="$OFFLINE" \
    "${FLASH_ATTN_ARGS[@]}"
else
  echo "Skipping flash-attn source build (FAST_INFER_SKIP_FLASH_ATTN=1)"
fi

cat <<EOF

B200 venv ready: $VENV_DIR
Activate with:
  source "$VENV_DIR/bin/activate"

Before SGLang MR-DFlash cache, ensure the system packages exist:
  libnuma1 libnuma-dev
and keep the SpecForge PYTHONPATH free of externals/SSSD/python.
EOF
