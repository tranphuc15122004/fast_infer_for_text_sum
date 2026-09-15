#!/usr/bin/env bash
# Build the mirrored FlashAttention sdist against the installed Torch 2.14.
#
# The mirror currently exposes flash-attn 2.8.3.post1 as an sdist.  Its
# upstream setup.py still passes C++17, while Torch 2.14 headers use C++20.
# This helper patches only the unpacked temporary copy and never touches the
# repository or reaches a public wheel URL.
set -euo pipefail

PYTHON_BIN="${FAST_INFER_PYTHON:-python3}"
WHEELHOUSE="${B200_WHEELHOUSE:-}"
OFFLINE="${B200_OFFLINE:-0}"
VERSION="${FLASH_ATTENTION_VERSION:-2.8.3.post1}"
MAX_JOBS="${FLASH_ATTENTION_MAX_JOBS:-8}"

usage() {
  cat <<'EOF'
Usage: scripts/install_flash_attn_b200.sh [--python PATH] [--offline]

Environment:
  FAST_INFER_PYTHON          Python executable used for pip (default: python3)
  B200_WHEELHOUSE            wheelhouse containing the mirrored sdist
  B200_OFFLINE=1             use only B200_WHEELHOUSE
  FLASH_ATTENTION_VERSION    version to build (default: 2.8.3.post1)
  FLASH_ATTENTION_MAX_JOBS   Ninja jobs (default: 8)
EOF
}

die() {
  echo "install_flash_attn_b200: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || die "--python requires a path"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --offline)
      OFFLINE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -x "$PYTHON_BIN" || "$PYTHON_BIN" == */* ]] \
  || PYTHON_BIN="$(command -v "$PYTHON_BIN" 2>/dev/null || true)"
[[ -x "$PYTHON_BIN" ]] || die "Python executable not found: $PYTHON_BIN"

if [[ "$OFFLINE" == "1" ]]; then
  [[ -n "$WHEELHOUSE" ]] || die "B200_WHEELHOUSE is required in offline mode"
  [[ -d "$WHEELHOUSE" ]] || die "wheelhouse directory not found: $WHEELHOUSE"
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fast-infer-flash-attn.XXXXXX")"
cleanup() {
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

DOWNLOAD_ARGS=(
  download
  --no-deps
  --no-binary flash-attn
  --dest "$WORK_DIR"
  "flash-attn==$VERSION"
)
if [[ "$OFFLINE" == "1" ]]; then
  DOWNLOAD_ARGS+=(--no-index --find-links "$WHEELHOUSE")
fi

echo "Downloading mirrored flash-attn==$VERSION source"
PIP_DISABLE_PIP_VERSION_CHECK=1 \
  "$PYTHON_BIN" -m pip "${DOWNLOAD_ARGS[@]}"

shopt -s nullglob
ARCHIVES=("$WORK_DIR"/flash_attn-*.tar.gz "$WORK_DIR"/flash_attn-*.zip)
[[ "${#ARCHIVES[@]}" -eq 1 ]] \
  || die "expected one flash-attn sdist in $WORK_DIR, found ${#ARCHIVES[@]}"

ARCHIVE="${ARCHIVES[0]}"
if [[ "$ARCHIVE" == *.tar.gz ]]; then
  tar -xzf "$ARCHIVE" -C "$WORK_DIR"
else
  command -v unzip >/dev/null 2>&1 || die "unzip is required for $ARCHIVE"
  unzip -q "$ARCHIVE" -d "$WORK_DIR"
fi

SOURCE_DIRS=("$WORK_DIR"/flash-attn-* "$WORK_DIR"/flash_attn-*)
SOURCE_DIR=""
for candidate in "${SOURCE_DIRS[@]}"; do
  if [[ -d "$candidate" && -f "$candidate/setup.py" ]]; then
    SOURCE_DIR="$candidate"
    break
  fi
done
[[ -n "$SOURCE_DIR" ]] || die "could not locate extracted flash-attn setup.py"

SETUP_PY="$SOURCE_DIR/setup.py"
grep -q -- "-std=c++17" "$SETUP_PY" \
  || die "flash-attn setup.py has no expected C++17 flags; refuse unknown patch"
sed -i 's/-std=c++17/-std=c++20/g' "$SETUP_PY"
grep -q -- "-std=c++17" "$SETUP_PY" \
  && die "failed to patch flash-attn setup.py to C++20"

echo "Building flash-attn==$VERSION with C++20 and MAX_JOBS=$MAX_JOBS"
FLASH_ATTENTION_FORCE_BUILD=TRUE \
MAX_JOBS="$MAX_JOBS" \
PIP_DISABLE_PIP_VERSION_CHECK=1 \
  "$PYTHON_BIN" -m pip install \
    --no-build-isolation \
    --no-deps \
    "$SOURCE_DIR"

