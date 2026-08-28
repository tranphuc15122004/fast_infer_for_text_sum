#!/usr/bin/env bash
# Conda-based installation for non-docker environments

set -euo pipefail

# ----- Conda bootstrap (non-interactive) -----
# Prefer sourcing the canonical profile script over eval; works in non-login shells too.
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME="sglang_sssd"

# Channel hygiene (optional but recommended for determinism)
conda config --set channel_priority strict
# If you actually need Anaconda main, keep it; otherwise conda-forge-only is simpler.
# conda config --add channels conda-forge
# conda config --add channels nvidia

# Accept ToS only if the subcommand exists (older/newer conda may not have it)
if conda help | grep -q "^ *tos "; then
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
fi

# Create env if missing
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating env '$ENV_NAME'..."
  conda create -n "$ENV_NAME" python=3.11 -y
else
  echo "Environment '$ENV_NAME' already exists."
fi

# Activate
conda activate "$ENV_NAME"

# Keep pip/build tooling fresh for editable installs and PEP 517 builds
python -m pip install --upgrade pip wheel build

# CUDA (ensure your host NVIDIA driver supports it: `nvidia-smi` should show >= 12.9)
conda install -y -c nvidia -c conda-forge cuda-toolkit=12.9

# Compilers & build tools
conda install -y -c conda-forge cxx-compiler ninja cmake

# (Linux-only sanity check if you rely on CUDA; skip on macOS)
case "$(uname -s)" in
  Linux) : ;;
  *) echo "Warning: CUDA setup is Linux/NVIDIA-only; skipping GPU-specific steps." ;;
esac

# Install SGLang (use python within the env; avoid bare `python3`)
python -m pip install -e "python"

# Clean, then reinstall SSSD from source
if [[ -d sssd_speculator ]]; then
  pushd sssd_speculator >/dev/null
  python -m pip uninstall -y sssd_speculator || true
  rm -rf build/ dist/ ./**/*.egg-info 2>/dev/null || true
  python -m pip install -e . --config-settings editable_mode=compat
  popd >/dev/null
else
  echo "Directory 'sssd_speculator' not found; skipping SSSD install."
fi

# Optional: compile parallelism for CMake-based builds
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
