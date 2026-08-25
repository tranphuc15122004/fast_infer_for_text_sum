#!/usr/bin/env bash
# Fresh-machine bootstrap: install uv, sync every baseline env from committed
# lock files, and run a quick smoke test of each baseline that can run on the
# current hardware.
#
#   bash scripts/bootstrap.sh            # T4 / no flash-attn
#   EXTRA_FLASH=1 bash scripts/bootstrap.sh   # big-GPU servers
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 1) uv ---------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# 2) sync all envs -----------------------------------------------------
bash scripts/setup_envs.sh

# 3) smoke tests that are hardware-agnostic / T4-friendly --------------
echo
echo "==> Smoke: LLMLingua"
bash scripts/run.sh llmlingua || echo "[skip/fail] llmlingua (needs compressor model download?)"

echo
echo "==> Smoke: RocketKV (kernel)"
bash scripts/run.sh rocketkv || echo "[fail] rocketkv smoke"

echo
echo "Bootstrap done. Full per-baseline runs: bash scripts/run.sh <baseline>"
