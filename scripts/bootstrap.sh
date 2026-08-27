#!/usr/bin/env bash
# Bootstrap the single shared Python 3.12 environment and run quick smoke tests.
# The server is offline: uv and Python 3.12 must already be available, together
# with the local cache/wheelhouse required by requirements.txt.
#
#   bash scripts/bootstrap.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 1) uv ---------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required and must be preinstalled on the offline server" >&2
  exit 1
fi
echo "uv: $(uv --version)"

# 2) create/install the shared environment -----------------------------
bash scripts/setup_venv.sh --offline

# 3) smoke tests that are hardware-agnostic / T4-friendly ---------------
echo
echo "==> Smoke: LLMLingua"
bash scripts/run.sh llmlingua || echo "[skip/fail] llmlingua (needs compressor model download?)"

echo
echo "==> Smoke: RocketKV (kernel)"
bash scripts/run.sh rocketkv || echo "[fail] rocketkv smoke"

echo
echo "Bootstrap done. Full per-baseline runs: bash scripts/run.sh <baseline>"
