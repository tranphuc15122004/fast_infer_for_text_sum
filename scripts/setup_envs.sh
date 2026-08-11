#!/usr/bin/env bash
# Sync (install) every baseline env from committed uv.lock files.
#
# Usage:
#   scripts/setup_envs.sh                 # T4: no flash-attn (source build needed)
#   EXTRA_FLASH=1 scripts/setup_envs.sh   # big-GPU servers: install flash-attn
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTRA_FLASH="${EXTRA_FLASH:-0}"

cd "$ROOT"

UV_ARGS=(sync --locked)
if [[ "$EXTRA_FLASH" == "1" ]]; then
  UV_ARGS+=(--extra flash)
fi

for env_dir in envs/*/; do
  [[ -f "$env_dir/pyproject.toml" ]] || continue
  name="$(basename "$env_dir")"
  echo "==> [envs/$name] uv ${UV_ARGS[*]}"
  uv sync --project "$ROOT/$env_dir" "${UV_ARGS[@]}"
done

echo "==> [core (root project)] uv ${UV_ARGS[*]}"
uv sync --project "$ROOT" "${UV_ARGS[@]}"

echo "Done. Envs ready. Run: bash scripts/run.sh <baseline>"
