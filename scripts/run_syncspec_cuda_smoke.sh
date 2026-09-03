#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# Probe before loading the shared master.  This keeps the local/T4 behavior
# honest and machine-readable even when the production-only master path is not
# mounted on the development host.
if [[ -n "${FAST_INFER_PYTHON:-}" ]]; then
  PROBE_PYTHON="$FAST_INFER_PYTHON"
elif [[ -n "${FAST_INFER_VENV:-}" ]]; then
  PROBE_PYTHON="$FAST_INFER_VENV/bin/python"
else
  PROBE_PYTHON="${FI_PYTHON:-python3}"
fi
CUDA_OK="$($PROBE_PYTHON -c 'import torch; print(int(torch.cuda.is_available()))' 2>/dev/null)" || CUDA_OK=0
if [[ "$CUDA_OK" != "1" ]]; then
  "$PROBE_PYTHON" -c 'import json; print(json.dumps({"method": "syncspec_cuda_smoke", "status": "BLOCKED", "reason": "cuda_unavailable"}))'
  exit 2
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config syncspec || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# This command is intentionally self-contained: it validates the CUDA engine
# and real microbatch path without requiring a target model/checkpoint. It is
# complementary to run_syncspec_b200_smoke.sh, which validates real assets.

exec "$FAST_INFER_PYTHON" scripts/infer_syncspec.py \
  --backend synthetic --device cuda \
  --output "${SYNCSPEC_CUDA_SMOKE_OUTPUT:-outputs/syncspec_cuda_smoke.jsonl}" \
  --max-samples "${SYNCSPEC_CUDA_SMOKE_SAMPLES:-2}" \
  --batch-size "${SYNCSPEC_CUDA_SMOKE_BATCH_SIZE:-2}" \
  --max-new-tokens "${SYNCSPEC_CUDA_SMOKE_NEW_TOKENS:-4}" \
  "$@"
