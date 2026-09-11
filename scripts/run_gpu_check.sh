#!/usr/bin/env bash
# Kiểm tra GPU id + VRAM còn trống trước khi chạy benchmark LongBench (B200).
#
# Wrapper này load master env với profile `longbench` — giống hệt
# scripts/run_longbench_200.sh — để lựa chọn GPU mặc định của job là chính xác
# (LONG_BENCH_GPU_IDS/FI_GPU_IDS/CUDA_VISIBLE_DEVICES), rồi gọi
# scripts/check_gpu_vram.py bằng shared interpreter 3.12.
#
# Usage:
#   bash scripts/run_gpu_check.sh                         # master qua pointer
#   bash scripts/run_gpu_check.sh --config /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env
#   bash scripts/run_gpu_check.sh --config /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env --gpu-ids 2 --min-free-gb 120
#   bash scripts/run_gpu_check.sh --config /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env --json gpu_report.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Optional positional/config flag override the repository's master pointer.
if [[ "${1:-}" == "--config" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "--config requires a master env path" >&2
    exit 2
  fi
  export FAST_INFER_MASTER_CONFIG="$2"
  shift 2
elif [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh" || exit 1
fast_infer_load_config longbench || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

cd "$ROOT"
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/check_gpu_vram.py" "$@"
