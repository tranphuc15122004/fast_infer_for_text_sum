#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

source "$ROOT/scripts/common/config.sh"
fast_infer_load_config fa4_native
source "$ROOT/scripts/common/runtime.sh"

export MODEL_TARGET="${LONG_BENCH_MODEL:-${MODEL_TARGET:-}}"
export MODEL_EAGLE_DRAFT="${LONG_BENCH_EAGLE_MODEL:-${MODEL_EAGLE_DRAFT:-}}"
export MODEL_DFLASH_DRAFT="${LONG_BENCH_DFLASH_MODEL:-${MODEL_DFLASH_DRAFT:-}}"
export MODEL_DOMINO_DRAFT="${LONG_BENCH_DOMINO_MODEL:-${MODEL_DOMINO_DRAFT:-${MODEL_DOMINO:-}}}"
export MODEL_DSPARK_DRAFT="${LONG_BENCH_DSPARK_MODEL:-${MODEL_DSPARK_DRAFT:-${MODEL_DSPARK:-}}}"

DATA_DIR="${FA4_DATA_DIR:-${LONG_BENCH_DATA_DIR:-data/longbench_100_14k}}"
if [[ "$DATA_DIR" != /* ]]; then
  if [[ -d "$ROOT/$DATA_DIR" ]]; then
    DATA_DIR="$ROOT/$DATA_DIR"
  elif [[ "$DATA_DIR" == *"eval_100"* ]]; then
    if [[ -d "/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/eval_100" ]]; then
      DATA_DIR="/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/eval_100"
    elif [[ -d "$ROOT/data/eval_100" ]]; then
      DATA_DIR="$ROOT/data/eval_100"
    elif [[ -d "$ROOT/datasets/eval_100" ]]; then
      DATA_DIR="$ROOT/datasets/eval_100"
    elif [[ -d "/home/tuantb/fast_infer_text_sum_Viet/datasets/eval_100" ]]; then
      DATA_DIR="/home/tuantb/fast_infer_text_sum_Viet/datasets/eval_100"
    fi
  else
    DATA_DIR="$ROOT/$DATA_DIR"
  fi
elif [[ ! -d "$DATA_DIR" && "$DATA_DIR" == *"eval_100"* ]]; then
  if [[ -d "/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/eval_100" ]]; then
    DATA_DIR="/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/eval_100"
  elif [[ -d "$ROOT/data/eval_100" ]]; then
    DATA_DIR="$ROOT/data/eval_100"
  elif [[ -d "/home/tuantb/fast_infer_text_sum_Viet/datasets/eval_100" ]]; then
    DATA_DIR="/home/tuantb/fast_infer_text_sum_Viet/datasets/eval_100"
  fi
fi

OUTPUT_DIR="${FA4_OUTPUT_ROOT:-$ROOT/outputs/fa4_native_benchmark}"
[[ "$OUTPUT_DIR" = /* ]] || OUTPUT_DIR="$ROOT/$OUTPUT_DIR"
export FA4_EXECUTION_BACKEND=server
export FA4_DATA_DIR="$DATA_DIR"
export FA4_OUTPUT_ROOT="$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${FI_GPU_IDS:-${LONG_BENCH_GPU_IDS:-0}}}"
export PYTHONPATH="$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

if [[ "${SMOKE:-0}" == "1" ]]; then export LONG_BENCH_MODE=smoke; fi
if [[ "${FULL:-0}" == "1" ]]; then export LONG_BENCH_MODE=full; fi
cd "$ROOT"
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/run_fa4_server.py" "$@"
