#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# Keep explicit Phase A CLI environment overrides authoritative when the
# external master config also contains DFLASH_PHASE_A_* assignments.
declare -A DFLASH_PHASE_A_CALLER_OVERRIDES=()
for phase_a_var in \
  DFLASH_PHASE_A_BUDGET_MODE DFLASH_PHASE_A_BUDGET \
  DFLASH_PHASE_A_MIN_BUDGET DFLASH_PHASE_A_MAX_BUDGET \
  DFLASH_PHASE_A_CHUNK_SIZE DFLASH_PHASE_A_WINDOW_ROUNDS \
  DFLASH_PHASE_A_MIN_GAIN DFLASH_PHASE_A_DECAY DFLASH_PHASE_A_RECHECK_ROUNDS \
  DFLASH_PHASE_A_LAYER DFLASH_PHASE_A_SELECTOR DFLASH_PHASE_A_SINK \
  DFLASH_PHASE_A_RECENT DFLASH_PHASE_A_POOLING; do
  if [[ -v "$phase_a_var" ]]; then
    DFLASH_PHASE_A_CALLER_OVERRIDES["$phase_a_var"]="${!phase_a_var}"
  fi
done

source "$ROOT/scripts/common/config.sh"
fast_infer_load_config fa4_native
for phase_a_var in "${!DFLASH_PHASE_A_CALLER_OVERRIDES[@]}"; do
  printf -v "$phase_a_var" '%s' "${DFLASH_PHASE_A_CALLER_OVERRIDES[$phase_a_var]}"
  export "$phase_a_var"
done
source "$ROOT/scripts/common/runtime.sh" || exit 1

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
