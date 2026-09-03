#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi
# Keep this helper subject to the same master-config and Python 3.12 checks as
# every other run_*.sh launcher.  The smoke's inference/training arguments
# below intentionally force CPU and synthetic data after loading the config.
# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config syncspec || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

PYTHON="$FAST_INFER_PYTHON"
WORK_DIR="${SYNCSPEC_CPU_SMOKE_DIR:-$ROOT/outputs/syncspec_cpu_smoke}"
SAMPLES="${SYNCSPEC_CPU_SMOKE_SAMPLES:-2}"
NEW_TOKENS="${SYNCSPEC_CPU_SMOKE_NEW_TOKENS:-4}"
STEPS="${SYNCSPEC_CPU_SMOKE_STEPS:-1}"
TRAIN_BATCH_SIZE="${SYNCSPEC_CPU_SMOKE_TRAIN_BATCH_SIZE:-1}"
INFER_BATCH_SIZE="${SYNCSPEC_CPU_SMOKE_BATCH_SIZE:-2}"
PROFILE_SPECS="${SYNCSPEC_CPU_SMOKE_PROFILE_SPECS:-4:2,4:4}"
SEED="${SYNCSPEC_CPU_SMOKE_SEED:-42}"

if [[ "$SAMPLES" -le 0 || "$NEW_TOKENS" -le 0 || "$STEPS" -le 0 ]]; then
  echo "CPU smoke sample/token/step limits must be positive" >&2
  exit 2
fi
if [[ "$TRAIN_BATCH_SIZE" -le 0 || "$INFER_BATCH_SIZE" -le 0 ]]; then
  echo "CPU smoke batch sizes must be positive" >&2
  exit 2
fi

TRAJECTORY="$WORK_DIR/trajectories.pt"
CHECKPOINT="$WORK_DIR/checkpoint"
PROFILE="$WORK_DIR/profile.json"
INFER_OUTPUT="$WORK_DIR/infer.jsonl"
mkdir -p "$WORK_DIR"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
"$PYTHON" scripts/build_syncspec_trajectories.py \
  --backend synthetic --output "$TRAJECTORY" --samples "$SAMPLES" \
  --max-new-tokens "$NEW_TOKENS" --source-chunk-size 2 \
  --include-target-features --include-source-memory --seed "$SEED"

"$PYTHON" scripts/train_syncspec.py \
  --stage joint --data "$TRAJECTORY" --output-dir "$CHECKPOINT" \
  --device cpu --dtype float32 --vocab-size 256 --hidden-size 16 \
  --layers 1 --heads 2 --groups 2 --kd 4 --steps "$STEPS" \
  --train-batch-size "$TRAIN_BATCH_SIZE" --seed "$SEED" --no-amp

for artifact in "$CHECKPOINT/pytorch_model.bin" \
  "$CHECKPOINT/selector.pt" "$CHECKPOINT/survival.pt"; do
  if [[ ! -f "$artifact" ]]; then
    echo "CPU smoke missing trained artifact: $artifact" >&2
    exit 1
  fi
done

"$PYTHON" scripts/profile_syncspec.py \
  --backend synthetic --device cpu --context-length 3 \
  --batch-size "$INFER_BATCH_SIZE" --budget-profiles "$PROFILE_SPECS" \
  --repeats 1 --warmup-runs 0 --output "$PROFILE"

"$PYTHON" scripts/infer_syncspec.py \
  --backend synthetic --device cpu --max-samples "$SAMPLES" \
  --batch-size "$INFER_BATCH_SIZE" --max-new-tokens "$NEW_TOKENS" \
  --budget-profiles "$PROFILE_SPECS" --profile "$PROFILE" \
  --check-exactness --output "$INFER_OUTPUT"

echo "{\"status\":\"ok\",\"trajectory\":\"$TRAJECTORY\",\"checkpoint\":\"$CHECKPOINT\",\"profile\":\"$PROFILE\",\"inference\":\"$INFER_OUTPUT\"}"
