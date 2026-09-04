#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi
# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config syncspec || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

cd "$ROOT"
TARGET_MODEL="${TARGET_MODEL:-}"
DATA_FILE="${DATA_FILE:-}"
TRAIN_OUTPUT_DIR="${SYNCSPEC_TRAIN_OUTPUT_DIR:-checkpoints/syncspec_b200_train_smoke}"
TRAIN_ARTIFACT_DIR="${SYNCSPEC_TRAIN_ARTIFACT_DIR:-outputs/syncspec_b200_train_smoke}"
# Binary torch cache is the safe default for long-context target features;
# callers may override this with a `.jsonl` path for human inspection.
TRAJECTORY="${SYNCSPEC_TRAIN_TRAJECTORY:-$TRAIN_ARTIFACT_DIR/trajectories.pt}"
PREFLIGHT="${SYNCSPEC_TRAIN_PREFLIGHT_OUTPUT:-$TRAIN_ARTIFACT_DIR/preflight.json}"
INFER_PREFLIGHT="${SYNCSPEC_INFER_PREFLIGHT_OUTPUT:-$TRAIN_ARTIFACT_DIR/infer_preflight.json}"
OUTPUT_FILE="${SYNCSPEC_TRAIN_INFER_OUTPUT:-$TRAIN_ARTIFACT_DIR/infer.jsonl}"
PROFILE="${SYNCSPEC_TRAIN_PROFILE:-$TRAIN_ARTIFACT_DIR/profile.json}"
MAX_SAMPLES="${SYNCSPEC_TRAIN_MAX_SAMPLES:-1}"
MAX_NEW_TOKENS="${SYNCSPEC_TRAIN_MAX_NEW_TOKENS:-8}"
MAX_INPUT_TOKENS="${SYNCSPEC_TRAIN_MAX_INPUT_TOKENS:-${MAX_INPUT_TOKENS:-0}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SYNCSPEC_TRAIN_SEED:-42}"
STEPS="${SYNCSPEC_TRAIN_STEPS:-1}"
TRAIN_BATCH_SIZE="${SYNCSPEC_TRAIN_BATCH_SIZE:-1}"
KD="${SYNCSPEC_TRAIN_KD:-4}"
PROFILE_KD="${SYNCSPEC_TRAIN_PROFILE_KD:-${SYNCSPEC_KD:-16}}"
PROFILE_KV="${SYNCSPEC_TRAIN_PROFILE_KV:-${SYNCSPEC_KV:-8}}"
PROFILE_SPECS="${SYNCSPEC_TRAIN_PROFILE_SPECS:-${PROFILE_KD}:${PROFILE_KV}}"

if (( MAX_SAMPLES < BATCH_SIZE )); then
  echo "max_samples must be >= batch size (MAX_SAMPLES=$MAX_SAMPLES BATCH_SIZE=$BATCH_SIZE)" >&2
  exit 2
fi

if [[ "$PROFILE_SPECS" == "${PROFILE_KD}:${PROFILE_KV}" ]]; then
  PROFILE_ARGS=(--kd "$PROFILE_KD" --kv "$PROFILE_KV")
else
  PROFILE_ARGS=(--budget-profiles "$PROFILE_SPECS")
fi

TRAJECTORY_ARGS=(
  --backend transformers --target-model "$TARGET_MODEL" --input "$DATA_FILE"
  --output "$TRAJECTORY" --device "${DEVICE:-cuda}" --dtype "${DTYPE:-bfloat16}"
  --samples "$MAX_SAMPLES" --max-new-tokens "$MAX_NEW_TOKENS"
  --max-input-tokens "$MAX_INPUT_TOKENS"
  --seed "$SEED"
  --source-chunk-size "${SYNCSPEC_SOURCE_CHUNK_SIZE:-128}"
  --include-target-features --include-source-memory --resume --local-files-only
)
if [[ "${SYNCSPEC_TRAIN_INCLUDE_LOGITS:-0}" == "1" ]]; then
  TRAJECTORY_ARGS+=(--include-logits)
fi

"$FAST_INFER_PYTHON" scripts/check_syncspec_b200.py \
  --phase train --target-model "$TARGET_MODEL" --data-file "$DATA_FILE" \
  --precision "${DTYPE:-bfloat16}" --batch-size "$BATCH_SIZE" \
  --output "$PREFLIGHT" --strict

"$FAST_INFER_PYTHON" scripts/build_syncspec_trajectories.py "${TRAJECTORY_ARGS[@]}"

TRAIN_ARGS=(
  --stage joint --target-model "$TARGET_MODEL" --data "$TRAJECTORY" \
  --output-dir "$TRAIN_OUTPUT_DIR" --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-bfloat16}" --kd "$KD" --steps "$STEPS" \
  --train-batch-size "$TRAIN_BATCH_SIZE" --seed "$SEED" --local-files-only
)
if [[ -n "${SYNCSPEC_TRAIN_LEARNING_RATE:-}" ]]; then
  TRAIN_ARGS+=(--learning-rate "$SYNCSPEC_TRAIN_LEARNING_RATE")
fi
if [[ -n "${SYNCSPEC_TRAIN_GRAD_ACCUMULATION_STEPS:-}" ]]; then
  TRAIN_ARGS+=(--grad-accumulation-steps "$SYNCSPEC_TRAIN_GRAD_ACCUMULATION_STEPS")
fi
if [[ -n "${SYNCSPEC_TRAIN_GRAD_CLIP_NORM:-}" ]]; then
  TRAIN_ARGS+=(--grad-clip-norm "$SYNCSPEC_TRAIN_GRAD_CLIP_NORM")
fi
if [[ "${SYNCSPEC_TRAIN_AMP:-1}" == "0" ]]; then
  TRAIN_ARGS+=(--no-amp)
fi
if [[ -n "${SYNCSPEC_TRAIN_POSITION_DECAY:-}" ]]; then
  TRAIN_ARGS+=(--position-decay "$SYNCSPEC_TRAIN_POSITION_DECAY")
fi
if [[ -n "${SYNCSPEC_TRAIN_NUM_ANCHORS:-}" ]]; then
  TRAIN_ARGS+=(--num-anchors "$SYNCSPEC_TRAIN_NUM_ANCHORS")
fi
if [[ -n "${SYNCSPEC_TRAIN_ATTENTION_BACKEND:-}" ]]; then
  TRAIN_ARGS+=(--attention-backend "$SYNCSPEC_TRAIN_ATTENTION_BACKEND")
fi
if [[ -n "${SYNCSPEC_TRAIN_KL_WEIGHT:-}" ]]; then
  TRAIN_ARGS+=(--kl-weight "$SYNCSPEC_TRAIN_KL_WEIGHT")
fi
if [[ -n "${SYNCSPEC_TRAIN_RANK_WEIGHT:-}" ]]; then
  TRAIN_ARGS+=(--rank-weight "$SYNCSPEC_TRAIN_RANK_WEIGHT")
fi
if [[ -n "${SYNCSPEC_TRAIN_RANK_MARGIN:-}" ]]; then
  TRAIN_ARGS+=(--rank-margin "$SYNCSPEC_TRAIN_RANK_MARGIN")
fi
if [[ -n "${SYNCSPEC_TRAIN_RANK_TOP_M:-}" ]]; then
  TRAIN_ARGS+=(--rank-top-m "$SYNCSPEC_TRAIN_RANK_TOP_M")
fi
if [[ "${SYNCSPEC_TRAIN_JOINT_FINETUNE:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--joint-finetune)
  if [[ -n "${SYNCSPEC_TRAIN_JOINT_STEPS:-}" ]]; then
    TRAIN_ARGS+=(--joint-steps "$SYNCSPEC_TRAIN_JOINT_STEPS")
  fi
  if [[ -n "${SYNCSPEC_TRAIN_JOINT_LEARNING_RATE:-}" ]]; then
    TRAIN_ARGS+=(--joint-learning-rate "$SYNCSPEC_TRAIN_JOINT_LEARNING_RATE")
  fi
fi
"$FAST_INFER_PYTHON" scripts/train_syncspec.py "${TRAIN_ARGS[@]}"

"$FAST_INFER_PYTHON" scripts/profile_syncspec.py \
  --backend transformers --target-model "$TARGET_MODEL" \
  --drafter-checkpoint "$TRAIN_OUTPUT_DIR" \
  --selector-checkpoint "$TRAIN_OUTPUT_DIR" --survival-checkpoint "$TRAIN_OUTPUT_DIR" \
  --input "$DATA_FILE" \
  --output "$PROFILE" --device "${DEVICE:-cuda}" --dtype "${DTYPE:-bfloat16}" \
  --max-input-tokens "$MAX_INPUT_TOKENS" \
  --batch-size "${BATCH_SIZE:-1}" "${PROFILE_ARGS[@]}" \
  --repeats "${SYNCSPEC_TRAIN_PROFILE_REPEATS:-3}" \
  --warmup-runs "${SYNCSPEC_TRAIN_PROFILE_WARMUP_RUNS:-1}" \
  --local-files-only

"$FAST_INFER_PYTHON" scripts/check_syncspec_b200.py \
  --phase infer --target-model "$TARGET_MODEL" --drafter-checkpoint "$TRAIN_OUTPUT_DIR" \
  --selector-checkpoint "$TRAIN_OUTPUT_DIR" --survival-checkpoint "$TRAIN_OUTPUT_DIR" \
  --data-file "$DATA_FILE" --profile "$PROFILE" \
  --precision "${DTYPE:-bfloat16}" --batch-size "$BATCH_SIZE" \
  --output "$INFER_PREFLIGHT" --strict

"$FAST_INFER_PYTHON" scripts/infer_syncspec.py \
  --backend transformers --target-model "$TARGET_MODEL" \
  --drafter-checkpoint "$TRAIN_OUTPUT_DIR" \
  --selector-checkpoint "$TRAIN_OUTPUT_DIR" \
  --survival-checkpoint "$TRAIN_OUTPUT_DIR" \
  --input "$DATA_FILE" --output "$OUTPUT_FILE" \
  --device "${DEVICE:-cuda}" --dtype "${DTYPE:-bfloat16}" \
  --max-samples "$MAX_SAMPLES" --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-input-tokens "$MAX_INPUT_TOKENS" \
  --batch-size "${BATCH_SIZE:-1}" \
  "${PROFILE_ARGS[@]}" \
  --profile "$PROFILE" \
  --local-files-only --smoke --check-exactness "$@"
