#!/usr/bin/env bash
# Train and validate the complete SyncSpec pipeline on the canonical server.
#
# The runtime configuration is read from the shared master shell-env through
# config/master.path (or FAST_INFER_MASTER_CONFIG/--config). This launcher is
# orchestration-only: model and training behavior remain in the Python CLIs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/train.sh [options]

Run the SyncSpec target-trajectory -> joint-training -> profile -> inference
pipeline using the shared master config.

Options:
  --config PATH             master shell-env; default: config/master.path
  --mode MODE               smoke or full; default: smoke
  --resume                  resume trajectory cache and existing drafter state
  --skip-preflight          do not run the strict B200 train preflight
  --skip-profile            reuse an existing profile at SYNCSPEC_TRAIN_PROFILE
  --skip-infer              stop after training/checkpoint validation
  --check-exactness         compare inference with vanilla target AR
  --no-check-exactness      do not run the vanilla target AR comparison
  -h, --help                show this help

The full mode uses SYNCSPEC_TRAIN_FULL_* from the master config, with built-in
long-run defaults. Smoke mode overrides sample count, token count, KD and
steps with safe small defaults. Use absolute paths in the master config for
model/data/checkpoints.
EOF
}

die() {
  echo "syncspec train: $*" >&2
  exit 2
}

MODE="${SYNCSPEC_TRAIN_MODE:-smoke}"
CONFIG_ARG=""
RESUME="${SYNCSPEC_TRAIN_RESUME:-0}"
SKIP_PREFLIGHT=0
SKIP_PROFILE=0
SKIP_INFER=0
CHECK_EXACTNESS="${SYNCSPEC_TRAIN_CHECK_EXACTNESS:-}"

while (($# > 0)); do
  case "$1" in
    --config|--master-config)
      (($# >= 2)) || die "$1 requires a path"
      CONFIG_ARG="$2"
      shift 2
      ;;
    --mode)
      (($# >= 2)) || die "--mode requires smoke or full"
      MODE="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --skip-preflight)
      SKIP_PREFLIGHT=1
      shift
      ;;
    --skip-profile)
      SKIP_PROFILE=1
      shift
      ;;
    --skip-infer)
      SKIP_INFER=1
      shift
      ;;
    --check-exactness)
      CHECK_EXACTNESS=1
      shift
      ;;
    --no-check-exactness)
      CHECK_EXACTNESS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      (($# == 0)) || die "unexpected positional arguments: $*"
      ;;
    -* )
      die "unknown option: $1 (use --help)"
      ;;
    *)
      if [[ -z "$CONFIG_ARG" ]]; then
        CONFIG_ARG="$1"
        shift
      else
        die "unexpected positional argument: $1"
      fi
      ;;
  esac
done

case "$MODE" in
  smoke|full) ;;
  *) die "--mode must be smoke or full, got: $MODE" ;;
esac

if [[ -n "$CONFIG_ARG" ]]; then
  export FAST_INFER_MASTER_CONFIG="$CONFIG_ARG"
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config syncspec || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

cd "$ROOT"

TARGET_MODEL="${TARGET_MODEL:-}"
DATA_FILE="${DATA_FILE:-}"
DEVICE="${DEVICE:-${FI_DEVICE:-cuda}}"
DTYPE="${DTYPE:-bfloat16}"
LOCAL_FILES_ONLY="${SYNCSPEC_LOCAL_FILES_ONLY:-1}"

[[ -n "$TARGET_MODEL" ]] || die "target model is empty; set SYNCSPEC_TARGET_MODEL or MODEL_TARGET in master config"
[[ -n "$DATA_FILE" ]] || die "training data is empty; set SYNCSPEC_DATA_FILE or DATA_INPUT in master config"

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  LOCAL_FILES_ARGS=(--local-files-only)
else
  LOCAL_FILES_ARGS=(--no-local-files-only)
fi

if [[ "$MODE" == "full" ]]; then
  TRAIN_OUTPUT_DIR="${SYNCSPEC_TRAIN_FULL_OUTPUT_DIR:-checkpoints/syncspec_train}"
  TRAIN_ARTIFACT_DIR="${SYNCSPEC_TRAIN_FULL_ARTIFACT_DIR:-outputs/syncspec_train}"
  TRAJECTORY="${SYNCSPEC_TRAIN_FULL_TRAJECTORY:-$TRAIN_ARTIFACT_DIR/trajectories.pt}"
  PREFLIGHT="${SYNCSPEC_TRAIN_FULL_PREFLIGHT_OUTPUT:-$TRAIN_ARTIFACT_DIR/preflight_train.json}"
  INFER_PREFLIGHT="${SYNCSPEC_TRAIN_FULL_INFER_PREFLIGHT_OUTPUT:-$TRAIN_ARTIFACT_DIR/preflight_infer.json}"
  OUTPUT_FILE="${SYNCSPEC_TRAIN_FULL_INFER_OUTPUT:-$TRAIN_ARTIFACT_DIR/infer.jsonl}"
  PROFILE="${SYNCSPEC_TRAIN_FULL_PROFILE:-$TRAIN_ARTIFACT_DIR/profile.json}"
  RUN_LOG="${SYNCSPEC_TRAIN_FULL_LOG:-$TRAIN_ARTIFACT_DIR/train.log}"
  MAX_SAMPLES="${SYNCSPEC_TRAIN_FULL_MAX_SAMPLES:-1000}"
  MAX_NEW_TOKENS="${SYNCSPEC_TRAIN_FULL_MAX_NEW_TOKENS:-64}"
  MAX_INPUT_TOKENS="${SYNCSPEC_TRAIN_FULL_MAX_INPUT_TOKENS:-0}"
  STEPS="${SYNCSPEC_TRAIN_FULL_STEPS:-1000}"
  TRAIN_BATCH_SIZE="${SYNCSPEC_TRAIN_FULL_BATCH_SIZE:-1}"
  BATCH_SIZE="${SYNCSPEC_TRAIN_FULL_PROFILE_BATCH_SIZE:-1}"
  KD="${SYNCSPEC_TRAIN_FULL_KD:-16}"
  NUM_ANCHORS="${SYNCSPEC_TRAIN_FULL_NUM_ANCHORS:-512}"
  PROFILE_KD="${SYNCSPEC_TRAIN_FULL_PROFILE_KD:-16}"
  PROFILE_KV="${SYNCSPEC_TRAIN_FULL_PROFILE_KV:-8}"
  PROFILE_SPECS="${SYNCSPEC_TRAIN_FULL_PROFILE_SPECS:-8:4,8:8,16:4,16:8,16:12,16:16}"
  JOINT_FINETUNE="${SYNCSPEC_TRAIN_FULL_JOINT_FINETUNE:-0}"
else
  TRAIN_OUTPUT_DIR="${SYNCSPEC_TRAIN_OUTPUT_DIR:-checkpoints/syncspec_train_smoke}"
  TRAIN_ARTIFACT_DIR="${SYNCSPEC_TRAIN_ARTIFACT_DIR:-outputs/syncspec_train_smoke}"
  TRAJECTORY="${SYNCSPEC_TRAIN_TRAJECTORY:-$TRAIN_ARTIFACT_DIR/trajectories.pt}"
  PREFLIGHT="${SYNCSPEC_TRAIN_PREFLIGHT_OUTPUT:-$TRAIN_ARTIFACT_DIR/preflight_train.json}"
  INFER_PREFLIGHT="${SYNCSPEC_INFER_PREFLIGHT_OUTPUT:-$TRAIN_ARTIFACT_DIR/preflight_infer.json}"
  OUTPUT_FILE="${SYNCSPEC_TRAIN_INFER_OUTPUT:-$TRAIN_ARTIFACT_DIR/infer.jsonl}"
  PROFILE="${SYNCSPEC_TRAIN_PROFILE:-$TRAIN_ARTIFACT_DIR/profile.json}"
  RUN_LOG="${SYNCSPEC_TRAIN_LOG:-$TRAIN_ARTIFACT_DIR/train.log}"
  MAX_SAMPLES="${SYNCSPEC_TRAIN_MAX_SAMPLES:-1}"
  MAX_NEW_TOKENS="${SYNCSPEC_TRAIN_MAX_NEW_TOKENS:-64}"
  MAX_INPUT_TOKENS="${SYNCSPEC_TRAIN_MAX_INPUT_TOKENS:-0}"
  STEPS="${SYNCSPEC_TRAIN_STEPS:-1000}"
  TRAIN_BATCH_SIZE="${SYNCSPEC_TRAIN_BATCH_SIZE:-1}"
  BATCH_SIZE="${SYNCSPEC_BATCH_SIZE:-1}"
  KD="${SYNCSPEC_TRAIN_KD:-16}"
  NUM_ANCHORS="${SYNCSPEC_TRAIN_NUM_ANCHORS:-512}"
  PROFILE_KD="${SYNCSPEC_TRAIN_PROFILE_KD:-16}"
  PROFILE_KV="${SYNCSPEC_TRAIN_PROFILE_KV:-8}"
  PROFILE_SPECS="${SYNCSPEC_TRAIN_PROFILE_SPECS:-${PROFILE_KD}:${PROFILE_KV}}"
  JOINT_FINETUNE="${SYNCSPEC_TRAIN_JOINT_FINETUNE:-0}"
fi

SEED="${SYNCSPEC_TRAIN_SEED:-42}"
SOURCE_CHUNK_SIZE="${SYNCSPEC_SOURCE_CHUNK_SIZE:-128}"
POSITION_DECAY="${SYNCSPEC_TRAIN_POSITION_DECAY:-7}"
ATTENTION_BACKEND="${SYNCSPEC_TRAIN_ATTENTION_BACKEND:-flash}"
AMP="${SYNCSPEC_TRAIN_AMP:-1}"
INCLUDE_LOGITS="${SYNCSPEC_TRAIN_INCLUDE_LOGITS:-0}"
JOINT_STEPS="${SYNCSPEC_TRAIN_JOINT_STEPS:-100}"
JOINT_LEARNING_RATE="${SYNCSPEC_TRAIN_JOINT_LEARNING_RATE:-}"
PROFILE_REPEATS="${SYNCSPEC_TRAIN_PROFILE_REPEATS:-3}"
PROFILE_WARMUP_RUNS="${SYNCSPEC_TRAIN_PROFILE_WARMUP_RUNS:-1}"

if [[ "$MODE" == "smoke" ]]; then
  MAX_SAMPLES="${SYNCSPEC_TRAIN_SMOKE_MAX_SAMPLES:-1}"
  MAX_NEW_TOKENS="${SYNCSPEC_TRAIN_SMOKE_MAX_NEW_TOKENS:-8}"
  MAX_INPUT_TOKENS="${SYNCSPEC_TRAIN_SMOKE_MAX_INPUT_TOKENS:-8192}"
  STEPS="${SYNCSPEC_TRAIN_SMOKE_STEPS:-1}"
  TRAIN_BATCH_SIZE="${SYNCSPEC_TRAIN_SMOKE_BATCH_SIZE:-1}"
  BATCH_SIZE="${SYNCSPEC_TRAIN_SMOKE_PROFILE_BATCH_SIZE:-1}"
  KD="${SYNCSPEC_TRAIN_SMOKE_KD:-4}"
  NUM_ANCHORS="${SYNCSPEC_TRAIN_SMOKE_NUM_ANCHORS:-32}"
  PROFILE_KD="${SYNCSPEC_TRAIN_SMOKE_PROFILE_KD:-4}"
  PROFILE_KV="${SYNCSPEC_TRAIN_SMOKE_PROFILE_KV:-2}"
  PROFILE_SPECS="${SYNCSPEC_TRAIN_SMOKE_PROFILE_SPECS:-${PROFILE_KD}:${PROFILE_KV}}"
  JOINT_FINETUNE="${SYNCSPEC_TRAIN_SMOKE_JOINT_FINETUNE:-0}"
  [[ -n "${SYNCSPEC_TRAIN_CHECK_EXACTNESS:-}" ]] || CHECK_EXACTNESS=1
else
  [[ -n "$CHECK_EXACTNESS" ]] || CHECK_EXACTNESS=0
fi

[[ "$MAX_SAMPLES" =~ ^[0-9]+$ ]] || die "max samples must be an integer: $MAX_SAMPLES"
[[ "$BATCH_SIZE" =~ ^[0-9]+$ && "$BATCH_SIZE" -gt 0 ]] || die "profile batch size must be positive: $BATCH_SIZE"
[[ "$TRAIN_BATCH_SIZE" =~ ^[0-9]+$ && "$TRAIN_BATCH_SIZE" -gt 0 ]] || die "train batch size must be positive: $TRAIN_BATCH_SIZE"
[[ "$MAX_SAMPLES" -ge "$BATCH_SIZE" ]] || die "max samples must be >= profile batch size"
[[ "$STEPS" =~ ^[0-9]+$ && "$STEPS" -gt 0 ]] || die "training steps must be positive: $STEPS"
[[ "$KD" =~ ^[0-9]+$ && "$KD" -gt 0 ]] || die "training KD must be positive: $KD"

mkdir -p "$TRAIN_OUTPUT_DIR" "$TRAIN_ARTIFACT_DIR"
touch "$RUN_LOG"

run_step() {
  local name="$1"
  shift
  echo
  echo "===== SyncSpec: $name =====" | tee -a "$RUN_LOG"
  set +e
  "$@" 2>&1 | tee -a "$RUN_LOG"
  local pipeline_status=("${PIPESTATUS[@]}")
  set -e
  if [[ "${pipeline_status[0]}" != "0" ]]; then
    echo "SyncSpec step failed ($name), exit=${pipeline_status[0]}; see $RUN_LOG" >&2
    exit "${pipeline_status[0]}"
  fi
  if [[ "${pipeline_status[1]}" != "0" ]]; then
    echo "SyncSpec log writer failed ($name), exit=${pipeline_status[1]}; see $RUN_LOG" >&2
    exit "${pipeline_status[1]}"
  fi
}

if [[ "$RESUME" != "1" && "${SYNCSPEC_TRAIN_ALLOW_OVERWRITE:-0}" != "1" \
      && -s "$TRAIN_OUTPUT_DIR/pytorch_model.bin" ]]; then
  die "checkpoint already exists at $TRAIN_OUTPUT_DIR; use --resume, set a new output dir, or set SYNCSPEC_TRAIN_ALLOW_OVERWRITE=1"
fi

if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  run_step preflight \
    "$FAST_INFER_PYTHON" "$ROOT/scripts/check_syncspec_b200.py" \
    --phase train --target-model "$TARGET_MODEL" --data-file "$DATA_FILE" \
    --precision "$DTYPE" --batch-size "$BATCH_SIZE" \
    --output "$PREFLIGHT" --strict
fi

TRAJECTORY_ARGS=(
  --backend transformers
  --target-model "$TARGET_MODEL"
  --input "$DATA_FILE"
  --output "$TRAJECTORY"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --samples "$MAX_SAMPLES"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --max-input-tokens "$MAX_INPUT_TOKENS"
  --source-chunk-size "$SOURCE_CHUNK_SIZE"
  --num-anchors "$NUM_ANCHORS"
  --seed "$SEED"
  --include-target-features
  --include-source-memory
  "${LOCAL_FILES_ARGS[@]}"
)
if [[ "$RESUME" == "1" ]]; then
  TRAJECTORY_ARGS+=(--resume)
fi
if [[ "$INCLUDE_LOGITS" == "1" ]]; then
  TRAJECTORY_ARGS+=(--include-logits)
fi
run_step build_trajectory "$FAST_INFER_PYTHON" "$ROOT/scripts/build_syncspec_trajectories.py" "${TRAJECTORY_ARGS[@]}"

TRAIN_ARGS=(
  --stage joint
  --target-model "$TARGET_MODEL"
  --data "$TRAJECTORY"
  --output-dir "$TRAIN_OUTPUT_DIR"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --kd "$KD"
  --steps "$STEPS"
  --train-batch-size "$TRAIN_BATCH_SIZE"
  --num-anchors "$NUM_ANCHORS"
  --position-decay "$POSITION_DECAY"
  --attention-backend "$ATTENTION_BACKEND"
  --seed "$SEED"
  --log-file "$TRAIN_OUTPUT_DIR/training_steps.jsonl"
  "${LOCAL_FILES_ARGS[@]}"
)
if [[ "$AMP" == "1" ]]; then
  TRAIN_ARGS+=(--amp)
else
  TRAIN_ARGS+=(--no-amp)
fi
if [[ "$RESUME" == "1" && -s "$TRAIN_OUTPUT_DIR/pytorch_model.bin" ]]; then
  TRAIN_ARGS+=(--init-checkpoint "$TRAIN_OUTPUT_DIR")
fi
for pair in \
  "learning-rate:${SYNCSPEC_TRAIN_LEARNING_RATE:-}" \
  "grad-accumulation-steps:${SYNCSPEC_TRAIN_GRAD_ACCUMULATION_STEPS:-}" \
  "grad-clip-norm:${SYNCSPEC_TRAIN_GRAD_CLIP_NORM:-}" \
  "kl-weight:${SYNCSPEC_TRAIN_KL_WEIGHT:-}" \
  "rank-weight:${SYNCSPEC_TRAIN_RANK_WEIGHT:-}" \
  "rank-margin:${SYNCSPEC_TRAIN_RANK_MARGIN:-}" \
  "rank-top-m:${SYNCSPEC_TRAIN_RANK_TOP_M:-}"; do
  key="${pair%%:*}"
  value="${pair#*:}"
  [[ -n "$value" ]] || continue
  TRAIN_ARGS+=("--$key" "$value")
done
if [[ "$JOINT_FINETUNE" == "1" ]]; then
  TRAIN_ARGS+=(--joint-finetune --joint-steps "$JOINT_STEPS")
  [[ -n "$JOINT_LEARNING_RATE" ]] && TRAIN_ARGS+=(--joint-learning-rate "$JOINT_LEARNING_RATE")
fi
run_step train "$FAST_INFER_PYTHON" "$ROOT/scripts/train_syncspec.py" "${TRAIN_ARGS[@]}"

for artifact in \
  "$TRAIN_OUTPUT_DIR/config.json" \
  "$TRAIN_OUTPUT_DIR/pytorch_model.bin" \
  "$TRAIN_OUTPUT_DIR/selector.pt" \
  "$TRAIN_OUTPUT_DIR/selector_config.json" \
  "$TRAIN_OUTPUT_DIR/survival.pt" \
  "$TRAIN_OUTPUT_DIR/training_summary.json" \
  "$TRAIN_OUTPUT_DIR/training_steps.jsonl"; do
  [[ -s "$artifact" ]] || die "required training artifact missing or empty: $artifact"
done

PROFILE_ARGS=()
if [[ "$PROFILE_SPECS" == "${PROFILE_KD}:${PROFILE_KV}" ]]; then
  PROFILE_ARGS+=(--kd "$PROFILE_KD" --kv "$PROFILE_KV")
else
  PROFILE_ARGS+=(--budget-profiles "$PROFILE_SPECS")
fi

if [[ "$SKIP_PROFILE" != "1" ]]; then
  run_step profile "$FAST_INFER_PYTHON" "$ROOT/scripts/profile_syncspec.py" \
    --backend transformers \
    --target-model "$TARGET_MODEL" \
    --drafter-checkpoint "$TRAIN_OUTPUT_DIR" \
    --selector-checkpoint "$TRAIN_OUTPUT_DIR" \
    --survival-checkpoint "$TRAIN_OUTPUT_DIR" \
    --input "$DATA_FILE" \
    --output "$PROFILE" \
    --device "$DEVICE" --dtype "$DTYPE" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    "${PROFILE_ARGS[@]}" \
    --repeats "$PROFILE_REPEATS" \
    --warmup-runs "$PROFILE_WARMUP_RUNS" \
    "${LOCAL_FILES_ARGS[@]}"
else
  if [[ "$SKIP_INFER" != "1" ]]; then
    [[ -s "$PROFILE" ]] || die "--skip-profile requires an existing profile: $PROFILE"
  fi
fi

if [[ "$SKIP_INFER" != "1" ]]; then
  run_step infer_preflight "$FAST_INFER_PYTHON" "$ROOT/scripts/check_syncspec_b200.py" \
    --phase infer \
    --target-model "$TARGET_MODEL" \
    --drafter-checkpoint "$TRAIN_OUTPUT_DIR" \
    --selector-checkpoint "$TRAIN_OUTPUT_DIR" \
    --survival-checkpoint "$TRAIN_OUTPUT_DIR" \
    --data-file "$DATA_FILE" --profile "$PROFILE" \
    --precision "$DTYPE" --batch-size "$BATCH_SIZE" \
    --output "$INFER_PREFLIGHT" --strict

  INFER_ARGS=(
    --backend transformers
    --target-model "$TARGET_MODEL"
    --drafter-checkpoint "$TRAIN_OUTPUT_DIR"
    --selector-checkpoint "$TRAIN_OUTPUT_DIR"
    --survival-checkpoint "$TRAIN_OUTPUT_DIR"
    --input "$DATA_FILE"
    --output "$OUTPUT_FILE"
    --device "$DEVICE" --dtype "$DTYPE"
    --max-samples "$MAX_SAMPLES"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --max-input-tokens "$MAX_INPUT_TOKENS"
    --batch-size "$BATCH_SIZE"
    "${PROFILE_ARGS[@]}"
    --profile "$PROFILE"
    "${LOCAL_FILES_ARGS[@]}"
  )
  if [[ "$MODE" == "smoke" ]]; then
    INFER_ARGS+=(--smoke)
  fi
  if [[ "$CHECK_EXACTNESS" == "1" ]]; then
    INFER_ARGS+=(--check-exactness)
  fi
  run_step inference "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_syncspec.py" "${INFER_ARGS[@]}"
fi

echo
echo "SyncSpec training completed"
echo "  mode:       $MODE"
echo "  checkpoint: $TRAIN_OUTPUT_DIR"
echo "  trajectory: $TRAJECTORY"
echo "  profile:    $PROFILE"
echo "  log:        $RUN_LOG"
if [[ "$SKIP_INFER" == "1" ]]; then
  echo "  inference:  skipped"
else
  echo "  inference:  $OUTPUT_FILE"
fi
