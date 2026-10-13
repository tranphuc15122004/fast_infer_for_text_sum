#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARGS=("$@")

for ((INDEX = 0; INDEX < ${#ARGS[@]}; INDEX++)); do
  case "${ARGS[$INDEX]}" in
    --config)
      INDEX=$((INDEX + 1))
      if (( INDEX >= ${#ARGS[@]} )); then
        echo "--config requires a path" >&2
        exit 2
      fi
      export FAST_INFER_MASTER_CONFIG="${ARGS[$INDEX]}"
      ;;
    --config=*) export FAST_INFER_MASTER_CONFIG="${ARGS[$INDEX]#--config=}" ;;
  esac
done

SAVED_NAMES=()
SAVED_VALUES=()
for NAME in FAST_INFER_PYTHON FAST_INFER_VENV MODEL_TARGET TARGET_MODEL \
  MODEL_DFLASH_DRAFT DRAFT_MODEL DATA_INPUT DATA_FILE LONG_BENCH_DATA_DIR \
  LONG_BENCH_DATA_FILE CUDA_VISIBLE_DEVICES; do
  if [[ -v "$NAME" ]]; then
    SAVED_NAMES+=("$NAME")
    SAVED_VALUES+=("${!NAME}")
  fi
done

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config dflash || exit 1
for INDEX in "${!SAVED_NAMES[@]}"; do
  printf -v "${SAVED_NAMES[$INDEX]}" '%s' "${SAVED_VALUES[$INDEX]}"
  export "${SAVED_NAMES[$INDEX]}"
done

was_caller_set() {
  local wanted="$1" saved
  for saved in "${SAVED_NAMES[@]}"; do
    [[ "$saved" == "$wanted" ]] && return 0
  done
  return 1
}
if was_caller_set TARGET_MODEL && ! was_caller_set MODEL_TARGET; then
  export MODEL_TARGET="$TARGET_MODEL"
fi
if was_caller_set DRAFT_MODEL && ! was_caller_set MODEL_DFLASH_DRAFT; then
  export MODEL_DFLASH_DRAFT="$DRAFT_MODEL"
fi
if was_caller_set DATA_FILE && ! was_caller_set DATA_INPUT; then
  export DATA_INPUT="$DATA_FILE"
fi

for ((INDEX = 0; INDEX < ${#ARGS[@]}; INDEX++)); do
  case "${ARGS[$INDEX]}" in
    --gpu-ids)
      INDEX=$((INDEX + 1))
      if (( INDEX >= ${#ARGS[@]} )); then
        echo "--gpu-ids requires comma-separated ids" >&2
        exit 2
      fi
      export CUDA_VISIBLE_DEVICES="${ARGS[$INDEX]}"
      ;;
    --gpu-ids=*) export CUDA_VISIBLE_DEVICES="${ARGS[$INDEX]#--gpu-ids=}" ;;
  esac
done

# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1
cd "$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/externals/dflash${PYTHONPATH:+:$PYTHONPATH}"
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/run_dflash_speculation_B.py" "${ARGS[@]}"
