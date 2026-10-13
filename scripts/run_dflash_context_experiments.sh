#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARGS=("$@")

# Resolve --config before loading the shared model/data environment.
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
for NAME in \
  FAST_INFER_PYTHON FAST_INFER_VENV FAST_INFER_SYSTEM_PYTHON FAST_INFER_CACHE_ROOT \
  MODEL_TARGET TARGET_MODEL MODEL_DFLASH_DRAFT DRAFT_MODEL DATA_INPUT DATA_FILE \
  LONG_BENCH_DATA_DIR LONG_BENCH_DATA_FILE CUDA_VISIBLE_DEVICES FI_GPU_IDS \
  DCE_OUTPUT_ROOT DCE_RUN_ID; do
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

prefer_canonical_override() {
  local canonical="$1" alias="$2"
  if was_caller_set "$canonical" && ! was_caller_set "$alias" && [[ -n "${!canonical}" ]]; then
    printf -v "$alias" '%s' "${!canonical}"
    export "$alias"
  fi
}
prefer_canonical_override MODEL_TARGET TARGET_MODEL
prefer_canonical_override MODEL_DFLASH_DRAFT DRAFT_MODEL
prefer_canonical_override DATA_INPUT DATA_FILE

# Honor --gpu-ids before Python imports torch. The Python entrypoint validates
# the visible GPU and records the physical GPU details in preflight artifacts.
for ((INDEX = 0; INDEX < ${#ARGS[@]}; INDEX++)); do
  case "${ARGS[$INDEX]}" in
    --gpu-ids)
      INDEX=$((INDEX + 1))
      if (( INDEX >= ${#ARGS[@]} )); then
        echo "--gpu-ids requires a comma-separated list" >&2
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/externals/dflash${PYTHONPATH:+:$PYTHONPATH}"

exec "$FAST_INFER_PYTHON" "$ROOT/scripts/run_dflash_context_experiments.py" "${ARGS[@]}"
