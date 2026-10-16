#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == "--config" ]]; then
  export FAST_INFER_MASTER_CONFIG="${2:?--config requires a path}"
  shift 2
elif [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config vllm_research || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

OUTPUT_ROOT="${VLLM_RESEARCH_OUTPUT_ROOT:-$ROOT/outputs/vllm_research}"
RUN_ID="${VLLM_RESEARCH_RUN_ID:-}"
RESUMING=0
ARGS=("$@")
for ((index = 0; index < ${#ARGS[@]}; index++)); do
  argument="${ARGS[$index]}"
  case "$argument" in
    --run-id)
      if (( index + 1 < ${#ARGS[@]} )); then RUN_ID="${ARGS[$((index + 1))]}"; fi
      ;;
    --run-id=*) RUN_ID="${argument#--run-id=}" ;;
    --resume)
      RESUMING=1
      if (( index + 1 < ${#ARGS[@]} )) && [[ "${ARGS[$((index + 1))]}" != -* ]]; then
        RUN_ID="${ARGS[$((index + 1))]}"
      fi
      ;;
    --resume=*) RESUMING=1; RUN_ID="${argument#--resume=}" ;;
    --output-root)
      if (( index + 1 < ${#ARGS[@]} )); then OUTPUT_ROOT="${ARGS[$((index + 1))]}"; fi
      ;;
    --output-root=*) OUTPUT_ROOT="${argument#--output-root=}" ;;
  esac
done
if [[ "$OUTPUT_ROOT" != /* ]]; then OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT"; fi
if [[ -z "$RUN_ID" ]]; then RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$BASHPID"; fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "VLLM_RESEARCH_RUN_ID may contain only letters, digits, '.', '_' and '-'" >&2
  exit 2
fi
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
if [[ -e "$RUN_DIR" && "$RESUMING" != "1" ]]; then
  echo "Campaign already exists; use --resume $RUN_ID: $RUN_DIR" >&2
  exit 2
fi
if [[ "$RESUMING" == "1" && ! -d "$RUN_DIR" ]]; then
  echo "Campaign to resume was not found: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
export VLLM_RESEARCH_OUTPUT_ROOT="$OUTPUT_ROOT"
export VLLM_RESEARCH_RUN_ID="$RUN_ID"
exec > >(tee -a "$RUN_DIR/console.log") 2>&1

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

exec "$FAST_INFER_PYTHON" -u "$ROOT/scripts/infer_vllm_research.py" "${ARGS[@]}"
