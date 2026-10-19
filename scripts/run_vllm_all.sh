#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--config" ]]; then
  export FAST_INFER_MASTER_CONFIG="${2:?--config requires a path}"
  shift 2
elif [[ $# -gt 0 && "$1" != -* ]]; then
  export FAST_INFER_MASTER_CONFIG="$1"
  shift
fi

FULL_REQUESTED=0
for arg in "$@"; do
  if [[ "$arg" == "--full" ]]; then
    FULL_REQUESTED=1
    break
  fi
done

NEW_ARGS=()
CLI_OUTPUT_DIR=""
CLI_RUN_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      CLI_OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-dir=*)
      CLI_OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --run-id)
      CLI_RUN_ID="$2"
      shift 2
      ;;
    --run-id=*)
      CLI_RUN_ID="${1#*=}"
      shift
      ;;
    *)
      NEW_ARGS+=("$1")
      shift
      ;;
  esac
done
if [[ ${#NEW_ARGS[@]} -gt 0 ]]; then
  set -- "${NEW_ARGS[@]}"
else
  set --
fi

OUTPUT_DIR="${CLI_OUTPUT_DIR:-${VLLM_OUTPUT_DIR:-$ROOT/outputs/vllm_unified}}"
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$ROOT/$OUTPUT_DIR"
fi
RUN_ID="${CLI_RUN_ID:-${VLLM_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$BASHPID}}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "VLLM_RUN_ID may contain only letters, digits, '.', '_' and '-'" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
RUN_DIR="$OUTPUT_DIR/$RUN_ID"
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir "$RUN_DIR"
RUN_LOG="$RUN_DIR/console.log"
: > "$RUN_LOG"
export VLLM_OUTPUT_DIR="$OUTPUT_DIR"
export VLLM_RUN_ID="$RUN_ID"
exec > >(tee -a "$RUN_LOG") 2>&1

# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config vllm_all || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

: "${VLLM_TARGET_MODEL:?VLLM_TARGET_MODEL or MODEL_TARGET is required}"
: "${VLLM_DATA_FILE:?VLLM_DATA_FILE, LONG_BENCH_DATA_FILE or DATA_INPUT is required}"
VLLM_EAGLE3_MODEL="${VLLM_EAGLE3_MODEL:-}"
VLLM_DFLASH_MODEL="${VLLM_DFLASH_MODEL:-}"
VLLM_DOMINO_MODEL="${VLLM_DOMINO_MODEL:-}"
VLLM_DSPARK_MODEL="${VLLM_DSPARK_MODEL:-}"

MAX_NEW_TOKENS="${VLLM_MAX_NEW_TOKENS:-${RUN_MAX_NEW_TOKENS:-512}}"
MAX_INPUT_TOKENS="${VLLM_MAX_INPUT_TOKENS:-${RUN_MAX_INPUT_TOKENS:-8192}}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
METHODS="${VLLM_METHODS:-vanilla_vllm,eagle3,dflash,domino,dspark,dflash_phase_a}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ARGS=(
  --model "$VLLM_TARGET_MODEL"
  --data-file "$VLLM_DATA_FILE"
  --output-dir "$OUTPUT_DIR"
  --run-id "$RUN_ID"
  --eagle3-model "$VLLM_EAGLE3_MODEL"
  --dflash-model "$VLLM_DFLASH_MODEL"
  --domino-model "$VLLM_DOMINO_MODEL"
  --dspark-model "$VLLM_DSPARK_MODEL"
  --methods "$METHODS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --max-input-tokens "$MAX_INPUT_TOKENS"
  --max-model-len "$MAX_MODEL_LEN"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --measurement "${VLLM_MEASUREMENT:-throughput}"
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-16}"
  --warmup-prompts "${VLLM_WARMUP_PROMPTS:-3}"
  --warmup-tokens "${VLLM_WARMUP_TOKENS:-64}"
  --diagnostics "${VLLM_DIAGNOSTICS:-report-only}"
)

if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  ARGS+=(--max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS")
fi

if [[ "${VLLM_SEED:-42}" != "42" ]]; then
  ARGS+=(--seed "$VLLM_SEED")
fi
if [[ "$FULL_REQUESTED" != "1" && ( "${SMOKE:-0}" == "1" || "${RUN_MODE:-}" == "smoke" ) ]]; then
  ARGS+=(--smoke)
fi
if [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi

exec "$FAST_INFER_PYTHON" -u -m Benchmark.vllm_all_baselines "${ARGS[@]}" "$@"
