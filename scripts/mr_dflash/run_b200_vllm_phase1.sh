#!/usr/bin/env bash
# MR-DFlash B200 launcher:
#   mode 1: start and health-check vLLM only
#   mode 2: start/reuse vLLM, regenerate all full-context splits, stop vLLM,
#           then validate/tokenize/cache train+val

set -u

usage() {
  cat <<'EOF'
Usage:
  bash scripts/mr_dflash/run_b200_vllm_phase1.sh 1 [options]
  bash scripts/mr_dflash/run_b200_vllm_phase1.sh 2 [options]

Modes:
  1    Start vLLM, wait for /health and /v1/models, then return.
  2    Start/reuse vLLM, run regenerate_full_{train,val,test}, stop vLLM,
       then run validate/tokenize/cache for Phase 1.

Options:
  --run-root PATH       Output root; default is under phuc_projects/outputs.
  --prepared-root PATH  Existing prepare root containing normalized/*.jsonl.
  --target-model PATH   Local Qwen3-4B model path.
  --port PORT           vLLM loopback port; default 38147.
  --gpu-id ID           Physical GPU id; default 0.
  -h, --help            Show this help.

The launcher intentionally uses a visible stop_parallel file and visible
output/log directories. It does not export VLLM_TMP because vLLM treats that
name as an unknown environment variable.
EOF
}

die() {
  echo "[b200-vllm] ERROR: $*" >&2
  exit 2
}

MODE="${1:-}"
if [[ "$MODE" == "--mode" ]]; then
  [[ $# -ge 2 ]] || die "--mode cần 1 hoặc 2"
  MODE="$2"
  shift 2
else
  shift || true
fi

case "$MODE" in
  1|2) ;;
  -h|--help|"") usage; [[ "$MODE" == "" ]] && exit 2 || exit 0 ;;
  *) die "mode phải là 1 hoặc 2, nhận được: $MODE" ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/../outputs}"
RUN_ROOT="${RUN_ROOT:-$OUTPUT_ROOT/mr_dflash_phase1_20k_sharegpt_30k_arxiv_vllm_b200}"
PREPARED_ROOT="${PREPARED_ROOT:-$PROJECT_ROOT/../data/mr_dflash_phase1_20k_sharegpt_30k_arxiv}"
TARGET_MODEL="${TARGET_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B}"
RUNTIME_TMP="${RUNTIME_TMP:-$PROJECT_ROOT/../vllm_tmp}"
VLLM_PORT="${VLLM_PORT:-38147}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_BIN="${VLLM_BIN:-vllm}"

SHAREGPT_SOURCE="${SHAREGPT_SOURCE:-/workspace/storage-shared/nlp/tungdd11/tungdecoder/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json}"
ARXIV_SOURCE="${ARXIV_SOURCE:-/workspace/storage-shared/nlp/dungdx4/datasets/arxiv/train.label.jsonl}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root)
      [[ $# -ge 2 ]] || die "--run-root cần PATH"
      RUN_ROOT="$2"
      shift 2
      ;;
    --prepared-root)
      [[ $# -ge 2 ]] || die "--prepared-root cần PATH"
      PREPARED_ROOT="$2"
      shift 2
      ;;
    --target-model)
      [[ $# -ge 2 ]] || die "--target-model cần PATH"
      TARGET_MODEL="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || die "--port cần số"
      VLLM_PORT="$2"
      shift 2
      ;;
    --gpu-id)
      [[ $# -ge 2 ]] || die "--gpu-id cần số"
      GPU_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "option không hợp lệ: $1"
      ;;
  esac
done

export PROJECT_ROOT OUTPUT_ROOT RUN_ROOT PREPARED_ROOT TARGET_MODEL RUNTIME_TMP VLLM_PORT GPU_ID
export TMPDIR="$RUNTIME_TMP"
export TEMP="$RUNTIME_TMP"
export TMP="$RUNTIME_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FI_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_CACHE_ROOT="$RUN_ROOT/vllm_cache"
export TRITON_CACHE_DIR="$RUN_ROOT/triton_cache"
export TORCH_EXTENSIONS_DIR="$RUN_ROOT/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$RUN_ROOT/flashinfer_workspace"

unset VLLM_TMP
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT

LOG_ROOT="$RUN_ROOT/logs"
PID_FILE="$LOG_ROOT/vllm.pid"
VLLM_LOG="$LOG_ROOT/vllm.log"
STOP_FILE="$RUN_ROOT/stop_parallel"
mkdir -p \
  "$LOG_ROOT" \
  "$RUNTIME_TMP" \
  "$VLLM_CACHE_ROOT" \
  "$TRITON_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$FLASHINFER_WORKSPACE_BASE"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "không tìm thấy Python: $PYTHON_BIN"
command -v "$VLLM_BIN" >/dev/null 2>&1 || die "không tìm thấy vLLM: $VLLM_BIN; hãy activate phucvenv trước"
command -v curl >/dev/null 2>&1 || die "cần curl để health-check vLLM"
[[ -d "$TARGET_MODEL" ]] || die "không tìm thấy target model: $TARGET_MODEL"

SERVER_ADDRESS="http://127.0.0.1:${VLLM_PORT}/v1"
HEALTH_ADDRESS="http://127.0.0.1:${VLLM_PORT}/health"
MODELS_ADDRESS="http://127.0.0.1:${VLLM_PORT}/v1/models"

server_ready() {
  curl -fsS --max-time 3 "$HEALTH_ADDRESS" >/dev/null 2>&1 || return 1
  curl -fsS --max-time 3 "$MODELS_ADDRESS" 2>/dev/null \
    | grep -q 'qwen3-4b' || return 1
  return 0
}

print_server_failure() {
  echo "[b200-vllm] vLLM chưa ready; log cuối: $VLLM_LOG" >&2
  tail -n 100 "$VLLM_LOG" 2>/dev/null || true
}

start_server() {
  if server_ready; then
    if [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "[b200-vllm] reuse vLLM PID=$(cat "$PID_FILE") port=$VLLM_PORT"
      SERVER_MANAGED=1
      return 0
    fi
    if [[ "$MODE" == "2" ]]; then
      die "port $VLLM_PORT đang là vLLM bên ngoài nhưng không có PID file; không thể tự dừng trước cache"
    fi
    echo "[b200-vllm] vLLM bên ngoài đã ready trên port=$VLLM_PORT"
    SERVER_MANAGED=0
    return 0
  fi

  if [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    die "PID file tồn tại nhưng server chưa ready: $(cat "$PID_FILE"); xem $VLLM_LOG"
  fi

  : > "$VLLM_LOG"
  echo "[b200-vllm] starting vLLM on GPU=$GPU_ID port=$VLLM_PORT"
  nohup env \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    TMPDIR="$TMPDIR" \
    TEMP="$TEMP" \
    TMP="$TMP" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    FI_OFFLINE=1 \
    VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
    TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" \
    FLASHINFER_WORKSPACE_BASE="$FLASHINFER_WORKSPACE_BASE" \
    "$VLLM_BIN" serve "$TARGET_MODEL" \
    --served-model-name qwen3-4b \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 262144 \
    --attention-config '{"backend":"FLASH_ATTN"}' \
    --kernel-config '{"enable_flashinfer_autotune":false}' \
    --enforce-eager \
    --kv-cache-metrics \
    > "$VLLM_LOG" 2>&1 < /dev/null &
  echo "$!" > "$PID_FILE"
  SERVER_MANAGED=1

  local attempt
  for attempt in $(seq 1 180); do
    if server_ready; then
      echo "[b200-vllm] READY pid=$(cat "$PID_FILE") address=$SERVER_ADDRESS"
      return 0
    fi
    if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      print_server_failure
      return 1
    fi
    sleep 2
  done
  print_server_failure
  return 1
}

stop_server() {
  [[ "${SERVER_MANAGED:-0}" == "1" ]] || return 0
  [[ -s "$PID_FILE" ]] || return 0
  local pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "[b200-vllm] stopping vLLM pid=$pid before HF cache"
    kill -TERM "$pid" 2>/dev/null || true
    local attempt
    for attempt in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 2
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
}

prepare_phase1_root() {
  mkdir -p "$RUN_ROOT"
  if [[ ! -e "$RUN_ROOT/normalized" ]]; then
    [[ -d "$PREPARED_ROOT/normalized" ]] || die "thiếu prepare root: $PREPARED_ROOT/normalized"
    ln -s "$PREPARED_ROOT/normalized" "$RUN_ROOT/normalized"
  fi
  local split
  for split in train val test; do
    [[ -s "$RUN_ROOT/normalized/${split}_prompts.jsonl" ]] \
      || die "thiếu file prepare: $RUN_ROOT/normalized/${split}_prompts.jsonl"
  done
}

run_phase1() {
  prepare_phase1_root
  local common_args=(
    --repo-root "$PROJECT_ROOT"
    --data-root "$RUN_ROOT"
    --target-model-path "$TARGET_MODEL"
    --sharegpt-source "$SHAREGPT_SOURCE"
    --arxiv-source "$ARXIV_SOURCE"
    --sharegpt-count 20000
    --arxiv-count 30000
    --full-context
    --full-context-length 32768
    --cache-splits train val
    --target-layer-ids 1 9 17 25 33
    --supervision-mode last_assistant
    --seed 42
    --max-new-tokens 2048
    --overflow-policy error
    --sample-error-policy error
    --temperature 0.0
    --regenerate-backend vllm
    --vllm-server-addresses "$SERVER_ADDRESS"
    --vllm-model qwen3-4b
    --vllm-request-concurrency 128
    --vllm-request-concurrency-start 8
    --vllm-max-batched-tokens 262144
    --vllm-max-batched-tokens-start 65536
    --vllm-request-growth-factor 2.0
    --vllm-request-timeout-seconds 600
    --vllm-request-retries 2
    --vllm-metrics-address "http://127.0.0.1:${VLLM_PORT}/metrics"
    --vllm-metrics-poll-interval-seconds 0.5
    --vllm-gpu-cache-target 0.85
    --vllm-gpu-cache-hard 0.95
    --device cuda
    --torch-dtype bfloat16
    --cache-batch-size-3k 4
    --cache-batch-size-long-context 1
    --cache-bucket-buffer-3k 16
    --cache-bucket-buffer-long-context 8
    --cache-shard-size-3k 64
    --cache-shard-size-long-context 32
    --cache-attention-backend sdpa
    --cache-backend hf
    --cache-io-threads 2
    --cache-io-queue-size 4
    --parallel-gpu-ids "$GPU_ID"
    --parallel-scheduler shared_lease
    --parallel-queue-quantum-items 512
    --progress-interval-tokens 256
    --regenerate-output-batch-size 1
    --worker-stall-timeout-seconds 0
    --local-files-only
    --resume
  )

  echo "[b200-vllm] phase 1 regenerate: outputs=$RUN_ROOT"
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/mr_dflash/run_preprocess_pipeline.py" \
    "${common_args[@]}" \
    --from-stage regenerate_full_train \
    --stop-after regenerate_full_test
  local regenerate_rc=$?
  if [[ $regenerate_rc -ne 0 ]]; then
    echo "[b200-vllm] regenerate failed rc=$regenerate_rc; keeping log/output for resume" >&2
    return "$regenerate_rc"
  fi

  stop_server
  SERVER_MANAGED=0
  echo "[b200-vllm] vLLM stopped; phase 1 cache can use the GPU"

  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/mr_dflash/run_preprocess_pipeline.py" \
    "${common_args[@]}" \
    --from-stage validate_full_train
  local remaining_rc=$?
  echo "[b200-vllm] phase 1 remaining rc=$remaining_rc"
  return "$remaining_rc"
}

SERVER_MANAGED=0
if [[ "$MODE" == "1" ]]; then
  if start_server; then
    echo "[b200-vllm] mode 1 completed; vLLM remains running"
    exit 0
  fi
  exit 1
fi

if ! start_server; then
  exit 1
fi

cleanup_mode2() {
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[b200-vllm] mode 2 failed rc=$rc; stopping managed vLLM" >&2
  fi
  # Uninstall the EXIT/INT/TERM trap before exiting; otherwise exit from the
  # cleanup function can re-enter the same trap on some bash versions.
  trap - EXIT INT TERM
  stop_server
  exit "$rc"
}
trap cleanup_mode2 EXIT INT TERM

run_phase1
