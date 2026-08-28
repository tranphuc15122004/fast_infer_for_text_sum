#!/usr/bin/env bash
set -euo pipefail

#############################################
# Dist-PD benchmark runner (hardened)
# - PID-aware TCP/health waits (no "refused" spam)
# - Clear error trap with failing line + log tails
# - Router starts only after workers are healthy
# - Supervisor starts only after router is up
#############################################

#: <<'USAGE'
# Example:
# ./run_dispd_bench.sh --method eagle3 \
#   --prefill-devices 0,1 --decode-devices 2,3 \
#   --prefill-tp 2 --decode-tp 2 --decode-bs 32 \
#   --datasets mt-bench --num-prompts 1000 --sharegpt-output-len 1024
#USAGE

### ========= Defaults (tune as needed) =========
MODEL_MAIN="/storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
EAGLE2_DRAFT="/storage/datasets/huggingface/models/jamesliu1/sglang-EAGLE-Llama-3.1-Instruct-8B"
EAGLE3_DRAFT="/storage/datasets/huggingface/models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"
SSSD_IDX="/storage/datasets/sssd_speculator/sssd-llama-3.1-8B.idx"

TRANSFER_BACKEND="nixl"

# Networking
HOST="127.0.0.1"              # worker bind/probe host
PREFILL_PORT="30000"
DECODE_PORT="30001"
ROUTER_HOST="0.0.0.0"         # router bind
ROUTER_PORT="8000"
ROUTER_PROBE_HOST="127.0.0.1" # where we probe router TCP

# Prefill
PREFILL_TP="2"
PREFILL_DEVICES="0,1"
PREFILL_DP="1"
PREFILL_MULT_FACTOR=8         # prefill-bs = MULT × decode-bs
PREFILL_CHUNK=""

# Decode (experiment axis)
DECODE_TP="2"
DECODE_DEVICES="2,3"
DECODE_BS="32"
METHOD="eagle3"               # autoreg | eagle2 | eagle3 | sssd | ngram

# Benchmark
BENCH_BACKEND="sglang"
DATASETS="mt-bench"
NUM_PROMPTS="1000"
SHAREGPT_OUTPUT_LEN="1024"
REQUEST_RATE="inf"

# Output
RESULTS_DIR="$(pwd)/results"

# Misc
DISABLE_TQDM="false"
OUTPUT_DETAILS="false"
### ==============================================

usage() {
  cat <<EOF
Usage: $0 [options]

--method {autoreg|eagle2|eagle3|sssd|ngram}
--model PATH
--prefill-port P              --decode-port P               --router-port P
--prefill-tp N                --decode-tp N
--prefill-dp N
--prefill-devices CSV         --decode-devices CSV
--decode-bs N                 (prefill-bs = MULT × decode-bs)
--prefill-chunk N
--datasets CSV
--num-prompts N               --sharegpt-output-len N
--request-rate R
--results-dir PATH
--disable-tqdm                --output-details
--eagle2-draft PATH           --eagle3-draft PATH           --sssd-idx PATH
EOF
}

# -------- Parse CLI --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2;;
    --model) MODEL_MAIN="$2"; shift 2;;
    --prefill-port) PREFILL_PORT="$2"; shift 2;;
    --decode-port) DECODE_PORT="$2"; shift 2;;
    --router-port) ROUTER_PORT="$2"; shift 2;;
    --prefill-tp) PREFILL_TP="$2"; shift 2;;
    --prefill-dp) PREFILL_DP="$2"; shift 2;;
    --prefill-mult-factor) PREFILL_MULT_FACTOR="$2"; shift 2;;
    --decode-tp) DECODE_TP="$2"; shift 2;;
    --prefill-devices) PREFILL_DEVICES="$2"; shift 2;;
    --decode-devices) DECODE_DEVICES="$2"; shift 2;;
    --prefill-chunk) PREFILL_CHUNK="$2"; shift 2;;
    --decode-bs) DECODE_BS="$2"; shift 2;;
    --datasets) DATASETS="$2"; shift 2;;
    --num-prompts) NUM_PROMPTS="$2"; shift 2;;
    --sharegpt-output-len) SHAREGPT_OUTPUT_LEN="$2"; shift 2;;
    --request-rate) REQUEST_RATE="$2"; shift 2;;
    --results-dir) RESULTS_DIR="$2"; shift 2;;
    --disable-tqdm) DISABLE_TQDM="true"; shift 1;;
    --output-details) OUTPUT_DETAILS="true"; shift 1;;
    --eagle2-draft) EAGLE2_DRAFT="$2"; shift 2;;
    --eagle3-draft) EAGLE3_DRAFT="$2"; shift 2;;
    --sssd-idx) SSSD_IDX="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

mkdir -p "${RESULTS_DIR}"

# ===== Derived coupling =====
if [[ ! "${DECODE_BS}" =~ ^[0-9]+$ ]] || [[ "${DECODE_BS}" -le 0 ]]; then
  echo "Invalid --decode-bs '${DECODE_BS}' (must be positive integer)"; exit 1
fi
PREFILL_BS="$(( DECODE_BS * PREFILL_MULT_FACTOR ))"
MAX_CONCURRENCY="$(( PREFILL_BS * PREFILL_DP ))"

echo ">>> Derived settings:"
echo "    decode-bs            = ${DECODE_BS}"
echo "    prefill-bs           = ${PREFILL_BS}   (${PREFILL_MULT_FACTOR} x decode-bs)"
echo "    max-concurrency      = ${MAX_CONCURRENCY}   (== prefill-bs x prefill-dp-size)"
# ===========================

# Validate TP vs device lists
IFS=',' read -r -a prefill_ids <<< "$PREFILL_DEVICES"
IFS=',' read -r -a decode_ids  <<< "$DECODE_DEVICES"

expected_prefill_gpus=$(( PREFILL_TP * PREFILL_DP ))
[[ "${#prefill_ids[@]}" -eq "${expected_prefill_gpus}" ]] || {
  echo "PREFILL_TP=${PREFILL_TP}, PREFILL_DP=${PREFILL_DP} "
  echo "=> expected ${expected_prefill_gpus} prefill GPUs but got ${#prefill_ids[@]} ids (${PREFILL_DEVICES})"
  exit 1
}

[[ "${#decode_ids[@]}" -eq "${DECODE_TP}" ]] || {
  echo "DECODE_TP=${DECODE_TP} but DECODE_DEVICES has ${#decode_ids[@]} ids (${DECODE_DEVICES})"
  exit 1
}

BASE_GPU_ID_DECODE="0"   # relative to CUDA_VISIBLE_DEVICES

# -------- Logging & traps --------
LOG_ROOT="/workspace/sglang-sssd/data/logs"
logdir="${LOG_ROOT}/logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$logdir"

PIDS=()
prefill_log="${logdir}/prefill.log"
decode_log="${logdir}/decode.log"
router_log="${logdir}/router.log"

dump_last_logs () {
  echo "---- Last 200 lines of prefill log ----"; tail -n 200 "$prefill_log" 2>/dev/null || true
  echo "---- Last 200 lines of decode log ----";  tail -n 200 "$decode_log"  2>/dev/null || true
  echo "---- Last 200 lines of router log ----";  tail -n 200 "$router_log"  2>/dev/null || true
}

on_err() {
  local exit_code=$? line_no=$1
  echo
  echo "✗ ERROR on line ${line_no} (exit ${exit_code}). Dumping logs…"
  dump_last_logs
}
set -o errtrace
trap 'on_err $LINENO' ERR

cleanup() {
  echo; echo ">>> Cleaning up…"
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    sleep 0.3 || true
    kill "${PIDS[@]}" 2>/dev/null || true
    sleep 1 || true
    kill -9 "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

timestamp() { date +%Y%m%d_%H%M%S; }

# -------- PID-aware waits --------
wait_for_tcp_pid () {
  local host="$1" port="$2" pid="$3" timeout="${4:-900}"
  echo -n "Waiting for TCP ${host}:${port} (pid=${pid}) "
  local start; start=$(date +%s)
  while true; do
    # Abort if process died
    if ! kill -0 "$pid" 2>/dev/null; then
      echo; echo "✗ ${host}:${port} failed: process $pid exited"
      return 1
    fi
    # Try to connect (quietly)
    if exec 3<>/dev/tcp/"$host"/"$port" 2>/dev/null; then
      exec 3>&-
      echo "— up!"
      return 0
    fi
    # Timeout check after attempt
    if (( $(date +%s) - start > timeout )); then
      echo; echo "✗ Timeout waiting for ${host}:${port}"
      return 1
    fi
    echo -n "."
    sleep 1
  done
}

wait_for_health_pid () {
  local url="$1" pid="$2" timeout="${3:-900}"
  echo -n "Waiting for health 200 from ${url} (pid=${pid}) "
  local start; start=$(date +%s)
  while true; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo; echo "✗ ${url} failed: process $pid exited"
      return 1
    fi
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "— healthy!"
      return 0
    fi
    if (( $(date +%s) - start > timeout )); then
      echo; echo "✗ Timeout waiting for health on ${url}"
      return 1
    fi
    echo -n "."
    sleep 1
  done
}

#############################################
# 1) Launch PREFILL
#############################################
prefill_cmd=( python3 -m sglang.launch_server
  --model "${MODEL_MAIN}"
  --disaggregation-mode prefill
  --port "${PREFILL_PORT}"
  --disaggregation-transfer-backend "${TRANSFER_BACKEND}"
  --tp "${PREFILL_TP}"
  --cuda-graph-max-bs "${PREFILL_BS}"
  --max-running-requests "${PREFILL_BS}"
  --chunked-prefill-size -1
)
if [[ "${PREFILL_DP}" -gt 1 ]]; then
  prefill_cmd+=( --dp "${PREFILL_DP}" )
fi
case "$METHOD" in
  autoreg) : ;;
  eagle2)  prefill_cmd+=( --speculative-algorithm EAGLE  --speculative-draft-model-path "${EAGLE2_DRAFT}" ) ;;
  eagle3)  prefill_cmd+=( --speculative-algorithm EAGLE3 --speculative-draft-model-path "${EAGLE3_DRAFT}" ) ;;
  ngram|sssd|pld|rest) : ;;
  *) echo "Unsupported --method $METHOD"; exit 1 ;;
esac
# Only set chunked prefill size if user passed --prefill-chunk
if [[ -n "${PREFILL_CHUNK}" ]]; then
  prefill_cmd+=( --chunked-prefill-size "${PREFILL_CHUNK}" )
fi
echo ">>> Launching PREFILL on :${PREFILL_PORT} | TP=${PREFILL_TP} | DP=${PREFILL_DP} | GPUs=${PREFILL_DEVICES}"
echo "+ ${prefill_cmd[*]}" | tee -a "$prefill_log"
CUDA_VISIBLE_DEVICES="${PREFILL_DEVICES}" "${prefill_cmd[@]}" >>"$prefill_log" 2>&1 & PREFILL_PID=$!
PIDS+=("$PREFILL_PID")

# Wait for prefill: TCP then health
wait_for_tcp_pid    "$HOST" "$PREFILL_PORT" "$PREFILL_PID" 900
wait_for_health_pid "http://${HOST}:${PREFILL_PORT}/health" "$PREFILL_PID" 900

#############################################
# 2) Launch DECODE
#############################################
decode_cmd=( python3 -m sglang.launch_server
  --model "${MODEL_MAIN}"
  --disaggregation-mode decode
  --port "${DECODE_PORT}"
  --base-gpu-id "0"                    # relative to CUDA_VISIBLE_DEVICES
  --disaggregation-transfer-backend "${TRANSFER_BACKEND}"
  --tp "${DECODE_TP}"
  --cuda-graph-max-bs "${DECODE_BS}"
  --max-running-requests "${DECODE_BS}"
  # --disable-cuda-graph-padding
)
case "$METHOD" in
  autoreg) : ;;
  eagle2)  decode_cmd+=( --speculative-algorithm EAGLE  --speculative-draft-model-path "${EAGLE2_DRAFT}" ) ;;
  eagle3)  decode_cmd+=( --speculative-algorithm EAGLE3 --speculative-draft-model-path "${EAGLE3_DRAFT}" ) ;;
  ngram)   decode_cmd+=( --speculative-algorithm NGRAM ) ;;
  sssd)    decode_cmd+=( --speculative-algorithm SSSD  --speculative-draft-model-path "${SSSD_IDX}" --speculative-adaptive ) ;;
  pld)    decode_cmd+=( --speculative-algorithm PLD --speculative-adaptive ) ;;
  rest)    decode_cmd+=( --speculative-algorithm REST  --speculative-draft-model-path "${SSSD_IDX}" --speculative-adaptive ) ;;
  *) echo "Unsupported --method $METHOD"; exit 1 ;;
esac
echo ">>> Launching DECODE (${METHOD}) on :${DECODE_PORT} | TP=${DECODE_TP} | GPUs=${DECODE_DEVICES}"
echo "+ ${decode_cmd[*]}" | tee -a "$decode_log"
CUDA_VISIBLE_DEVICES="${DECODE_DEVICES}" "${decode_cmd[@]}" >>"$decode_log" 2>&1 & DECODE_PID=$!
PIDS+=("$DECODE_PID")

# Wait for decode: TCP then health
wait_for_tcp_pid    "$HOST" "$DECODE_PORT" "$DECODE_PID" 900
wait_for_health_pid "http://${HOST}:${DECODE_PORT}/health" "$DECODE_PID" 900

#############################################
# 3) Launch ROUTER (after both workers are healthy)
#############################################
router_cmd=( python -m sglang_router.launch_router
  --pd-disaggregation
  --prefill "http://${HOST}:${PREFILL_PORT}"
  --decode  "http://${HOST}:${DECODE_PORT}"
  --host "${ROUTER_HOST}"
  --port "${ROUTER_PORT}"
)
echo ">>> Launching ROUTER on :${ROUTER_PORT}"
echo "+ ${router_cmd[*]}" | tee -a "$router_log"
"${router_cmd[@]}" >>"$router_log" 2>&1 & ROUTER_PID=$!
PIDS+=("$ROUTER_PID")

# Wait for router TCP
echo -n "Waiting for TCP ${ROUTER_PROBE_HOST}:${ROUTER_PORT} "
_start=$(date +%s)
while ! exec 3<>/dev/tcp/"$ROUTER_PROBE_HOST"/"$ROUTER_PORT" 2>/dev/null; do
  echo -n "."
  sleep 1
  (( $(date +%s) - _start > 180 )) && { echo; echo "✗ Timeout ${ROUTER_PROBE_HOST}:${ROUTER_PORT}"; dump_last_logs; exit 1; }
done
exec 3>&-; echo "— up!"
echo "Router is up at http://${ROUTER_PROBE_HOST}:${ROUTER_PORT}"

#############################################
# 4) Start supervisor AFTER router is up
#############################################
supervisor() {
  local dying=""
  while true; do
    if ! kill -0 "$PREFILL_PID" 2>/dev/null; then dying="PREFILL"; break; fi
    if ! kill -0 "$DECODE_PID" 2>/dev/null; then dying="DECODE";  break; fi
    if ! kill -0 "$ROUTER_PID" 2>/dev/null; then dying="ROUTER";  break; fi
    sleep 1
  done
  echo; echo ">>> ${dying} process exited unexpectedly. Aborting benchmark."
  case "$dying" in
    PREFILL) tail -n 200 "$prefill_log" || true ;;
    DECODE)  tail -n 200 "$decode_log"  || true ;;
    ROUTER)  tail -n 200 "$router_log"  || true ;;
  esac
  sleep 1 || true
  kill $$ >/dev/null 2>&1 || true
}
supervisor & SUPERVISOR_PID=$!
trap 'kill "$SUPERVISOR_PID" 2>/dev/null || true; cleanup' EXIT

#############################################
# 5) Run benchmarks
#############################################
echo ">>> Running benchmark(s)…"
mkdir -p "${RESULTS_DIR}"
IFS=',' read -r -a dataset_list <<< "$DATASETS"
for ds in "${dataset_list[@]}"; do
  ds_trim="$(echo "$ds" | xargs)"
  out_file="${RESULTS_DIR}/pd_${METHOD}_${ds_trim}_bs_${DECODE_BS}_$(timestamp).json"

  bench_cmd=( python3 -m sglang.bench_serving
    --backend "${BENCH_BACKEND}"
    --base-url "http://${ROUTER_PROBE_HOST}:${ROUTER_PORT}"
    --dataset-name "${ds_trim}"
    --num-prompts "${NUM_PROMPTS}"
    --sharegpt-output-len "${SHAREGPT_OUTPUT_LEN}"
    --disable-ignore-eos
    --request-rate "${REQUEST_RATE}"
    --max-concurrency "${MAX_CONCURRENCY}"
    --pd-separated
    --output-file "${out_file}"
  )
  [[ "${DISABLE_TQDM}" == "true" ]] && bench_cmd+=( --disable-tqdm )
  [[ "${OUTPUT_DETAILS}" == "true" ]] && bench_cmd+=( --output-details )

  echo ">>> Running benchmark for dataset='${ds_trim}' -> ${out_file}"
  echo "+ ${bench_cmd[*]}"
  "${bench_cmd[@]}"
done

# Stop supervisor before cleanup so it doesn't race teardown
kill "$SUPERVISOR_PID" 2>/dev/null || true
wait "$SUPERVISOR_PID" 2>/dev/null || true

echo ">>> All benchmarks complete."
echo "    Results in: ${RESULTS_DIR}"
echo "    Logs in:    ${logdir}"
