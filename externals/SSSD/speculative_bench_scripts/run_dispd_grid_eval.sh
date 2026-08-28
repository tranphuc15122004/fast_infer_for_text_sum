#!/usr/bin/env bash
set -euo pipefail

# Decoding methods and beam sizes to sweep
METHODS=("autoreg" "eagle3" "sssd" "ngram" "pld")
DECODE_BS_LIST=(1 4 8 16 32 48)

# Datasets (regular multi-dataset run)
DATASETS="mt-bench,humaneval,math-500,mt-bench-fr,mt-bench-ru"

MODEL_MAIN="/storage/datasets/huggingface/models/Llama-3.3-70B-Instruct"
EAGLE2_DRAFT="/storage/datasets/huggingface/models/jamesliu1/sglang-EAGLE-Llama-3.1-Instruct-8B"
EAGLE3_DRAFT="/storage/datasets/huggingface/models/sglang-EAGLE3-LLaMA3.3-Instruct-70B"
SSSD_IDX="/storage/datasets/sssd_speculator/sssd-llama-3.3-70B.idx"

# num_prompts rule for the REGULAR RUN ONLY:
#   num_prompts = max(MIN_NUM_PROMPTS, min(MAX_NUM_PROMPTS, 10 * <decode batch size>)
MIN_NUM_PROMPTS=80
MAX_NUM_PROMPTS=256
PROMPTS_PER_BS=10
PREFILL_MULT_FACTOR=16

SHAREGPT_OUTPUT_LEN=1024
RESULTS_DIR="/workspace/sglang-sssd/data/results"
LOG_ROOT="/workspace/sglang-sssd/data/grid_logs"

# Device / parallelism config
PREFILL_DEVICES="2,3"
DECODE_DEVICES="0,1"
PREFILL_TP=2
PREFILL_DP=1
DECODE_TP=2

# Retry policy
RETRIES=0          # number of retries after the first failure (0 = no retry)
RETRY_SLEEP=5      # seconds to wait before retry

mkdir -p "${RESULTS_DIR}" "${LOG_ROOT}"

# Keep a summary of failures
declare -a FAILED_RUNS=()

timestamp() { date +%Y%m%d_%H%M%S; }

for bs in "${DECODE_BS_LIST[@]}"; do
  for method in "${METHODS[@]}"; do
    echo

    RUN_NUM_PROMPTS=$(( bs * PROMPTS_PER_BS ))
    if (( RUN_NUM_PROMPTS > MAX_NUM_PROMPTS )); then
      RUN_NUM_PROMPTS=${MAX_NUM_PROMPTS}
    fi
    if (( RUN_NUM_PROMPTS < MIN_NUM_PROMPTS )); then
      RUN_NUM_PROMPTS=${MIN_NUM_PROMPTS}
    fi

    echo "========== RUN: method=${method} decode_bs=${bs} num_prompts=${RUN_NUM_PROMPTS} =========="
    run_id="${method}_bs${bs}_$(timestamp)"
    log_file="${LOG_ROOT}/${run_id}.log"

    attempt=0
    success=false
    while : ; do
      attempt=$((attempt + 1))
      echo "--- Attempt ${attempt} for ${run_id} ---" | tee -a "${log_file}"

      # Run one stack; the stack script will derive prefill-bs / concurrency etc.
      if ./run_dispd_bench.sh \
            --method "${method}" \
            --prefill-devices "${PREFILL_DEVICES}" \
            --decode-devices "${DECODE_DEVICES}" \
            --prefill-tp "${PREFILL_TP}" \
            --prefill-dp "${PREFILL_DP}" \
            --prefill-mult-factor "${PREFILL_MULT_FACTOR}" \
            --decode-tp "${DECODE_TP}" \
            --decode-bs "${bs}" \
            --datasets "${DATASETS}" \
            --num-prompts "${RUN_NUM_PROMPTS}" \
            --sharegpt-output-len "${SHAREGPT_OUTPUT_LEN}" \
            --results-dir "${RESULTS_DIR}" \
            --disable-tqdm \
            --model "${MODEL_MAIN}" \
            --eagle2-draft "${EAGLE2_DRAFT}" \
            --eagle3-draft "${EAGLE3_DRAFT}" \
            --sssd-idx "${SSSD_IDX}" \
            | tee -a "${log_file}"; then
        success=true
        echo "✓ SUCCESS: ${run_id}" | tee -a "${log_file}"
        break
      else
        echo "✗ FAILED: ${run_id} (attempt ${attempt})" | tee -a "${log_file}"
        if [[ ${attempt} -le ${RETRIES} ]]; then
          echo "…retrying in ${RETRY_SLEEP}s" | tee -a "${log_file}"
          sleep "${RETRY_SLEEP}"
        else
          break
        fi
      fi
    done

    if [[ "${success}" != "true" ]]; then
      FAILED_RUNS+=("${run_id}")
    fi
  done
done

echo
echo ">>> GRID COMPLETE. Results in: ${RESULTS_DIR}"
echo "    Logs in:    ${LOG_ROOT}"

if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
  echo
  echo "The following runs failed (see logs for details):"
  for r in "${FAILED_RUNS[@]}"; do
    echo "  - ${r}"
  done
  exit 2
fi
