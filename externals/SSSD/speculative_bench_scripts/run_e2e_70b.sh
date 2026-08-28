#!/usr/bin/env bash
set -e

export MODEL_DIR="${MODEL_DIR:-/storage}" # Where to save models (large files)

DATA_DIR="${DATA_DIR:-data}" # Where to save benchmark results and other data
SAMPLE_SIZE="${SAMPLE_SIZE:-1}"  # Number of times to run each benchmark to take the median performance.
EVAL_LOGLEVEL="${EVAL_LOGLEVEL:-WARNING}"  # Set to DEBUG or INFO to see more detailed logs

EAGLE3_LLAMA_3_3_70B_PATH=${MODEL_DIR}/datasets/huggingface/models/sglang-EAGLE3-LLaMA3.3-Instruct-70B
LLAMA3_3_70B_PATH=$MODEL_DIR/datasets/huggingface/models/Llama-3.3-70B-Instruct
MAGPIE_DATASTORE_IDX_PATH=$MODEL_DIR/datasets/sssd_speculator/sssd-llama-3.3-70B.idx

start=$SECONDS

echo "Running benchmarks..."
BATCH_SIZES=(1 4 8 16 32 48 64)
OUTPUT_LEN=1024
DATASETS=("mt-bench" "math-500" "humaneval" "mt-bench-de" "swe-bench")
EVAL_DATA_DIR="$DATA_DIR/evaluation_70B/$DATASET"
mkdir -p $EVAL_DATA_DIR


for DATASET in "${DATASETS[@]}"; do
    EVAL_DATA_DIR="$DATA_DIR/evaluation/$DATASET"
    mkdir -p "$EVAL_DATA_DIR"

    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        COMMON_ARGS="--model $LLAMA3_3_70B_PATH \
            --dataset-name $DATASET \
            --sharegpt-output-len $OUTPUT_LEN \
            --disable-ignore-eos \
            --apply-chat-template \
            --enable-metrics \
            --num-prompts 100 \
            --sample-size $SAMPLE_SIZE \
            --log-level $EVAL_LOGLEVEL \
            --cuda-graph-max-bs $BATCH_SIZE \
            --max-running-requests $BATCH_SIZE \
            --tp 2"

        echo "Running autoregressive baseline on $DATASET for batch size ${BATCH_SIZE}"
        python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
            --result-filename "$EVAL_DATA_DIR/Autoregressive_${DATASET}_${BATCH_SIZE}.json"

        echo "Running EAGLE3 (default parameters) on $DATASET for batch size ${BATCH_SIZE}"
        python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
            --speculative-algorithm EAGLE3 \
            --speculative-draft-model-path $EAGLE3_LLAMA_3_3_70B_PATH \
            --result-filename "$EVAL_DATA_DIR/EAGLE3_${DATASET}_${BATCH_SIZE}.json"

        echo "Running SSSD on $DATASET for batch size: ${BATCH_SIZE}"
        python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
            --speculative-algorithm SSSD \
            --speculative-draft-model-path $MAGPIE_DATASTORE_IDX_PATH \
            --speculative-adaptive \
            --result-filename "$EVAL_DATA_DIR/SSSD_${DATASET}_${BATCH_SIZE}.json"
            # Remove --speculative-adaptive and use --use-hparam-mapping if you want to use some decent defaults
            # without dynamicity

        # echo "Running REST on $DATASET for batch size: ${BATCH_SIZE}"
        # python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
        #     --speculative-algorithm REST \
        #     --speculative-draft-model-path $MAGPIE_DATASTORE_IDX_PATH \
        #     --speculative-adaptive \
        #     --result-filename "$EVAL_DATA_DIR/REST_${DATASET}_${BATCH_SIZE}.json"

        # echo "Running PLD on $DATASET for batch size: ${BATCH_SIZE}"
        # python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
        #     --speculative-algorithm PLD \
        #     --speculative-adaptive \
        #     --result-filename "$EVAL_DATA_DIR/PLD_${DATASET}_${BATCH_SIZE}.json"

        echo "Running NGRAM on $DATASET for batch size: ${BATCH_SIZE}"
        python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
            --speculative-algorithm NGRAM \
            --result-filename "$EVAL_DATA_DIR/NGRAM_${DATASET}_${BATCH_SIZE}.json"
    done
done

echo "Gathering Results..."

for DATASET in "${DATASETS[@]}"; do
    echo "Collecting results for dataset: $DATASET"

    # Run the collector for this dataset
    python3 speculative_bench_scripts/collect_results.py \
        --data-path "$DATA_DIR" \
        --eval-benchmark "$DATASET"

    # The collector always writes to this fixed path:
    RESULTS_FILE="$DATA_DIR/collected_results.json"

    # We'll stash a dataset-specific copy so it doesn't get clobbered next iteration
    DATASET_RESULTS_FILE="$DATA_DIR/collected_results_${DATASET}.json"
    if [ -f "$RESULTS_FILE" ]; then
        mv "$RESULTS_FILE" "$DATASET_RESULTS_FILE"
    else
        echo "Error: expected $RESULTS_FILE was not created for $DATASET"
        continue
    fi
done

# After the loop, timing:
runtime=$((SECONDS - start))

hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$(((runtime % 3600) % 60))

printf "\nRuntime: %02d:%02d:%02d (hh:mm:ss)\n" "$hours" "$minutes" "$seconds"
