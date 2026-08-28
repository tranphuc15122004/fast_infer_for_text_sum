#!/usr/bin/env bash
set -e

export MODEL_DIR="${MODEL_DIR:-/storage}" # Where to save models (large files)

DATA_DIR="${DATA_DIR:-data}" # Where to save benchmark results and other data
SAMPLE_SIZE="${SAMPLE_SIZE:-1}"  # Number of times to run each benchmark to take the median performance.
EVAL_LOGLEVEL="${EVAL_LOGLEVEL:-WARNING}"  # Set to DEBUG or INFO to see more detailed logs

EAGLE2_LLAMA_3_1_8B_PATH=${MODEL_DIR}/datasets/huggingface/models/jamesliu1/sglang-EAGLE-Llama-3.1-Instruct-8B
EAGLE3_LLAMA_3_1_8B_PATH=${MODEL_DIR}/datasets/huggingface/models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B
LLAMA3_1_8B_PATH=$MODEL_DIR/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/
STANDALONE_DRAFT_PATH=${MODEL_DIR}/datasets/huggingface/models/Llama-3.2-1B-Instruct
MAGPIE_DATASTORE_IDX_PATH=$MODEL_DIR/datasets/sssd_speculator/sssd-llama-3.1-8B.idx


start=$SECONDS

echo "Running benchmarks..."
BATCH_SIZES=(1 4 8 16 32 48 64)
OUTPUT_LEN=1024
DATASETS=("math-500" "humaneval" "mt-bench-de" "mt-bench")

# Map batch size -> num-prompts
declare -A NUM_PROMPTS_BY_BS=(
    [1]=40
    [4]=40
    [8]=80
    [16]=80
    [32]=96
    [48]=144
    [64]=192
)

for DATASET in "${DATASETS[@]}"; do
    EVAL_DATA_DIR="$DATA_DIR/evaluation/$DATASET"
    mkdir -p "$EVAL_DATA_DIR"

    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        NUM_PROMPTS=${NUM_PROMPTS_BY_BS[$BATCH_SIZE]}
        if [ -z "$NUM_PROMPTS" ]; then
            echo "Warning: no num-prompts configured for batch size $BATCH_SIZE, defaulting to 80"
            NUM_PROMPTS=80
        fi

        COMMON_ARGS="--model $LLAMA3_1_8B_PATH \
            --dataset-name $DATASET \
            --sharegpt-output-len $OUTPUT_LEN \
            --disable-ignore-eos \
            --apply-chat-template \
            --enable-metrics \
            --sample-size $SAMPLE_SIZE \
            --log-level $EVAL_LOGLEVEL \
            --cuda-graph-max-bs $BATCH_SIZE \
            --max-running-requests $BATCH_SIZE \
            --num-prompts $NUM_PROMPTS"

        echo "Running autoregressive baseline on $DATASET for batch size ${BATCH_SIZE}"
        python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
            --result-filename "$EVAL_DATA_DIR/Autoregressive_${DATASET}_${BATCH_SIZE}.json"

        echo "Running STANDALONE (default parameters) on $DATASET for batch size ${BATCH_SIZE}"
        python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
            --speculative-algorithm STANDALONE \
            --speculative-draft-model-path $STANDALONE_DRAFT_PATH \
            --result-filename "$EVAL_DATA_DIR/STANDALONE_${DATASET}_${BATCH_SIZE}.json"

        echo "Running EAGLE2 (default parameters) on $DATASET for batch size ${BATCH_SIZE}"
        python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
            --speculative-algorithm EAGLE \
            --speculative-draft-model-path $EAGLE2_LLAMA_3_1_8B_PATH \
            --result-filename "$EVAL_DATA_DIR/EAGLE2_${DATASET}_${BATCH_SIZE}.json"

        echo "Running EAGLE3 (default parameters) on $DATASET for batch size ${BATCH_SIZE}"
        python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
            --speculative-algorithm EAGLE3 \
            --speculative-draft-model-path $EAGLE3_LLAMA_3_1_8B_PATH \
            --result-filename "$EVAL_DATA_DIR/EAGLE3_${DATASET}_${BATCH_SIZE}.json"

        echo "Running SSSD on $DATASET for batch size: ${BATCH_SIZE}"
        python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
            --speculative-algorithm SSSD \
            --speculative-draft-model-path $MAGPIE_DATASTORE_IDX_PATH \
            --speculative-adaptive \
            --result-filename "$EVAL_DATA_DIR/SSSD_${DATASET}_${BATCH_SIZE}.json"
            # Remove --speculative-adaptive and use --use-hparam-mapping if you want to use some decent defaults
            # without dynamicity

        echo "Running REST on $DATASET for batch size: ${BATCH_SIZE}"
        python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
            --speculative-algorithm REST \
            --speculative-draft-model-path $MAGPIE_DATASTORE_IDX_PATH \
            --speculative-adaptive \
            --result-filename "$EVAL_DATA_DIR/REST_${DATASET}_${BATCH_SIZE}.json"

        # echo "Running PIA on $DATASET for batch size: ${BATCH_SIZE}"
        # python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
        #     --speculative-algorithm PIA \
        #     --speculative-draft-model-path $PIA_CACHE_PATH \
        #     --speculative-adaptive \
        #     --result-filename "$EVAL_DATA_DIR/PIA_${DATASET}_${BATCH_SIZE}.json"

        echo "Running PLD on $DATASET for batch size: ${BATCH_SIZE}"
        python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
            --speculative-algorithm PLD \
            --speculative-adaptive \
            --result-filename "$EVAL_DATA_DIR/PLD_${DATASET}_${BATCH_SIZE}.json"

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
        --submission-url "$RESULT_REPO_URL" \
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
