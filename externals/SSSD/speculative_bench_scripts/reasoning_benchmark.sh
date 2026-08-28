#!/usr/bin/env bash
# Downloads models, creates SSSD datastores, and runs benchmarks
set -e

export MODEL_DIR="${MODEL_DIR:-/storage}" # Where to save models (large files)

DATA_DIR="${DATA_DIR:-data_reasoning}" # Where to save benchmark results and other data
SAMPLE_SIZE="${SAMPLE_SIZE:-5}"  # Number of times to run each benchmark to take the median performance.
EVAL_LOGLEVEL="${EVAL_LOGLEVEL:-WARNING}"  # Set to DEBUG or INFO to see more detailed logs

echo "Getting user machine specs..."
python3 collect_env.py --out-dir "$DATA_DIR"

echo "Downloading models..."
models=(
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B main ${MODEL_DIR}/datasets/huggingface/models/DeepSeek-R1-Distill-Llama-8B"
  "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B main ${MODEL_DIR}/datasets/huggingface/models/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"
)

# 4. Loop through and download
for entry in "${models[@]}"; do
  read -r repo commit target <<< "$entry"
  echo
  echo "➡️  Downloading '$repo' at revision '$commit' into '$target'"

  mkdir -p "$target"
    hf download "$repo" \
    ${commit:+--revision "$commit"} \
    --repo-type model \
    --local-dir "$target" \
    --exclude "*consolidated*"

  echo "✅ Done: $repo"
done


echo "Converting EAGLE head to SGLang format..."
EAGLE3_PATH=${MODEL_DIR}/datasets/huggingface/models/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B
python3 change_model_type.py --model $EAGLE3_PATH


echo "Creating SSSD datastore..."
MODEL_PATH=$MODEL_DIR/datasets/huggingface/models/DeepSeek-R1-Distill-Llama-8B
SSSD_DATASTORE_PATH=$MODEL_DIR/datasets/sssd_speculator/sssd-llama-3-8B-Deepseek.idx

python3 ../sssd_speculator/datastore_creation/create_datastore.py --model $MODEL_PATH --index_file_path $SSSD_DATASTORE_PATH --datasets deepseek-r1-llama70B deepseek-r1-dolphin deepseek-r1-distill deepseek-r1-math-response

start=$SECONDS

echo "Running benchmarks..."
BATCH_SIZES=(1 4 8 16 32 48 64)
OUTPUT_LEN=4096
DATASETS=("math-500")

# Map batch size -> num-prompts
declare -A NUM_PROMPTS_BY_BS=(
    [1]=40
    [4]=40
    [8]=80
    [16]=80
    [32]=80
    [48]=80
    [64]=100
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

        COMMON_ARGS="--model $MODEL_PATH \
            --dataset-name $DATASET \
            --sharegpt-output-len $OUTPUT_LEN \
            --disable-ignore-eos \
            --apply-chat-template \
            --enable-metrics \
            --sample-size $SAMPLE_SIZE \
            --log-level $EVAL_LOGLEVEL \
            --cuda-graph-max-bs $BATCH_SIZE \
            --max-running-requests $BATCH_SIZE \
            --num-prompts $NUM_PROMPTS \
            --temperature 0.6 \
            --top-p 0.95 \
            --keep-all-samples"

        echo "Running autoregressive baseline on $DATASET for batch size ${BATCH_SIZE}"
        python3 eval_benchmark.py $COMMON_ARGS \
            --result-filename "$EVAL_DATA_DIR/Autoregressive_${DATASET}_${BATCH_SIZE}.json"

        echo "Running EAGLE3 (default parameters) on $DATASET for batch size ${BATCH_SIZE}"
        python3 eval_benchmark.py $COMMON_ARGS \
            --speculative-algorithm EAGLE3 \
            --speculative-draft-model-path $EAGLE3_PATH \
            --result-filename "$EVAL_DATA_DIR/EAGLE3_${DATASET}_${BATCH_SIZE}.json"

        echo "Running SSSD on $DATASET for batch size: ${BATCH_SIZE}"
        python3 eval_benchmark.py $COMMON_ARGS \
            --speculative-algorithm SSSD \
            --speculative-draft-model-path $SSSD_DATASTORE_PATH \
            --speculative-adaptive \
            --result-filename "$EVAL_DATA_DIR/SSSD_${DATASET}_${BATCH_SIZE}.json"

        echo "Running NGRAM on $DATASET for batch size: ${BATCH_SIZE}"
        python3 eval_benchmark.py $COMMON_ARGS \
            --speculative-algorithm NGRAM \
            --result-filename "$EVAL_DATA_DIR/NGRAM_${DATASET}_${BATCH_SIZE}.json"

        echo "Running PLD on $DATASET for batch size: ${BATCH_SIZE}"
        python3 eval_benchmark.py $COMMON_ARGS \
            --speculative-algorithm PLD \
            --speculative-adaptive \
            --result-filename "$EVAL_DATA_DIR/PLD_${DATASET}_${BATCH_SIZE}.json"
    done
done

# After the loop, timing:
runtime=$((SECONDS - start))

hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$(((runtime % 3600) % 60))

printf "\nRuntime: %02d:%02d:%02d (hh:mm:ss)\n" "$hours" "$minutes" "$seconds"
