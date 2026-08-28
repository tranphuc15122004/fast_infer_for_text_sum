#!/usr/bin/env bash
# Downloads models, creates SSSD datastores, and runs benchmarks
set -e

export MODEL_DIR="${MODEL_DIR:-/storage}" # Where to save models (large files)

DATA_DIR="${DATA_DIR:-data_multilingual}" # Where to save benchmark results and other data
SAMPLE_SIZE="${SAMPLE_SIZE:-1}"  # Number of times to run each benchmark to take the median performance.
EVAL_LOGLEVEL="${EVAL_LOGLEVEL:-WARNING}"  # Set to DEBUG or INFO to see more detailed logs

MODEL_PATH=${MODEL_DIR}/datasets/huggingface/models/Qwen3-14B
EAGLE3_PATH=${MODEL_DIR}/datasets/huggingface/models/EAGLE3-Qwen3-14B
SSSD_DATASTORES_BASE_PATH=$MODEL_DIR/datasets/sssd_speculator

echo "Downloading models..."
models=(
  "Qwen/Qwen3-14B main ${MODEL_PATH}"
  "AngelSlim/Qwen3-14B_eagle3 main ${EAGLE3_PATH}"
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


echo "Creating SSSD datastores..."

python3 create_subdatastores.py --model $MODEL_PATH --index_file_path ${SSSD_DATASTORES_BASE_PATH}/sssd_qwen3-14b_italian.idx --datasets deepinfra_outputs/evol-instruct_italian_outputs.jsonl deepinfra_outputs/sharegpt_italian_outputs.jsonl --multi-incremental --multi-sizes 100k,1m,10m,100m &
python3 create_subdatastores.py --model $MODEL_PATH --index_file_path ${SSSD_DATASTORES_BASE_PATH}/sssd_qwen3-14b_indonesian.idx --datasets deepinfra_outputs/evol-instruct_indonesian_outputs.jsonl deepinfra_outputs/sharegpt_indonesian_outputs.jsonl --multi-incremental --multi-sizes 100k,1m,10m,100m &
python3 create_subdatastores.py --model $MODEL_PATH --index_file_path ${SSSD_DATASTORES_BASE_PATH}/sssd_qwen3-14b_japanese.idx --datasets deepinfra_outputs/evol-instruct_japanese_outputs.jsonl deepinfra_outputs/sharegpt_japanese_outputs.jsonl --multi-incremental --multi-sizes 100k,1m,10m,100m &
python3 create_subdatastores.py --model $MODEL_PATH --index_file_path ${SSSD_DATASTORES_BASE_PATH}/sssd_qwen3-14b_english.idx --datasets deepinfra_outputs/evol-instruct_english_outputs.jsonl deepinfra_outputs/sharegpt_english_outputs.jsonl deepinfra_outputs/wildchat_english_outputs.jsonl --multi-incremental --multi-sizes 100k,1m,10m,100m,1g &
# Full datastore, to check if other languages affect English
python3 create_subdatastores.py \
  --model "$MODEL_PATH" \
  --index_file_path "${SSSD_DATASTORES_BASE_PATH}/sssd_qwen3-14b_all.idx" \
  --datasets \
    deepinfra_outputs/evol-instruct_english_outputs.jsonl deepinfra_outputs/sharegpt_english_outputs.jsonl deepinfra_outputs/wildchat_english_outputs.jsonl \
    deepinfra_outputs/evol-instruct_italian_outputs.jsonl deepinfra_outputs/sharegpt_italian_outputs.jsonl \
    deepinfra_outputs/evol-instruct_indonesian_outputs.jsonl deepinfra_outputs/sharegpt_indonesian_outputs.jsonl \
    deepinfra_outputs/evol-instruct_japanese_outputs.jsonl deepinfra_outputs/sharegpt_japanese_outputs.jsonl \
  &

wait

# Build incremental language datastores starting from the largest English one

SIZES=(100k 1m 10m 100m)

# Languages and their JSONL suffixes
LANGS=(indonesian japanese italian)

for lang in "${LANGS[@]}"; do
  echo "=== Processing language: ${lang} ==="

  # Base "big" index that already has English + <lang>
  base_idx="${SSSD_DATASTORES_BASE_PATH}/sssd_qwen3-14b_english.1g.idx"

  # Datasets for this language
  evol_jsonl="deepinfra_outputs/evol-instruct_${lang}_outputs.jsonl"
  sgpt_jsonl="deepinfra_outputs/sharegpt_${lang}_outputs.jsonl"

  for size in "${SIZES[@]}"; do
    # Target incremental index: copy of base + extra <size> tokens
    target_idx="${SSSD_DATASTORES_BASE_PATH}/sssd_qwen3-14b_english_plus_${lang}.${size}.idx"

    echo "  -> Building incremental index ${target_idx} (+${size} tokens)"

    # If target already exists, log error and skip this dataset
    if [[ -f "${target_idx}" ]]; then
        echo "  [ERROR] Target index already exists: ${target_idx}. Not rebuilding." >&2
        continue
    fi

    # Also sanity-check that the base index exists
    if [[ ! -f "${base_idx}" ]]; then
        echo "  [ERROR] Base index not found: ${base_idx}. Cannot seed ${target_idx}." >&2
        continue
    fi

    # Seed target with base index; if copy fails, log error and skip
    if ! cp "${base_idx}" "${target_idx}"; then
        echo "  [ERROR] Failed to copy ${base_idx} to ${target_idx}. Skipping." >&2
        continue
    fi

    # Now extend this copy by exactly <size> tokens from the new datasets
    python3 create_subdatastores.py \
      --model "${MODEL_PATH}" \
      --index_file_path "${target_idx}" \
      --datasets "${evol_jsonl}" "${sgpt_jsonl}" \
      --extend-index \
      --extend-by "${size}" &
  done
done

# Wait for all background runs to finish
wait
echo "All incremental datastores built."

echo "Running benchmarks..."

start=$SECONDS

# Fixed settings
BATCH_SIZE=1
NUM_PROMPTS=80
OUTPUT_LEN=1024

# Languages we care about
LANGS=(english italian japanese indonesian)

# Map language -> dataset name
declare -A DATASET_BY_LANG=(
  [english]="mt-bench"
  [italian]="mt-bench-it"
  [japanese]="mt-bench-jp"
  [indonesian]="mt-bench-id"
)

# Map language -> base filename prefix
declare -A SSSD_PREFIX_BY_LANG=(
  [english]="sssd_qwen3-14b_english"
  [italian]="sssd_qwen3-14b_italian"
  [japanese]="sssd_qwen3-14b_japanese"
  [indonesian]="sssd_qwen3-14b_indonesian"
)

# Map language -> base filename prefix for *English+X* mixed datastores
declare -A SSSD_MIXED_PREFIX_BY_LANG=(
  [italian]="sssd_qwen3-14b_english_plus_italian"
  [japanese]="sssd_qwen3-14b_english_plus_japanese"
  [indonesian]="sssd_qwen3-14b_english_plus_indonesian"
)

# Incremental datastore sizes (in increasing order)
SSSD_SIZES=(100k 1m 10m 100m 1g)

for LANG in "${LANGS[@]}"; do
    DATASET="${DATASET_BY_LANG[$LANG]}"
    PREFIX="${SSSD_PREFIX_BY_LANG[$LANG]}"

    EVAL_DATA_DIR="$DATA_DIR/evaluation/$DATASET"
    mkdir -p "$EVAL_DATA_DIR"

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
        --temperature 0.7 \
        --top-p 0.8 \
        --top-k 20 \
        --keep-all-samples"

    echo "[$LANG] Running autoregressive baseline (dataset=$DATASET, bs=${BATCH_SIZE})"
    python3 eval_benchmark.py $COMMON_ARGS \
        --result-filename "$EVAL_DATA_DIR/Autoregressive_${DATASET}_bs${BATCH_SIZE}.json"

    echo "[$LANG] Running EAGLE3 (dataset=$DATASET, bs=${BATCH_SIZE})"
    python3 eval_benchmark.py $COMMON_ARGS \
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path "$EAGLE3_PATH" \
        --result-filename "$EVAL_DATA_DIR/EAGLE3_${DATASET}_bs${BATCH_SIZE}.json"

    echo "[$LANG] Running EAGLE3 (dataset=$DATASET, bs=${BATCH_SIZE})"
    python3 eval_benchmark.py $COMMON_ARGS \
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path "$EAGLE3_PATH" \
        --speculative-num-draft-tokens 32 \
        --speculative-num-steps 6 \
        --speculative-eagle-topk 5 \
        --result-filename "$EVAL_DATA_DIR/EAGLE3_${DATASET}_bs${BATCH_SIZE}_speclen32.json"

    echo "[$LANG] Running NGRAM (dataset=$DATASET, bs=${BATCH_SIZE})"
    python3 eval_benchmark.py $COMMON_ARGS \
        --speculative-algorithm NGRAM \
        --result-filename "$EVAL_DATA_DIR/NGRAM_${DATASET}_bs${BATCH_SIZE}.json"

    echo "[$LANG] Running SSSD with EMPTY datastore path (dataset=$DATASET, bs=${BATCH_SIZE})"
    python3 eval_benchmark.py $COMMON_ARGS \
        --speculative-algorithm SSSD \
        --speculative-draft-model-path "" \
        --speculative-adaptive \
        --result-filename "$EVAL_DATA_DIR/SSSD_${DATASET}_bs${BATCH_SIZE}_ds-none.json"

    # Monolingual SSSD datastores
    for SIZE in "${SSSD_SIZES[@]}"; do
        DATASTORE_PATH="${SSSD_DATASTORES_BASE_PATH}/${PREFIX}.${SIZE}.idx"

        if [[ ! -f "$DATASTORE_PATH" ]]; then
            echo "[$LANG] Warning: datastore '$DATASTORE_PATH' not found, skipping SIZE=$SIZE"
            continue
        fi

        echo "[$LANG] Running SSSD with datastore $SIZE (dataset=$DATASET, bs=${BATCH_SIZE})"
        python3 eval_benchmark.py $COMMON_ARGS \
            --speculative-algorithm SSSD \
            --speculative-draft-model-path "$DATASTORE_PATH" \
            --speculative-adaptive \
            --result-filename "$EVAL_DATA_DIR/SSSD_${DATASET}_bs${BATCH_SIZE}_ds-${SIZE}.json"
    done

    # English+X mixed SSSD datastores (for non-English LANGs)

    if [[ "$LANG" != "english" ]]; then
        # First, run other language with only English data
        echo "[$LANG] Running SSSD with english datastore (dataset=$DATASET, bs=${BATCH_SIZE})"
        python3 eval_benchmark.py $COMMON_ARGS \
            --speculative-algorithm SSSD \
            --speculative-draft-model-path "${SSSD_DATASTORES_BASE_PATH}/${SSSD_PREFIX_BY_LANG[english]}.1g.idx" \
            --speculative-adaptive \
            --result-filename "$EVAL_DATA_DIR/SSSD_${DATASET}_bs${BATCH_SIZE}_en-plus-${LANG}-ds-none.json"
    else  # If English, run instead the experiments were the full data from all languages is evaluated on English
        echo "[$LANG] Running SSSD with full multilingual datastore (dataset=$DATASET, bs=${BATCH_SIZE})"
        python3 eval_benchmark.py $COMMON_ARGS \
            --speculative-algorithm SSSD \
            --speculative-draft-model-path "${SSSD_DATASTORES_BASE_PATH}/sssd_qwen3-14b_all.idx" \
            --speculative-adaptive \
            --result-filename "$EVAL_DATA_DIR/SSSD_${DATASET}_bs${BATCH_SIZE}_all_languages.json"
    fi

    MIXED_PREFIX="${SSSD_MIXED_PREFIX_BY_LANG[$LANG]:-}"

    if [[ -n "$MIXED_PREFIX" ]]; then
        for SIZE in "${SSSD_SIZES[@]}"; do
            MIXED_DATASTORE_PATH="${SSSD_DATASTORES_BASE_PATH}/${MIXED_PREFIX}.${SIZE}.idx"

            if [[ ! -f "$MIXED_DATASTORE_PATH" ]]; then
                echo "[$LANG] (mixed) Warning: datastore '$MIXED_DATASTORE_PATH' not found, skipping SIZE=$SIZE"
                continue
            fi

            echo "[$LANG] Running SSSD (English+${LANG}) with datastore $SIZE (dataset=$DATASET, bs=${BATCH_SIZE})"
            python3 eval_benchmark.py $COMMON_ARGS \
                --speculative-algorithm SSSD \
                --speculative-draft-model-path "$MIXED_DATASTORE_PATH" \
                --speculative-adaptive \
                --result-filename "$EVAL_DATA_DIR/SSSD_${DATASET}_bs${BATCH_SIZE}_en-plus-${LANG}-ds-${SIZE}.json"
        done
    fi
done

# Timing
runtime=$((SECONDS - start))
hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 3600 % 60))

printf "\nRuntime: %02d:%02d:%02d (hh:mm:ss)\n" "$hours" "$minutes" "$seconds"
