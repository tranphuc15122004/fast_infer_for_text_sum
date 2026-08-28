#!/usr/bin/env bash
# Downloads models, creates SSSD datastores, and runs benchmarks
set -e

export MODEL_DIR="${MODEL_DIR:-/storage}" # Where to save models (large files)

DATA_DIR="${DATA_DIR:-data}" # Where to save benchmark results and other data


echo "Getting user machine specs..."
python3 speculative_bench_scripts/collect_env.py --out-dir "$DATA_DIR"

echo "Downloading models..."
./speculative_bench_scripts/download_models_70b.sh

echo "Converting EAGLE heads to BF16..."
EAGLE3_LLAMA_3_3_70B_PATH=${MODEL_DIR}/datasets/huggingface/models/sglang-EAGLE3-LLaMA3.3-Instruct-70B
python3 speculative_bench_scripts/change_model_type.py --model $EAGLE3_LLAMA_3_3_70B_PATH


echo "Creating SSSD datastore..."
LLAMA3_3_70B_PATH=$MODEL_DIR/datasets/huggingface/models/Llama-3.3-70B-Instruct
MAGPIE_DATASTORE_IDX_PATH=$MODEL_DIR/datasets/sssd_speculator/sssd-llama-3.3-70B.idx

python3 sssd_speculator/datastore_creation/create_datastore.py --model $LLAMA3_3_70B_PATH --index_file_path $MAGPIE_DATASTORE_IDX_PATH --datasets magpie-llama33-pro-1M magpie-llama33-reason llama_future_code oh-dcft llama_wildchat llama_ultrainteract --stack-token $STACK_TOKEN
