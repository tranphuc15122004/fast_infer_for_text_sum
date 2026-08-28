#!/usr/bin/env bash
# Downloads models, creates SSSD datastores, and runs benchmarks
set -e

export MODEL_DIR="${MODEL_DIR:-/storage}" # Where to save models (large files)

DATA_DIR="${DATA_DIR:-data}" # Where to save benchmark results and other data


echo "Getting user machine specs..."
python3 speculative_bench_scripts/collect_env.py --out-dir "$DATA_DIR"

echo "Downloading models..."
./speculative_bench_scripts/download_models_8B.sh

echo "Converting EAGLE heads to BF16..."
EAGLE2_LLAMA_3_1_8B_PATH=${MODEL_DIR}/datasets/huggingface/models/jamesliu1/sglang-EAGLE-Llama-3.1-Instruct-8B
EAGLE3_LLAMA_3_1_8B_PATH=${MODEL_DIR}/datasets/huggingface/models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B
python3 speculative_bench_scripts/change_model_type.py --model $EAGLE2_LLAMA_3_1_8B_PATH
python3 speculative_bench_scripts/change_model_type.py --model $EAGLE3_LLAMA_3_1_8B_PATH


echo "Creating SSSD datastore..."
LLAMA3_1_8B_PATH=$MODEL_DIR/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/
MAGPIE_DATASTORE_IDX_PATH=$MODEL_DIR/datasets/sssd_speculator/sssd-llama-3.1-8B.idx

python3 sssd_speculator/datastore_creation/create_datastore.py --model $LLAMA3_1_8B_PATH --index_file_path $MAGPIE_DATASTORE_IDX_PATH --datasets hitz-magpie-llama3.1-8b magpie-llama31-pro sharegpt-de synthia-german python-stack --stack-token $STACK_TOKEN

# echo "Creating PIA cache..."
# PIA_CACHE_PATH=$MODEL_DIR/datasets/sssd_speculator/pia-llama-3.1-8B.json
# python3 sssd_speculator/evaluation/create_pia_cache.py --model $LLAMA3_1_8B_PATH --cache_path $PIA_CACHE_PATH --datasets sharegpt-de python-stack hitz-magpie-llama3.1-8b --stack-token $STACK_TOKEN
