#!/usr/bin/env bash
set -e

# # AUTOREGRESSIVE
# python3 -m sglang.launch_server \
#     --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  \
#     --cuda-graph-max-bs 64 \
#     --max-running-requests 64 \
#     --disaggregation-mode prefill \
#     --disaggregation-transfer-backend nixl &


# python3 -m sglang.launch_server \
#     --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  \
#     --cuda-graph-max-bs 2 \
#     --max-running-requests 2 \
#     --disaggregation-mode decode \
#     --port 30001 \
#     --base-gpu-id 1 \
#     --disaggregation-transfer-backend nixl &


# EAGLE3

python3 -m sglang.launch_server \
    --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  \
    --cuda-graph-max-bs 64 \
    --max-running-requests 64 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path /storage/datasets/huggingface/models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl &


python3 -m sglang.launch_server \
    --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  \
    --cuda-graph-max-bs 2 \
    --max-running-requests 2 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path /storage/datasets/huggingface/models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
    --disaggregation-mode decode \
    --port 30001 \
    --base-gpu-id 1 \
    --disaggregation-transfer-backend nixl &


# # SSSD

# python3 -m sglang.launch_server \
#     --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  \
#     --cuda-graph-max-bs 64 \
#     --max-running-requests 64 \
#     --disaggregation-mode prefill \
#     --chunked-prefill-size 8192 \
#     --disaggregation-transfer-backend nixl &


# python3 -m sglang.launch_server \
#     --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  \
#     --cuda-graph-max-bs 2 \
#     --max-running-requests 2 \
#     --speculative-algorithm SSSD \
#     --speculative-draft-model-path /storage/datasets/sssd_speculator/sssd-llama-3.1-8B.idx \
#     --speculative-adaptive \
#     --disaggregation-mode decode \
#     --port 30001 \
#     --base-gpu-id 1 \
#     --disaggregation-transfer-backend nixl &


# TO RUN FOR ALL

python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:30000 \
    --decode http://127.0.0.1:30001 \
    --host 0.0.0.0 \
    --port 8000

# Done when it says "Starting server on 0.0.0.0:8000"

# python3 -m sglang.bench_serving \
#     --model /storage/datasets/huggingface/models/Llama-3.3-70B-Instruct \
#     --base-url http://127.0.0.1:8000 \
#     --dataset-name "sharegpt" \
#     --num-prompts 10 \
#     --sharegpt-output-len 100 \
#     --disable-ignore-eos \
#     --max-concurrency 4 \
#     --pd-separated
