# NOTE: This file is a collection of bash snippets.
# It is NOT meant to be executed directly.

## LLaMA 3.1 ##

# AUTOREGRESSIVE
python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --num-prompts 80 \
    --disable-ignore-eos \
    --apply-chat-template \
    --enable-metrics \
    --dataset-name mt-bench \
    --sharegpt-output-len 1024

# EAGLE2
python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path /storage/datasets/huggingface/models/jamesliu1/sglang-EAGLE-Llama-3.1-Instruct-8B \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --speculative-num-draft-tokens 8 \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 4 \
    --num-prompts 80 \
    --disable-ignore-eos \
    --apply-chat-template \
    --enable-metrics \
    --dataset-name mt-bench \
    --sharegpt-output-len 1024

# EAGLE 3
python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path /storage/datasets/huggingface/models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --speculative-num-draft-tokens 8 \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 4 \
    --num-prompts 80 \
    --disable-ignore-eos \
    --apply-chat-template \
    --enable-metrics \
    --dataset-name mt-bench \
    --sharegpt-output-len 1024

# SSSD with manual parameters
python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  \
    --speculative-algorithm SSSD \
    --speculative-draft-model-path /storage/datasets/sssd_speculator/sssd-llama-3.1-8B.idx \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --speculative-num-draft-tokens 8 \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 5 \
    --num-prompts 80 \
    --disable-ignore-eos \
    --apply-chat-template \
    --enable-metrics \
    --dataset-name mt-bench \
    --sharegpt-output-len 1024

# SSSD adaptive
python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  \
    --speculative-algorithm SSSD \
    --speculative-draft-model-path /storage/datasets/sssd_speculator/sssd-llama-3.1-8B.idx \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --num-prompts 80 \
    --disable-ignore-eos \
    --apply-chat-template \
    --enable-metrics \
    --dataset-name mt-bench \
    --sharegpt-output-len 1024 \
    --speculative-adaptive

# PIA
python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  \
    --speculative-algorithm PIA \
    --speculative-draft-model-path /storage/datasets/sssd_speculator/pia-llama-3.1-8B.json \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --num-prompts 80 \
    --disable-ignore-eos \
    --apply-chat-template \
    --enable-metrics \
    --dataset-name mt-bench \
    --sharegpt-output-len 1024 \
    --speculative-adaptive

# REST
python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  \
    --speculative-algorithm REST \
    --speculative-draft-model-path /storage/datasets/sssd_speculator/sssd-llama-3.1-8B.idx \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --num-prompts 80 \
    --disable-ignore-eos \
    --apply-chat-template \
    --enable-metrics \
    --dataset-name mt-bench \
    --sharegpt-output-len 1024 \
    --speculative-adaptive

# PLD
python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  \
    --speculative-algorithm PLD \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --num-prompts 80 \
    --disable-ignore-eos \
    --apply-chat-template \
    --enable-metrics \
    --dataset-name mt-bench \
    --sharegpt-output-len 1024 \
    --speculative-adaptive

#NGRAM
python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  \
    --speculative-algorithm NGRAM \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --num-prompts 80 \
    --disable-ignore-eos \
    --apply-chat-template \
    --enable-metrics \
    --dataset-name mt-bench \
    --sharegpt-output-len 1024
