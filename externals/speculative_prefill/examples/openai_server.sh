API_KEY=${2:-local_server}
PORT=${3:-8888}

echo "Using port ${PORT} and api key ${API_KEY}"

if [ "$1" = "native" ]; then
    echo "Starting native vllm serving"
    vllm serve \
        'meta-llama/Meta-Llama-3.1-8B-Instruct' \
        --dtype auto \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.8 \
        --enable-chunked-prefill=False \
        --enforce-eager \
        --api-key=$API_KEY \
        --port=$PORT

elif [ "$1" = "spec_prefill" ]; then
    echo "Starting vllm serving with speculative prefill" 
    python -m speculative_prefill.scripts serve \
        'meta-llama/Meta-Llama-3.1-8B-Instruct' \
        --dtype auto \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.8 \
        --enable-chunked-prefill=False \
        --enforce-eager \
        --api-key=$API_KEY \
        --port=$PORT
else
    echo "Invalid serving type..."
fi