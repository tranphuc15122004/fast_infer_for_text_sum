# bash eval/qps_server.sh
# SPEC_CONFIG_PATH=./configs/config_p3_full.yaml ENABLE_SP="meta-llama/Meta-Llama-3.1-8B-Instruct" bash eval/qps_server.sh

SIZE=${1:-"70B"}
API_KEY=${2:-local_server}
PORT=${3:-8888}

if [ $SIZE == "70B" ]; then
    MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct
elif [ $SIZE == "405B" ]; then
    echo "When using 405B model, it is recommended to run on 8xH200."
    MODEL_NAME=neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8
else
    echo "Invalid model size"
    exit 1
fi

echo "Starting vllm server using port ${PORT} and api key ${API_KEY}"

fuser -n tcp ${PORT}

python -m speculative_prefill.scripts serve \
    ${MODEL_NAME} \
    --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dtype auto \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.8 \
    --enable-chunked-prefill=False \
    --tensor-parallel-size 8 \
    --enforce-eager \
    --max-num-seqs 256 \
    --api-key=$API_KEY \
    --port=$PORT
