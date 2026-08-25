OUTPUT_DIR="./local/outputs/qps"
TIMEOUT=40
NUM_SAMPLES=32
mkdir -p $OUTPUT_DIR

SIZE=${1:-"70B"}
CATEGORY=summarization

if [ $SIZE == "70B" ]; then
    MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct
elif [ $SIZE == "405B" ]; then
    MODEL_NAME=neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8
else
    echo "Invalid model size"
    exit 1
fi

# warmup
echo "Warm up server"
python3 eval/qps_client.py \
    --model $MODEL_NAME \
    --qps 0.2 \
    --category $CATEGORY \
    --max-tokens 8 \
    --timeout $(( $TIMEOUT + 5 )) \
    --num-samples 4

echo "Start real profiling"
echo "" > $OUTPUT_DIR/${SIZE}_${2}_ttft_${CATEGORY}.txt

for qps in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2; do
    echo "Sleep for 10 seconds"
    sleep 10
    python3 eval/qps_client.py \
        --model $MODEL_NAME \
        --qps $qps \
        --category $CATEGORY \
        --timeout $(( $TIMEOUT + 5 )) \
        --num-samples $NUM_SAMPLES >> $OUTPUT_DIR/${SIZE}_${2}_ttft_${CATEGORY}.txt

    if cat $OUTPUT_DIR/${SIZE}_${2}_ttft_${CATEGORY}.txt | grep -q "Found timeout in queries"; then
        exit 1
    fi
done
