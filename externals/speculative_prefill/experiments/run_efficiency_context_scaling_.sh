OUTPUT_DIR=./local/outputs/efficiency/scaling
TOTAL_TOKEN=$((128 * 4096))

echo Running a total of $TOTAL_TOKEN tokens. 

echo Start benchmarking the baseline...

for bs in 128 64 32 16 8 4; do
    sl=$(($TOTAL_TOKEN / bs))
    python -m speculative_prefill.vllm_benchmarks.latency \
        --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
        --gpu_memory_utilization 0.8 \
        --enforce-eager \
        --enable-chunked-prefill False \
        --tensor-parallel-size 8 \
        --max_model_len 131072 \
        --input-len $sl \
        --output-len 1 \
        --batch-size $bs \
        --num-iters-warmup 2 \
        --num-iters 8 > $OUTPUT_DIR/baseline_bs${bs}_sl${sl}.txt
done

echo Start benchmarking minference...

for bs in 128 64 32 16 8 4; do
    sl=$(($TOTAL_TOKEN / bs))
    python eval/minference_latency.py \
        --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
        --gpu_memory_utilization 0.8 \
        --enforce-eager \
        --enable-chunked-prefill False \
        --tensor-parallel-size 8 \
        --max_model_len 131072 \
        --input-len $sl \
        --output-len 1 \
        --batch-size $bs \
        --num-iters-warmup 2 \
        --num-iters 8 > $OUTPUT_DIR/minference_bs${bs}_sl${sl}.txt
done

echo Start benchmarking spec prefill...

for sp in p1_full p3_full p5_full p7_full; do
    for bs in 128 64 32 16 8 4; do
        sl=$(($TOTAL_TOKEN / bs))
        SPEC_CONFIG_PATH=./configs/config_${sp}.yaml python -m speculative_prefill.vllm_benchmarks.latency \
            --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
            --spec-prefill \
            --spec-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
            --gpu_memory_utilization 0.8 \
            --enforce-eager \
            --enable-chunked-prefill False \
            --tensor-parallel-size 8 \
            --max_model_len 131072 \
            --input-len $sl \
            --output-len 1 \
            --batch-size $bs \
            --num-iters-warmup 2 \
            --num-iters 8 > $OUTPUT_DIR/spec_${sp}_bs${bs}_sl${sl}.txt
    done
done
