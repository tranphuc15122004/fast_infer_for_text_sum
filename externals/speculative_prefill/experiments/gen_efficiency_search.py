output_dir = "./experiments/run_efficiency_search.sh"

TOTAL_TOKEN = 128 * 4096

BASELINE_8B_COMMAND = """
python -m speculative_prefill.vllm_benchmarks.latency \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --enforce-eager \
    --enable-chunked-prefill False \
    --tensor-parallel-size 8 \
    --max_model_len 32768 \
    --input-len {seq_len} \
    --output-len 1 \
    --batch-size {bs} \
    --num-iters-warmup 4 \
    --num-iters 16 > $output_dir/baseline_8B_bs{bs}_sl{seq_len}.txt
"""

BASELINE_70B_COMMAND = """
python -m speculative_prefill.vllm_benchmarks.latency \
    --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    --enforce-eager \
    --enable-chunked-prefill False \
    --tensor-parallel-size 8 \
    --max_model_len 32768 \
    --input-len {seq_len} \
    --output-len 1 \
    --batch-size {bs} \
    --num-iters-warmup 4 \
    --num-iters 16 > $output_dir/baseline_70B_bs{bs}_sl{seq_len}.txt
"""

BASELINE_405B_COMMAND = """
python -m speculative_prefill.vllm_benchmarks.latency \
    --model "neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8" \
    --enforce-eager \
    --enable-chunked-prefill False \
    --tensor-parallel-size 8 \
    --max_model_len 32768 \
    --input-len {seq_len} \
    --output-len 1 \
    --batch-size {bs} \
    --num-iters-warmup 4 \
    --num-iters 16 > $output_dir/baseline_405B_bs{bs}_sl{seq_len}.txt
"""

SP_70B_COMMAND = """
SPEC_CONFIG_PATH=./configs/config_{sp}.yaml python -m speculative_prefill.vllm_benchmarks.latency \
    --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    --spec-prefill \
    --spec-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --enforce-eager \
    --enable-chunked-prefill False \
    --tensor-parallel-size 8 \
    --max_model_len 32768 \
    --input-len {seq_len} \
    --output-len 1 \
    --batch-size {bs} \
    --num-iters-warmup 4 \
    --num-iters 16 > $output_dir/spec_70B8B_{sp}_bs{bs}_sl{seq_len}.txt
"""

SP_405B_COMMAND = """
SPEC_CONFIG_PATH=./configs/config_{sp}.yaml python -m speculative_prefill.vllm_benchmarks.latency \
    --model "neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8" \
    --spec-prefill \
    --spec-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --enforce-eager \
    --enable-chunked-prefill False \
    --tensor-parallel-size 8 \
    --max_model_len 32768 \
    --input-len {seq_len} \
    --output-len 1 \
    --batch-size {bs} \
    --num-iters-warmup 4 \
    --num-iters 16 > $output_dir/spec_405B8B_{sp}_bs{bs}_sl{seq_len}.txt
"""

with open(output_dir, 'w') as f:
    f.write('output_dir="./local/outputs/efficiency/search"\n')
    f.write('mkdir -p $output_dir\n')

    for bs in [128, 64, 32, 16]:
        seq_len = TOTAL_TOKEN // bs
        
        f.write(BASELINE_8B_COMMAND.format(
            seq_len=seq_len, 
            bs=bs
        ))
        
        f.write(BASELINE_70B_COMMAND.format(
            seq_len=seq_len, 
            bs=bs
        ))

        f.write(BASELINE_405B_COMMAND.format(
            seq_len=seq_len, 
            bs=bs
        ))

        for sp in ["p1", "p3", "p5", "p7", 
            "p1_full_lah8", "p3_full_lah8", "p5_full_lah8", "p7_full_lah8"]:
            f.write(SP_70B_COMMAND.format(
                sp=sp, 
                seq_len=seq_len, 
                bs=bs
            ))

            f.write(SP_405B_COMMAND.format(
                sp=sp, 
                seq_len=seq_len, 
                bs=bs
            ))
