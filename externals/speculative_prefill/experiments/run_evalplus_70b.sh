mbpp_output_dir="./local/outputs/mbpp"
humaneval_output_dir="./local/outputs/humaneval"
mkdir -p $mbpp_output_dir
mkdir -p $humaneval_output_dir

{
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m eval.evalplus \
        --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
        --dataset humaneval \
        --backend vllm \
        --greedy \
        --enable-chunked-prefill False \
        --tp 4 \
        --root ./local/evalplus_result/humaneval/baseline > $humaneval_output_dir/baseline.txt
} &

{
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m eval.evalplus \
        --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --dataset mbpp \
        --backend vllm \
        --greedy \
        --enable_chunked_prefill False \
        --tp 4 \
        --root ./local/evalplus_result/mbpp/baseline > $mbpp_output_dir/baseline.txt
} &

wait

for exp in "p3" "p5" "p7" "p9" "p3_full" "p5_full" "p7_full" "p9_full" "p3_full_lah4" "p5_full_lah4" "p7_full_lah4" "p9_full_lah4"; do
    {
        CUDA_VISIBLE_DEVICES=0,1,2,3 ENABLE_SP=meta-llama/Meta-Llama-3.1-8B-Instruct SPEC_CONFIG_PATH=./configs/config_${exp}.yaml python -m eval.evalplus \
            --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
            --dataset humaneval \
            --backend vllm \
            --greedy \
            --enable_chunked_prefill False \
            --tp 4 \
            --root ./local/evalplus_result/humaneval/${exp} > $humaneval_output_dir/$exp.txt
    } & 
    
    {
        CUDA_VISIBLE_DEVICES=4,5,6,7 ENABLE_SP=meta-llama/Meta-Llama-3.1-8B-Instruct SPEC_CONFIG_PATH=./configs/config_${exp}.yaml python -m eval.evalplus \
            --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
            --dataset mbpp \
            --backend vllm \
            --greedy \
            --enable_chunked_prefill False \
            --tp 4 \
            --root ./local/evalplus_result/mbpp/${exp} > $mbpp_output_dir/$exp.txt
    } & 

    wait

done

