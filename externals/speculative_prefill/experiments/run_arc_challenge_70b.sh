output_dir="./local/outputs/arc_challenge"
mkdir -p $output_dir

python -m eval.lm_eval \
    --model vllm \
    --model_args pretrained=meta-llama/Meta-Llama-3.1-70B-Instruct,enforce_eager=True,max_model_len=16384,tensor_parallel_size=8,enable_chunked_prefill=False \
    --tasks arc_challenge_llama_instruct \
    --apply_chat_template \
    --gen_kwargs temperature=0 \
    --batch_size 128 > $output_dir/baseline.txt

for exp in "p3" "p5" "p7" "p9" "p3_full" "p5_full" "p7_full" "p9_full" "p3_full_lah4" "p5_full_lah4" "p7_full_lah4" "p9_full_lah4"; do
    SPEC_CONFIG_PATH=./configs/config_${exp}.yaml ENABLE_SP=meta-llama/Meta-Llama-3.1-8B-Instruct python -m eval.lm_eval \
        --model vllm \
        --model_args pretrained=meta-llama/Meta-Llama-3.1-70B-Instruct,enforce_eager=True,max_model_len=16384,tensor_parallel_size=8,enable_chunked_prefill=False \
        --tasks arc_challenge_llama_instruct \
        --apply_chat_template \
        --gen_kwargs temperature=0 \
        --batch_size 128 > $output_dir/${exp}.txt
done
