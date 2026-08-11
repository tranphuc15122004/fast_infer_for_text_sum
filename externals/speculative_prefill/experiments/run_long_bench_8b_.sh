cd ./eval/long_bench

python pred_vllm.py \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --tensor-parallel-size 8 \
    --exp baseline_8b

python eval.py --exp baseline_8b

for exp in "p1_full_lah8" "p3_full_lah8" "p5_full_lah8" "p7_full_lah8" "p9_full_lah8"; do
    SPEC_CONFIG_PATH=../../configs/config_${exp}.yaml python pred_vllm.py \
        --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --spec-model meta-llama/Llama-3.2-3B-Instruct \
        --spec-prefill \
        --tensor-parallel-size 8 \
        --exp spec_${exp}_8b3b

    python eval.py --exp spec_${exp}_8b3b
done
