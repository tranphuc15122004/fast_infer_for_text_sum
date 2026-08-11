cd ./eval/long_bench

for exp in 0.1 0.3 0.5 0.7 0.9; do
    python pred_rag_llama.py \
        --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
        --percentage ${exp} \
        --exp rag_new_70B_${exp}

    python eval.py --exp rag_new_70B_${exp}
done