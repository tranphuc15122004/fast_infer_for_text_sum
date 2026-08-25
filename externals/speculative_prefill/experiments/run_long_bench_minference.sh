cd ./eval/long_bench

python pred_vllm.py \
    --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    --minference \
    --tensor-parallel-size 8 \
    --exp minference

python eval.py --exp minference
