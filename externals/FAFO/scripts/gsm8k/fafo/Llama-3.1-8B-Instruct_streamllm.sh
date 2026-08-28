cuda_devices=$1
output_dir_root=$2

task="gsm8k"
dataset="gsm8k"
model="Llama-3.1-8B-Instruct"
kv_compression_method="stream-llm"
method="fafo"

CUDA_VISIBLE_DEVICES=${cuda_devices} python pipeline/fafo/main.py \
    --exp_desc ${task}_${dataset}_${model}_${method}_${kv_compression_method} \
    --pipeline_config_dir config/pipeline_config/${method}/${task}/${model}/${kv_compression_method}/default.json \
    --eval_config_dir config/eval_config/${task}/${dataset}.json \
    --output_folder_dir ${output_dir_root}/${task}/${method}/${model}/${dataset}/${kv_compression_method}
