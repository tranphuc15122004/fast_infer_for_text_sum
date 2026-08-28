cuda_devices=$1
output_dir_root=$2

task="gsm8k"
dataset="gsm8k"
model="Llama-3.1-8B-Instruct"
method="baseline"


CUDA_VISIBLE_DEVICES=${cuda_devices} python pipeline/baseline/main.py \
    --exp_desc ${task}_${dataset}_${model}_baseline \
    --pipeline_config_dir config/pipeline_config/${method}/${task}/${model}/default.json \
    --eval_config_dir config/eval_config/${task}/${dataset}.json \
    --output_folder_dir ${output_dir_root}/${task}/baseline/${model}/${dataset}
