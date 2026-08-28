#!/bin/bash
# Run baseline (plain auto-regressive) and FAFO (lookahead) on each dataset
# (gsm8k, humaneval, mtbench) with the same model, then report
#     speedup = throughput(fafo) / throughput(baseline).
#
# Run this INSIDE a GPU allocation (an interactive session or your own job
# submission wrapper), e.g.:
#     bash scripts/speedup/run_speedup.sh 0 results/speedup Llama-3.1-8B-Instruct
#
# Args:
#   $1  cuda_devices     e.g. 0
#   $2  output_root      e.g. results/speedup  (relative to repo root)
#   $3  model            (optional) model dir name, default Llama-3.1-8B-Instruct
#   $4  kv_method        (optional) FAFO KV-cache method: stream-llm | quest
#                        (default stream-llm; "streamllm" is accepted too)
set -euo pipefail

cuda_devices=${1:?usage: run_speedup.sh <cuda_devices> <output_root> [model] [kv_method]}
out_root=${2:?usage: run_speedup.sh <cuda_devices> <output_root> [model] [kv_method]}
model=${3:-Llama-3.1-8B-Instruct}
kv_method=${4:-stream-llm}
# normalize "streamllm" -> "stream-llm" (config dir uses the hyphen)
[ "$kv_method" = "streamllm" ] && kv_method="stream-llm"
if [ "$kv_method" != "stream-llm" ] && [ "$kv_method" != "quest" ]; then
    echo "!! kv_method must be 'stream-llm' or 'quest' (got '$kv_method')" >&2
    exit 1
fi

# repo root = two levels up from this script
cd "$(dirname "$0")/../.."

datasets=(${DATASETS:-gsm8k humaneval mtbench})

# Resolve a config path case-insensitively (model dir casing is inconsistent in
# the repo, e.g. llama-3.1-8B-Instruct vs Llama-3.1-8B-Instruct).
resolve() {
    find "$1" -ipath "$2" 2>/dev/null | head -1
}

for ds in "${datasets[@]}"; do
    eval_cfg="config/eval_config/${ds}/${ds}.json"
    base_cfg=$(resolve "config/pipeline_config/baseline/${ds}" "*/${model}/default.json")
    # FAFO lookahead config for the chosen KV-cache method
    fafo_cfg=$(resolve "config/pipeline_config/fafo/${ds}" "*/${model}/${kv_method}/default.json")

    if [ -z "$base_cfg" ] || [ -z "$fafo_cfg" ]; then
        echo "!! ${ds}: missing config (baseline='${base_cfg}' fafo[${kv_method}]='${fafo_cfg}') — skipping" >&2
        continue
    fi

    echo "======================================================================"
    echo "[$ds] baseline  <- $base_cfg"
    echo "======================================================================"
    CUDA_VISIBLE_DEVICES=${cuda_devices} python pipeline/baseline/main.py \
        --exp_desc "${ds}_${model}_baseline" \
        --pipeline_config_dir "$base_cfg" \
        --eval_config_dir "$eval_cfg" \
        --output_folder_dir "${out_root}/${ds}/baseline"

    echo "======================================================================"
    echo "[$ds] fafo (${kv_method})  <- $fafo_cfg"
    echo "======================================================================"
    CUDA_VISIBLE_DEVICES=${cuda_devices} python pipeline/fafo/main.py \
        --exp_desc "${ds}_${model}_fafo_${kv_method}" \
        --pipeline_config_dir "$fafo_cfg" \
        --eval_config_dir "$eval_cfg" \
        --output_folder_dir "${out_root}/${ds}/fafo"
done

echo "======================================================================"
echo "SPEEDUP SUMMARY (fafo / baseline)"
echo "======================================================================"
python scripts/speedup/compute_speedup.py "$out_root" "${datasets[@]}"
