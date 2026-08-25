# Append ENABLE_SP and SPEC_CONFIG_PATH if nececssary
python ruler_server.py \
    --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.6