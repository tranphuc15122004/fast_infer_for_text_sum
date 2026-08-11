import os

from speculative_prefill import enable_prefill_spec

spec_model = os.environ.get(
    "ENABLE_SP", None)

if spec_model:
    enable_prefill_spec(
        spec_model=spec_model, 
        spec_config_path='./configs/config.yaml'
    )

from lm_eval.__main__ import cli_evaluate

"""
python -m eval.lm_eval \
    --model vllm \
    --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,dtype=auto,gpu_memory_utilization=0.8,enable_chunked_prefill=False \
    --tasks mmlu_generative \
    --gen_kwargs do_sample=False,max_gen_toks=2 \
    --limit 100 \
    --batch_size 4
"""

if __name__ == "__main__":
    cli_evaluate()
