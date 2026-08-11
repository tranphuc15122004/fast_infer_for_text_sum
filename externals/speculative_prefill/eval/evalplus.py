import os

from speculative_prefill import enable_prefill_spec

spec_model = os.environ.get(
    "ENABLE_SP", None)

if spec_model:
    enable_prefill_spec(
        spec_model=spec_model, 
        spec_config_path='./configs/config.yaml'
    )

from evalplus.evaluate import main

"""
python -m eval.evalplus \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --dataset humaneval \
    --backend vllm \
    --greedy \
    --root ./local/evalplus_result
"""

if __name__ == "__main__":
    main()
