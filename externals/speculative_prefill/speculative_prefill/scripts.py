import os

from speculative_prefill import enable_prefill_spec

spec_model = os.environ.get(
    "ENABLE_SP", None)

if spec_model:
    enable_prefill_spec(
        spec_model=spec_model, 
        spec_config_path='./configs/config.yaml'
    )

from vllm.scripts import main

if __name__ == "__main__":    
    main()
