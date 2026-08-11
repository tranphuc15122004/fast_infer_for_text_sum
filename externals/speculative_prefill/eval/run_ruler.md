
First we need to run the ruler server: 
``` bash
SPEC_CONFIG_PATH=../configs/config_p1_full_lah8.yaml ENABLE_SP=meta-llama/Meta-Llama-3.1-8B-Instruct bash ruler_server.sh
```

Then we run the main client in RULER repo: 
``` bash
bash run.sh llama-70b synthetic
```