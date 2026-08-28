# FAFO: Lossy KV Cache Compression for Lossless Inference Acceleration via Draftless Fumble Decoding

**Hoang Anh Duy Le**\*¹  **Shaochen (Henry) Zhong**\*¹  **Yifan Lu**¹  **Yingtong Dou**²  **Jiayi Yuan**¹  **Yu-Neng Chuang**¹  **Xiran Fan**²  **Guanchu Wang**¹  **Yuzhong Chen**²  **Xia (Ben) Hu**¹

¹ Department of Computer Science, Rice University   ² Visa Research
\*Equal contribution.

*Published at the 43rd International Conference on Machine Learning (**ICML 2026**), Seoul, South Korea. PMLR 306, 2026.*

## Overview

![FAFO pipeline](figures/fafo_pipeline.png)

FAFO ("**F**umble **A**round and **F**ind **O**ut") accelerates LLM decoding by using **lossy** KV-cache compression as a *means* to deliver **lossless** generation. Within a single forward pass it runs two branches: **Fumble Around** generates many n-gram "guesses" cheaply on a *compressed* KV cache, while **Find Out** verifies previously-cached guesses against the *full* KV cache — accepting only tokens that match greedy decoding. Because drafting and verification happen in parallel in one pass, FAFO needs only **one model and one set of KV cache**, and it delivers a **1.20–2.71× latency speedup** over vanilla decoding while preserving the original output distribution.

## Abstract

Lossy KV cache compression is a well-explored subfield of machine learning efficiency, with improved latency being one of its major gains. However, lossy compression techniques can fumble from time to time, exhibiting various, and often catastrophic, failure patterns that are not only difficult to resolve but sometimes even hard to identify, making direct deployment of models with compressed KV cache a risky endeavor. In this work, we explore a way to preserve lossless generation quality while still benefiting from the acceleration provided by KV cache compression. Specifically, we draw inspiration from the n-gram candidate pool decoding paradigm where we purposely allow the model to **Fumble Around** with compressed KV cache to generate multiple lossy "n-gram guesses", while in parallel **Find Out** via lossless verification in the same forward pass. From a conceptual standpoint, our proposed framework is compatible with all typical static or dynamic KV cache compression methods from the token dropping realm, thus opening up a new avenue for the stagnant n-gram decoding paradigm. Practically, we show that this framework presents many useful traits that similar draftless baselines (e.g., Self-Speculative Decoding) cannot achieve, such as requiring only one set of KV cache and being far less sensitive to model, task, and input-length scenarios. Our comprehensive empirical results show FAFO provides 1.20–2.71× latency speedup over the original model, while consistently outperforming other lossless + draftless solutions.

## Installation

**Requirements:** Python 3.9.20, CUDA 12.1, a recent NVIDIA GPU (experiments in the paper use a single A100-80GB; H100/H200 also work).

1. Create a virtual environment and install the matching PyTorch build:
   ```bash
   cd FAFO_dev/
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
       --index-url https://download.pytorch.org/whl/cu121
   ```

2. Install the remaining pinned packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install **FastChat** from source (conversation templates and MT-Bench loading). Clone it *outside* `FAFO_dev/`:
   ```bash
   cd ../                       # anywhere except within FAFO_dev/
   git clone https://github.com/lm-sys/FastChat.git
   cd FastChat/
   pip3 install -e ".[model_worker,webui]"
   ```

4. Install **Human-Eval** from source (HumanEval benchmark):
   ```bash
   git clone https://github.com/openai/human-eval
   pip install -e human-eval
   ```

### HuggingFace access token

The pipeline downloads gated models (e.g. `meta-llama/Llama-3.1-8B-Instruct`) from the HuggingFace Hub.

1. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (a "Read" token is enough).
2. Set it in [`config/access_tokens.py`](config/access_tokens.py):
   ```python
   hf_access_token = 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
   ```
   or export `HF_TOKEN` in your environment.
3. Accept the license for each gated model on its HuggingFace model page.

## Quick Start

Run FAFO on a single benchmark with one GPU. FP16, batch size 1:

```bash
python pipeline/fafo/main.py \
    --exp_desc          "fafo_gsm8k_llama31_stream" \
    --pipeline_config_dir config/pipeline_config/fafo/gsm8k/Llama-3.1-8B-Instruct/stream-llm/default.json \
    --eval_config_dir     config/eval_config/gsm8k/gsm8k.json \
    --output_folder_dir   experiment-results/quickstart/
```

This runs Llama-3.1-8B-Instruct with FAFO-Stream (StreamingLLM compression backend) on GSM8K, writing the wall-clock speedup, average acceptance length (τ), and run config under `experiment-results/quickstart/`.

### Launcher scripts

Each `(dataset, model, KV-method)` combination ships a single-GPU launcher. Arguments are `<gpu_id> <output_dir>`.

**FAFO:**
```bash
# FAFO-Stream (StreamingLLM backend)
bash scripts/gsm8k/fafo/Llama-3.1-8B-Instruct_streamllm.sh 0 experiment-results/
# FAFO-Quest (Quest backend)
bash scripts/gsm8k/fafo/Llama-3.1-8B-Instruct_quest.sh     0 experiment-results/
```

**Baseline** — standard auto-regressive decoding, used as the speed reference:
```bash
bash scripts/gsm8k/baseline/Llama-3.1-8B-Instruct.sh 0 experiment-results/
```

### What ships

| axis | options |
|---|---|
| **datasets** | `gsm8k`, `humaneval`, `mtbench` |
| **models** | `Llama-3.1-8B-Instruct`, `llama-2-7b-chat-hf` |
| **KV-cache backends** | `stream-llm` (StreamingLLM — **FAFO-Stream**), `quest` (Quest — **FAFO-Quest**) |

Configs live under [`config/pipeline_config/fafo/<dataset>/<model>/<kv-method>/`](config/pipeline_config/fafo/) and eval configs under [`config/eval_config/<dataset>/`](config/eval_config/). 
## Baseline vs FAFO speedup

To measure `speedup = throughput(FAFO) / throughput(baseline)` across all datasets:

```bash
bash scripts/speedup/run_speedup.sh <gpu_id> <output_dir> <model> <kv_method>
python scripts/speedup/compute_speedup.py <output_dir>
```


## Hyperparameters

Set in each pipeline config JSON:

```
level            # controls the k-gram guess length (each guess is level-1 tokens)
window           # WINDOW_SIZE — Jacobi lookahead window
num_guesses      # number of parallel n-gram guesses verified per step
n_new_tokens     # max new tokens to generate

# stream-llm (FAFO-Stream) compression backend
num_init         # always-attended sink tokens
num_local        # sliding-window size   (Init+Local ≈ the KV budget)

# quest (FAFO-Quest) compression backend
page_size        # tokens per KV page
top_k            # pages kept in the attended region
update_interval  # steps between top-k page refreshes
```


## Repository layout

```
pipeline/fafo/          FAFO decoding
pipeline/baseline/      standard auto-regressive decoding (same model loading)
config/pipeline_config/ run configs, per (dataset, model, kv-method)
config/eval_config/     eval configs, per dataset
scripts/<dataset>/      launcher scripts (fafo/ and baseline/)
scripts/speedup/        baseline-vs-FAFO speedup measurement
eval/                   dataset loaders and scorers
figures/                pipeline figure
```

## Notes

- FAFO's decoding loop **adapts its algorithm structure from the [Lookahead Decoding](https://github.com/hao-ai-lab/LookaheadDecoding) repository**; we thank the authors for open-sourcing their work.
- **Speedups vary with the GPU** (and precision, context length, and task). The reported numbers are not guaranteed on every setup — please **tune the hyperparameters** for your hardware to get the best speedup.

## Citation

If you find FAFO useful, please cite:

```bibtex
@inproceedings{le2026fafo,
  title     = {{FAFO}: Lossy {KV} Cache Compression for Lossless Inference Acceleration via Draftless Fumble Decoding},
  author    = {Le, Hoang Anh Duy and Zhong, Shaochen and Lu, Yifan and Dou, Yingtong and Yuan, Jiayi and Chuang, Yu-Neng and Fan, Xiran and Wang, Guanchu and Chen, Yuzhong and Hu, Xia},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  publisher = {PMLR}
}
```
