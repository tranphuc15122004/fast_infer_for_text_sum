# HiGOE

This repository contains the implementation of the paper: **HiGoE: Hierarchical Graph of Evidence to Enhance Retrieval-Augmented Generation for Long-context Summarization**.

## 📌Preliminary


### Environment Setup

```bash
# python==3.10
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install dgl==1.0.0+cu113 -f https://data.dgl.ai/wheels/cu113/repo.html
pip install openai==0.28
pip install pandas
pip install langchain
pip install langchain-core
pip install langchain-community
pip install langchain-experimental
pip install tiktoken
pip install tqdm
pip install bert_score
pip install rouge_score
pip install networkx
pip install faiss-gpu
pip install transformers
```

### Dataset Preparation

[QMSum](https://github.com/Yale-LILY/QMSum)
[WCEP](https://huggingface.co/datasets/ccdv/WCEP-10)
[Booksum](https://huggingface.co/datasets/kmfoda/booksum)
[GovReport](https://huggingface.co/datasets/ccdv/govreport-summarization/tree/refs%2Fconvert%2Fparquet/document)
[SQuALITY](https://github.com/nyu-mll/SQuALITY)


Save the downloaded files in the `./data/[DATASET_NAME]` folder.


> \[!IMPORTANT\]
>
> Before running the experiment, please configure your API KEY in `"get_llm_response_via_api"` in `utils.py`



## ⭐Experiments

### Proposition-Evidence Graph Construction

**Arguments:**

- `--construct_mode claim`: Enables the proposition-evidence construction strategy.
- `--enable_llm_judge`: Activates the LLM judge to filter low-quality claims.
- `--judge_sample_ratio`: Ratio of samples to judge (1.0 means judge all).
- `--judge_threshold`: Minimum score (1-5) for a claim to be accepted.

The constructed graphs are saved in the `./graph` folder.

```bash
# DATASET Choices: qmsum, wcep, booksum, govreport, squality

# 1. Construct Training Graphs (Claim Mode with LLM Judge)
python graph_construction.py --cuda 0 --dataset [DATASET] --construct_mode claim --enable_llm_judge --judge_sample_ratio 1.0 --judge_threshold 3.5 --train

# 2. Construct Test Graphs
python graph_construction.py --cuda 0 --dataset [DATASET] --construct_mode claim --enable_llm_judge --judge_sample_ratio 1.0 --judge_threshold 3.5
```



### Hierarchical Enhancement

Enhance the constructed graphs by synthesizing knowledge clusters using Personal PageRank (PPR) diffusion. This creates a hierarchical structure for better context handling. The enhanced graphs are saved in the `./graph_hierarchical` folder.

```bash
python knowledge_synthesizer_ppr.py --dataset [DATASET] --cuda 0
```



### Training Preparation

Pre-compute BERTScore and save training data in the `./training_data` folder.

```bash
# DATASET Choices: qmsum, wcep, booksum, govreport, squality
python training_preparation.py --cuda 0 --dataset [DATASET]
```



### Training

Train the model using the loss-optimized training script (`train_lossnew.py`). The weights are saved in  the `./weights` folder.


```bash
# DATASET Choices: qmsum, wcep, booksum, govreport, squality
python train.py --cuda 0 --dataset [DATASET]
```



### Inference & Evaluation

Generate summaries using the trained model and the hierarchical graphs. The summaries are saved in the `./result` folder. 


```bash
# DATASET Choices: qmsum, wcep, booksum, govreport, squality
# Generate summary results
python eval.py --cuda 0 --dataset [DATASET]
```



### Metric Calculation

Calculate ROUGE scores for the generated summaries.


```bash
# DATASET Choices: qmsum, wcep, booksum, govreport, squality
python sum_eval.py --cuda 0 --file_name ./result/[DATASET].json
```

