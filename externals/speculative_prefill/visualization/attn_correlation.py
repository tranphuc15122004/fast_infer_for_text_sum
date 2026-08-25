import random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from scipy.stats import kendalltau
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

spec_model_name = 'meta-llama/Llama-3.2-1B-Instruct'
base_model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
tokenizer_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

seq_len = 512
num_samples = 32
seed = 227

# seed everything
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

# load data
dataset = load_dataset(
    "allenai/paloma", 
    name="c4_en", 
    split="test"
)
texts = dataset.shuffle(seed=seed)["text"]

# tokenizer data into batch
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token

# we want to find examples whose tokenized len >= seqlen
samples = []
for text in texts:
    tokenized_output = tokenizer(
        text, 
        truncation=True, 
        max_length=seq_len, 
        return_tensors='pt'
    )
    input_ids = tokenized_output["input_ids"][0]
    if len(input_ids) >= seq_len:
        samples.append(input_ids)
    if len(samples) == num_samples:
        break

print(f"Found {len(samples)} samples with number of tokens >= {seq_len}")
samples = torch.stack(samples, dim=0).to('cuda')

# get attn scores
@torch.inference_mode
def get_attn_scores(model_name) -> Tuple[torch.Tensor]:
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        attn_implementation="eager", # for outputing attn
    )

    output = model.forward(
        input_ids=samples, 
        output_attentions=True, 
        return_dict=True
    )

    # layer x [bs, head, q_len, k_len]
    return output.attentions

# store attn scores
spec_attns = get_attn_scores(spec_model_name)
base_attns = get_attn_scores(base_model_name)

# visualize relationship between two models
kls = torch.zeros((len(spec_attns), len(base_attns)))
taus = torch.zeros((len(spec_attns), len(base_attns)))

for si in range(len(spec_attns)):
    for bi in range(len(base_attns)):
        # [bs, head, q_len, k_len]
        # max over heads
        sa = torch.max(spec_attns[si][..., -1, :], dim=1)[0].to(torch.float32)
        ba = torch.max(base_attns[bi][..., -1, :], dim=1)[0].to(torch.float32)

        _, sa_ordering = torch.sort(sa, dim=-1)
        _, ba_ordering = torch.sort(ba, dim=-1)

        batch_taus = []
        for sao, bao in zip(sa_ordering, ba_ordering):
            batch_taus.append(kendalltau(
                sao.cpu().numpy(), bao.cpu().numpy()
            )[0])

        sa = torch.nn.functional.log_softmax(sa, dim=-1)
        ba = torch.nn.functional.softmax(ba, dim=-1)

        batch_kls = torch.nn.functional.kl_div(sa, ba, reduction='none')

        kls[si, bi] = batch_kls.mean()
        taus[si, bi] = sum(batch_taus) / len(batch_taus)

sns.set_style("whitegrid")
_, axes = plt.subplots(
    3, 1, 
    figsize=(6.4, 4.8 * 3))

sns.heatmap(kls, ax=axes[0])
sns.heatmap(taus, ax=axes[1], cmap=sns.color_palette("light:b", as_cmap=True))
axes[1].set_xlabel("Layer idx of base model")
axes[0].set_ylabel("Layer idx of spec model")
axes[1].set_ylabel("Layer idx of spec model")

axes[0].set_title("KL Divergence")
axes[1].set_title("Kendall Tau")

axes[0].invert_yaxis()
axes[1].invert_yaxis()

# topk plot [bs, k_len]
spec_attns = torch.stack([sa[..., -1, :] for sa in spec_attns], dim=0).max(2)[0].max(0)[0]
base_attns = torch.stack([ba[..., -1, :] for ba in base_attns], dim=0).max(2)[0].max(0)[0]

jaccords = []
topk = []

for ratio in range(1, 10):
    k = int(seq_len * ratio / 10)
    topk_sa = torch.topk(spec_attns, k=k, dim=-1).indices
    topk_ba = torch.topk(base_attns, k=k, dim=-1).indices

    mask_sa = torch.zeros_like(spec_attns, dtype=torch.bool)
    mask_ba = torch.zeros_like(base_attns, dtype=torch.bool)

    mask_sa.scatter_(1, topk_sa, 1)
    mask_ba.scatter_(1, topk_ba, 1)

    intersection = (mask_sa & mask_ba).sum(-1).float()
    union = (mask_sa | mask_ba).sum(-1).float() + 1e-9

    jaccord = (intersection / union).tolist()
    jaccords.append(jaccord)
    topk.append(ratio * 10)

df = pd.DataFrame(np.array(jaccords))
df['topk percentage'] = topk
df = df.melt(
    id_vars='topk percentage', 
    var_name='observation', 
    value_name='jaccord similarity'
)

sns.lineplot(
    data=df, 
    x='topk percentage', 
    y='jaccord similarity', 
    ax=axes[2]
)
axes[2].set_title("Jaccord Similarity of TopK Tokens")
axes[2].invert_xaxis()

plt.tight_layout()
plt.savefig('./visualization/vis_1b8b.png', dpi=300)
