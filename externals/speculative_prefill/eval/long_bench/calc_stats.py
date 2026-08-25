import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

categories = {
    "Summarization": ["gov_report", "qmsum", "multi_news", "vcsum"], 
    "Single-Document QA": ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"], 
    "Multi-Document QA": ["hotpotqa", "2wikimqa", "musique", "dureader"], 
    "Few-Shot Learning": ["trec", "triviaqa", "samsum", "lsht"], 
    "Synthetic Tasks": ["passage_count", "passage_retrieval_en", "passage_retrieval_zh"], 
    "Code Completion": ["lcc", "repobench-p"], 
}

dataset2prompt = json.load(open("./configs/dataset2prompt.json", "r"))

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

name = []
value = []
group = []

for category in categories.keys():
    for dataset_name in categories[category]:
        prompt = dataset2prompt[dataset_name]
        data = load_dataset('THUDM/LongBench', dataset_name, split='test')
        samples = [prompt.format(**s) for s in data]
        lens = [len(tokenizer.encode(s)) for s in samples]
        avg_len = int(sum(lens) / len(lens))
        
        name.append(dataset_name)
        value.append(avg_len)
        group.append(category)

# name = np.load("./name.npy")
# value = np.load("./value.npy")
# group = np.load("./group.npy")

df = pd.DataFrame({
    "name": name,
    "value": value,
    "group": group
})

def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    return rotation, alignment

def add_labels(angles, values, labels, offset, ax):
    
    # This is the space between the end of the bar and the label
    padding = 16000
    
    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle
        
        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            x=angle, 
            y=value + padding, 
            s=f"{label}\n({value})", 
            ha=alignment, 
            va="center", 
            rotation=rotation, 
            rotation_mode="anchor"
        )

VALUES = df["value"].values
LABELS = df["name"].values
GROUP = df["group"].values
OFFSET = np.pi / 2

# Add three empty bars to the end of each group
PAD = 1
ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
WIDTH = (2 * np.pi) / len(ANGLES)

# Obtain size of each group
GROUPS_SIZE = [len(c) for c in categories.values()]

# Obtaining the right indexes is now a little more complicated
offset = 0
IDXS = []
for size in GROUPS_SIZE:
    IDXS += list(range(offset + PAD, offset + size + PAD))
    offset += size + PAD

# Same layout as above
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

cmap = plt.get_cmap("viridis", 6)
COLORS = [cmap(i) for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]

ax.bar(
    ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS, 
    edgecolor="white", linewidth=0.5, bottom=15000
)

add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)

offset = 0 
for group, size in zip([
    "Sum",
    "Single-Doc\nQA",
    "Multi-Doc\nQA",
    "Few-Shots\nLearning",
    "Synthetic\nTasks", 
    "Code\nCompletion", 
], GROUPS_SIZE):
    # Add line below bars
    x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=100)
    ax.plot(x1, [14000] * 100, color="#333333")

    ax.text(
        np.mean(x1), 11000, group, color="#333333", fontsize=8, 
        fontweight="bold", ha="center", va="center"
    )
    
    offset += size + PAD

ax.set_theta_offset(OFFSET)
ax.set_ylim(5000, None)
ax.set_frame_on(False)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.subplots_adjust(bottom=-0.1)
plt.savefig("./longbench_len.pdf", dpi=500)