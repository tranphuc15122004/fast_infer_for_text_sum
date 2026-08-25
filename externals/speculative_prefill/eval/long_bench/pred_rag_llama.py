import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.append("../../")
from rag_baseline.rag_model import RagLlama


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument('--exp', type=str, default=None, help="Experiment name. ")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E. ")
    parser.add_argument('--no-8k', action='store_true', help="Exclude >8k samples. ")
    parser.add_argument('--percentage', type=float, default=0.5)

    return parser.parse_args(args)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_predictions(
    model: RagLlama, 
    max_gen, 
    data, 
    prompt_format, 
    dataset_name, 
    output_path, 
    no_8k
):
    print(f"Evaluating {dataset_name} with {len(data)} samples...")

    for json_obj in tqdm(data):
        if no_8k and json_obj["length"] >= 8000:
            continue
        apply_chat_template = dataset_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]    
        pred = model.generate(
            context=json_obj["context"], 
            input=json_obj["input"], 
            prompt_format=prompt_format, 
            dataset_name=dataset_name, 
            max_gen=max_gen, 
            apply_chat_template=apply_chat_template
        )
        
        with open(output_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred, 
                    "answers": json_obj["answers"], 
                    "all_classes": json_obj["all_classes"], 
                    "length": json_obj["length"]
                }, f, ensure_ascii=False
            )
            f.write('\n')


if __name__ == "__main__":
    seed_everything(227)
    args = parse_args()

    if args.no_8k:
        assert args.e, "No 8k is only supported for e dataset. "

    model_name = args.model

    model = RagLlama(
        llama_model_name=model_name, 
        percentage=args.percentage
    )

    # build dataset
    if args.e:
        if args.no_8k:
            # get rid of triviaqa and lcc
            datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "samsum", "passage_count", "passage_retrieval_en", "repobench-p"]
        else: 
            datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
                "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    dataset2prompt = json.load(open("./configs/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("./configs/dataset2maxlen.json", "r"))

    exp_name = args.exp if args.exp is not None else model_name.replace('/', '_')
    if args.e:
        output_path_base = f"../../local/long_bench/pred_e/{exp_name}"
    else:
        output_path_base = f"../../local/long_bench/pred/{exp_name}"
    os.makedirs(output_path_base, exist_ok=True)

    stats = {}

    for dataset_name in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset_name}_e", split='test')
        else:
            data = load_dataset('THUDM/LongBench', dataset_name, split='test')
        
        output_path = os.path.join(output_path_base, f"{dataset_name}.jsonl")
        prompt_format = dataset2prompt[dataset_name]
        max_gen = dataset2maxlen[dataset_name]
        data_all = [data_sample for data_sample in data]

        get_predictions(
            model=model, 
            max_gen=max_gen, 
            data=data_all, 
            prompt_format=prompt_format, 
            dataset_name=dataset_name, 
            output_path=output_path, 
            no_8k=args.no_8k
        )

        num_queries, avg_keep_percentage = model.print_stats()
        model.reset_stats()

        stats[dataset_name] = {
            "num_queries": num_queries, 
            "avg_keep_percentage": avg_keep_percentage
        }

    with open(os.path.join(output_path_base, "stats.json"), 'w') as f:
        json.dump(stats, f)
