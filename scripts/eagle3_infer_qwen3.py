#!/usr/bin/env python3
"""Minimal single-prompt EAGLE3 inference without the FastChat evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "externals" / "EAGLE"))

from eagle.model.ea_model import EaModel  # noqa: E402


def load_prompt(question_file: Path, begin: int, end: int) -> str:
    rows = [
        json.loads(line)
        for line in question_file.read_text().splitlines()
        if line.strip()
    ]
    selected = rows[begin:end]
    if not selected:
        raise ValueError(f"No question in range [{begin}, {end}) from {question_file}")
    return selected[0]["turns"][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--eagle-model", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--question-begin", type=int, default=0)
    parser.add_argument("--question-end", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--total-token", type=int, default=32)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    question = load_prompt(
        Path(args.question_file), args.question_begin, args.question_end
    )
    print(f"Question: {question}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("EAGLE3 inference requires a visible CUDA GPU")

    model = EaModel.from_pretrained(
        base_model_path=args.base_model,
        ea_model_path=args.eagle_model,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # EAGLE3 builds tree tensors with .tolist(), so CPU/meta offloading
        # from device_map="auto" is not supported here. Keep both models on
        # the single Tesla T4.
        device_map={"": "cuda:0"},
        use_eagle3=True,
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).cuda()

    with torch.inference_mode():
        output_ids, new_tokens, tree_steps = model.eagenerate(
            input_ids,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            max_length=input_ids.shape[1] + args.max_new_tokens + args.total_token + 32,
            log=True,
        )

    answer = tokenizer.decode(
        output_ids[0, input_ids.shape[1] :],
        skip_special_tokens=True,
    ).strip()
    print(f"Answer: {answer}")
    print(f"Accepted new tokens: {new_tokens}; tree steps: {tree_steps}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "question": question,
                "answer": answer,
                "new_tokens": int(new_tokens),
                "tree_steps": int(tree_steps),
                "base_model": args.base_model,
                "eagle_model": args.eagle_model,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
