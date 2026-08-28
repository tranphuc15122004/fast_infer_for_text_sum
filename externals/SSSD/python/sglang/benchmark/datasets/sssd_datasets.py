import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import requests
from datasets import load_dataset

from sglang.benchmark.datasets.common import DatasetRow

#### UTILS ####
SAVE_DIR = "/storage/datasets/huggingface/datasets"
MTBENCH_DE_URL = "https://huggingface.co/datasets/Aleph-Alpha/MTBench-German"

LONG_SYSTEM_PROMPT = """You are an advanced AI assistant designed to be helpful, honest, and harmless. Your role is to engage with the user in natural conversation, provide accurate and  reliable information, and assist with a wide variety of tasks.

## Core Principles:
1. **Helpfulness**: Always try to provide clear, complete, and practical answers to the user's requests. If the question is ambiguous, ask clarifying questions.
2. **Honesty**: Be truthful. If you are uncertain or lack sufficient knowledge, admit it clearly instead of fabricating details.
3. **Safety**: Refuse or redirect requests that involve harmful, illegal, or disallowed content (e.g., explicit violence, sexual exploitation, self-harm promotion, malware instructions, etc.).
4. **Respect**: Be polite, inclusive, and respectful. Do not use derogatory or discriminatory language.
5. **Neutrality**: Avoid taking strong political, religious, or personal stances. Present balanced perspectives when appropriate.

## Communication Style:
- Be clear, concise, and structured.
- Adapt tone to the user's needs (e.g., formal for professional use, casual for friendly conversation).
- Use examples, analogies, or step-by-step explanations when helpful.
- For lists or explanations, format with markdown for readability.

## Reasoning & Transparency:
- Show your reasoning process when it benefits the user (step-by-step solutions, breakdowns, comparisons).
- If you make assumptions, state them clearly.
- Distinguish between *facts*, *estimates*, and *opinions*.

## Knowledge & Limits:
- Your training only goes up until **June 2024**. You do not have access to information beyond that unless explicitly provided or retrieved via external tools.
- If the user asks for very recent, real-time, or niche information, tell them that your knowledge may be outdated, but still provide the best reasoning you can.

## Coding & Technical Tasks:
- When generating code, make it clean, correct, and well-documented.
- Prefer clarity over cleverness.
- Warn about potential security issues or limitations when relevant.

## Creative Tasks:
- Be imaginative, but always stay safe and respectful.
- If writing stories, roleplays, or fictional content, maintain coherence, consistency, and creativity.

## Refusals:
If the user requests something disallowed:
- Politely but firmly refuse.
- Offer a safe and constructive alternative if possible.
"""

DOLLY_15K_WITH_CONTEXT_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
)
DOLLY_15K_OPEN_PROMPT = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n"
)
PG_19_PROMPT = "Below is a text coming from a book. Provide a long and detailed summary of the text.\n\n"

SSSD_DATASETS = {
    "mt-bench",
    "gsm8k",
    "dolly-15k",
    "pg19-test",
    "math-500",
    "humaneval",
    "swe-bench",
    "hagrid",
    "mt-bench-de",
    "mt-bench-ru",
    "mt-bench-fr",
    "mt-bench-id",
    "mt-bench-jp",
    "mt-bench-it",
}


def truncate_at_sentence_boundary(tokenizer, tokenized_content, token_limit):
    detokenized_text = tokenizer.convert_tokens_to_string(tokenized_content)
    sentences = detokenized_text.split(". ")
    truncated_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.tokenize(sentence + ". ")
        sentence_length = len(sentence_tokens)
        if current_length + sentence_length > token_limit:
            break
        truncated_sentences.append(sentence + ".")
        current_length += sentence_length

    truncated_text = " ".join(truncated_sentences)
    truncated_tokens = tokenizer.tokenize(truncated_text)
    return truncated_tokens


def prepare_system_prompts(tokenizer):
    system_prefix_prompts = []
    for system_message in [LONG_SYSTEM_PROMPT]:
        message = [{"role": "system", "content": system_message}]

        chat_applied = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=False
        )

        system_prefix_prompts.append(chat_applied)

    return system_prefix_prompts


# Sample from datasets


def ensure_dataset(repo_id: str, split=None, subset: str = None):
    """
    Try to load a dataset directly from Hugging Face Hub.
    If it fails (e.g. mirror doesn't support API), download snapshot via mirror and load locally.
    """
    if subset:
        return load_dataset(repo_id, subset, split=split)
    elif split:
        return load_dataset(repo_id, split=split)
    else:
        return load_dataset(repo_id)


def _run(cmd: list, cwd: Optional[str] = None):
    """Run a shell command, raising on failure with nice stderr."""
    proc = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


# Helper for MT-Bench in German


def ensure_mtbench_german(base_dir: Optional[str] = None) -> Path:
    """
    Ensures the MTBench-German repo and patched files exist locally.
    Returns the path to the JSONL you should read:
      - MTBench-German/judge_prompts_de.jsonl  (after patch_script.sh)
      - or MTBench-German/question_de.jsonl    (fallback)
    """
    # Where to keep/clone the repo
    base_dir = base_dir or os.path.expanduser("~/.cache/mtbench-german")
    repo_dir = Path(base_dir) / "MTBench-German"
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    # If not present, clone
    if not repo_dir.exists():
        if shutil.which("git") is None:
            raise EnvironmentError(
                "`git` not found in PATH; please install git or clone the repo manually."
            )
        _run(["git", "clone", MTBENCH_DE_URL, str(repo_dir)])

    # If judge file missing (or user prefers it), run the patch script (idempotent).
    patch_script = repo_dir / "patch_script.sh"
    if patch_script.exists():
        _run(["bash", str(patch_script)], cwd=str(repo_dir))

    q_path = repo_dir / "question_de.jsonl"
    if q_path.exists():
        return q_path

    raise FileNotFoundError(
        "Could not find MTBench-German question files. "
        f"Tried: {q_path}. "
        "If the repo changed, inspect its tree or update the file names."
    )


# Helper for SWE-Bench (to load prompts that were selected and used to build the datastore)


def load_ok_pairs(jsonl_path):
    """Return a set of (repo, commit) pairs from jsonl where error is null."""
    ok = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("error") is None:
                repo = rec.get("repo")
                commit = rec.get("commit")
                if repo and commit:
                    ok.add((repo, commit))
    return ok


def commit_matches(a, b):
    """Allow exact match or prefix match either way (handles shortened hashes)."""
    if not a or not b:
        return False
    return a == b or a.startswith(b) or b.startswith(a)


def load_prompts(
    dataset_name,
    tokenizer,
    max_generated_toks,
    num_prompts=100,
    add_system_prompt=False,
):
    prompts = []

    if dataset_name == "mt-bench":
        dataset = ensure_dataset("philschmid/mt-bench", split="train")
        for idx in range(min(num_prompts, len(dataset))):
            message = (
                [{"role": "system", "content": LONG_SYSTEM_PROMPT}]
                if add_system_prompt
                else []
            )
            message.append({"role": "user", "content": dataset[idx]["turns"][0]})
            prompts.append(
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

    elif dataset_name == "gsm8k":
        dataset = ensure_dataset("openai/gsm8k", subset="main", split="test")
        for idx in range(min(num_prompts, len(dataset))):
            message = (
                [{"role": "system", "content": LONG_SYSTEM_PROMPT}]
                if add_system_prompt
                else []
            )
            message.append({"role": "user", "content": dataset[idx]["question"]})
            prompts.append(
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

    elif dataset_name == "dolly-15k":
        dataset = ensure_dataset("databricks/databricks-dolly-15k", split="train")
        for sample in dataset.select(range(min(num_prompts, len(dataset)))):
            context = sample["context"]
            if context == "":
                input_message = DOLLY_15K_OPEN_PROMPT
                user_prompt = sample["instruction"]
            else:
                input_message = DOLLY_15K_WITH_CONTEXT_PROMPT
                user_prompt = (
                    "Instruction:\n" + sample["instruction"] + "\nContext:\n" + context
                )

            message = (
                [{"role": "system", "content": LONG_SYSTEM_PROMPT}]
                if add_system_prompt
                else []
            )
            message.append({"role": "user", "content": input_message + user_prompt})

            prompts.append(
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

    elif dataset_name == "pg19-test":
        dataset = ensure_dataset("emozilla/pg19-test", split="test")
        max_context_len = min(
            40_000, tokenizer.model_max_length - max_generated_toks - 1000
        )
        for i in range(num_prompts):
            text_content = dataset[i]["text"]
            tokenized = tokenizer.tokenize(text_content)
            if len(tokenized) > max_context_len:
                tokenized = truncate_at_sentence_boundary(
                    tokenizer, tokenized, max_context_len
                )
            detok = tokenizer.convert_tokens_to_string(tokenized)
            message = (
                [{"role": "system", "content": LONG_SYSTEM_PROMPT}]
                if add_system_prompt
                else []
            )
            message.append({"role": "user", "content": PG_19_PROMPT + detok})
            prompts.append(
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

    elif dataset_name == "math-500":
        dataset = ensure_dataset("HuggingFaceH4/MATH-500", split="test")
        dataset = [ex for ex in dataset if ex["level"] in [4, 5]]
        for idx in range(min(num_prompts, len(dataset))):
            sample = dataset[idx]
            message = (
                [{"role": "system", "content": LONG_SYSTEM_PROMPT}]
                if add_system_prompt
                else []
            )
            message.append({"role": "user", "content": sample["problem"]})

            prompts.append(
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

    elif dataset_name == "humaneval":
        dataset = ensure_dataset("openai/openai_humaneval", split="test")
        for sample in dataset.select(range(min(num_prompts, len(dataset)))):
            task = sample["prompt"]
            message = (
                [{"role": "system", "content": LONG_SYSTEM_PROMPT}]
                if add_system_prompt
                else []
            )
            message.append({"role": "user", "content": task})

            prompts.append(
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

    elif dataset_name == "swe-bench":
        prompts = []
        try:
            dataset = ensure_dataset(
                "princeton-nlp/SWE-bench_Lite_oracle", split="test"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SWE-bench_Lite_oracle (test): {e}")

        for sample in dataset.select(range(min(num_prompts, len(dataset)))):
            task = sample.get("text")
            if not task:
                continue

            message = (
                [{"role": "system", "content": LONG_SYSTEM_PROMPT}]
                if add_system_prompt
                else []
            )
            message.append({"role": "user", "content": task})

            prompts.append(
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

    elif dataset_name == "hagrid":
        dataset = load_dataset("miracl/hagrid", split="train", trust_remote_code=True)
        prompts = []
        for sample in dataset.select(range(min(num_prompts, len(dataset)))):
            query = sample["query"]
            quotes = sample["quotes"]

            # Format the quotes as numbered snippets the model must cite
            quotes_str = "\n".join(
                f"[{i}] (docid={q['docid']}, idx={q['idx']}) {q['text']}"
                for i, q in enumerate(quotes)
            )

            user_content = f'Answer the question using ONLY the quotes below.\nCite the supporting quote(s) inline like [0], [3], etc.\nIf the quotes are insufficient, say "I don\'t know".\nQuestion: {query}\nQuotes: {quotes_str}'

            if add_system_prompt:
                message = [{"role": "system", "content": LONG_SYSTEM_PROMPT}]
            else:
                message = []
            message.append({"role": "user", "content": user_content})

            prompts.append(
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

    elif dataset_name == "mt-bench-de":
        data_path = ensure_mtbench_german()

        prompts = []
        with open(data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= num_prompts:
                    break
                obj = json.loads(line)
                turns = obj.get("turns") or obj.get("questions") or []
                if not turns:
                    raise ValueError(f"Unable to find turns in dataset: {dataset_name}")

                user_turn = turns[0]

                if add_system_prompt:
                    message = [{"role": "system", "content": LONG_SYSTEM_PROMPT}]
                else:
                    message = []
                message.append({"role": "user", "content": user_turn})
                prompt = tokenizer.apply_chat_template(
                    message,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
                prompts.append(prompt)

    elif dataset_name == "mt-bench-ru":
        url = "https://huggingface.co/datasets/NLPCoreTeam/ruMT-Bench/resolve/main/question.jsonl"

        response = requests.get(url)
        response.raise_for_status()

        dataset = [json.loads(line) for line in response.text.splitlines()]

        for idx in range(min(num_prompts, len(dataset))):
            message = (
                [{"role": "system", "content": LONG_SYSTEM_PROMPT}]
                if add_system_prompt
                else []
            )
            message.append({"role": "user", "content": dataset[idx]["turns"][0]})
            prompts.append(
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

    elif dataset_name == "mt-bench-fr":
        dataset = ensure_dataset("bofenghuang/mt-bench-french", split="test")
        for idx in range(min(num_prompts, len(dataset))):
            message = (
                [{"role": "system", "content": LONG_SYSTEM_PROMPT}]
                if add_system_prompt
                else []
            )
            message.append({"role": "user", "content": dataset[idx]["turns"][0]})
            prompts.append(
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

    elif dataset_name == "mt-bench-id":
        dataset = ensure_dataset("lightblue/mt_bench_indonesian", split="train")
        for idx in range(min(num_prompts, len(dataset))):
            message = (
                [{"role": "system", "content": LONG_SYSTEM_PROMPT}]
                if add_system_prompt
                else []
            )
            message.append({"role": "user", "content": dataset[idx]["turns"][0]})
            prompts.append(
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

    elif dataset_name == "mt-bench-jp":
        dataset = ensure_dataset("karakuri-ai/corrected-mt-bench-ja", split="test")
        for idx in range(min(num_prompts, len(dataset))):
            message = (
                [{"role": "system", "content": LONG_SYSTEM_PROMPT}]
                if add_system_prompt
                else []
            )
            message.append({"role": "user", "content": dataset[idx]["turns"][0]})
            prompts.append(
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

    elif dataset_name == "mt-bench-it":
        here = Path(__file__).resolve().parent
        jsonl_path = (
            here
            / ".."
            / ".."
            / ".."
            / ".."
            / "speculative_bench_scripts"
            / "datasets"
            / "mt-bench_ita.jsonl"
        )
        jsonl_path = jsonl_path.resolve()

        dataset = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                dataset.append(json.loads(line))

        for idx in range(min(num_prompts, len(dataset))):
            example = dataset[idx]

            message = (
                [{"role": "system", "content": LONG_SYSTEM_PROMPT}]
                if add_system_prompt
                else []
            )
            message.append({"role": "user", "content": example["turns"][0]})

            prompts.append(
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Please add it to the bench_datasets.py file."
        )

    random.seed(42)
    random.shuffle(prompts)

    return prompts


def get_sssd_dataset(bench_args, tokenizer) -> List[DatasetRow]:
    fixed_output_len = bench_args.sharegpt_output_len
    prompts = load_prompts(
        bench_args.dataset_name,
        tokenizer,
        max_generated_toks=fixed_output_len,
        num_prompts=bench_args.num_prompts,
    )
    rows = []
    for prompt in prompts:
        rows.append(
            DatasetRow(
                prompt=prompt,
                prompt_len=len(tokenizer(prompt).input_ids),
                output_len=fixed_output_len,
            )
        )
    return rows
