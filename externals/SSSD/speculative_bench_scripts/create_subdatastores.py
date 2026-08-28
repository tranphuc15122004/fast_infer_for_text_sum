"""
Datastore builder with optional *incremental multi-index* creation and per-run token caps.

New flags:
  --multi-incremental             Build cumulative sub-datastores in one pass.
  --multi-sizes 100k,1m,10m,...   Override default cutoffs (100k,1m,10m,100m,1g).
  --extend-by 100k                Cap this run to at most N tokens (single or multi mode).

Examples:
nohup python create_subdatastores.py \
    --index_file_path /storage/users/mmarzollo/datastores/incremental/llama3.1-8B_incremental.idx \
    --model /storage/datasets/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
    --datasets hitz-magpie-llama3.1-8b magpie-llama31-pro \
    --multi-incremental > en_incremental.log 2>&1 &

python create_subdatastores.py \
    --index_file_path /storage/datasets/stacked.idx \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --datasets python-stack \
    --stack-token $HF_TOKEN \
    --multi-incremental --multi-sizes 50k,250k,2m

# Extend an existing single index by *exactly* 100k tokens this run:
python create_subdatastores.py \
    --index_file_path /storage/users/mmarzollo/datastores/incremental/llama3.1-8B_incremental.1g.german_base.idx \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --datasets sharegpt-de synthia-german \
    --extend-index \
    --extend-by 100k

"""

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import sssd_speculator

# ---- Config ----

BATCH_SIZE_DEFAULT = 512

# Map short keywords -> (HF repo id, reader key)
KEYWORD_MAP = {
    # Magpie family (same reader)
    "magpie-pro": ("Magpie-Align/Magpie-Pro-MT-300K-v0.1", "magpie"),
    "magpie-air": ("Magpie-Align/Magpie-Air-MT-300K-v0.1", "magpie"),
    "magpie-llama31-pro": ("Magpie-Align/Magpie-Llama-3.1-Pro-MT-500K-v0.1", "magpie"),
    "magpie-qwen-coder": ("Magpie-Align/Magpie-Qwen2.5-Coder-Pro-300K-v0.1", "magpie"),
    "magpie-qwen2-cn": ("Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese", "magpie"),
    "magpie-llama33-pro-1M": ("Magpie-Align/Magpie-Llama-3.3-Pro-1M-v0.1", "magpie"),
    "magpie-llama33-reason": (
        "Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Llama3",
        "magpie",
    ),
    "hitz-magpie-llama3.1-8b": ("HiTZ/Magpie-Llama-3.1-8B-Instruct-Filtered", "magpie"),
    # ShareGPT / UltraChat
    "sharegpt": ("Aeala/ShareGPT_Vicuna_unfiltered", "sharegpt"),
    "sharegpt-de": ("FreedomIntelligence/sharegpt-deutsch", "sharegpt"),
    "sharegpt-ita": ("FreedomIntelligence/sharegpt-italian", "sharegpt"),
    "sharegpt-ind": ("FreedomIntelligence/sharegpt-indonesian", "sharegpt"),
    "sharegpt-jap": ("FreedomIntelligence/sharegpt-japanese", "sharegpt"),
    "evol-instruct-en": ("WizardLMTeam/WizardLM_evol_instruct_V2_196k", "sharegpt"),
    "evol-instruct-ita": ("FreedomIntelligence/evol-instruct-italian", "sharegpt"),
    "evol-instruct-ind": ("FreedomIntelligence/evol-instruct-indonesian", "sharegpt"),
    "evol-instruct-jap": ("FreedomIntelligence/evol-instruct-japanese", "sharegpt"),
    # Other large chats
    "wildchat-en": ("allenai/WildChat-1M", "wildchat"),
    "ultrachat": ("stingning/ultrachat", "ultrachat"),
    # DeepSeek-R1 family
    "deepseek-r1-dolphin": ("DKYoon/dolphin-r1-deepseek-filtered", "deepseek_dolphin"),
    "deepseek-r1-distill": ("tuanha1305/DeepSeek-R1-Distill", "deepseek_distill"),
    "deepseek-r1-chinese": (
        "Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT",
        "deepseek_chinese",
    ),
    "pile-val": ("monology/pile-uncopyrighted", "pile"),
    "synthia-german": ("jphme/synthia_german_experimental", "synthia_german"),
    "baseten-gpt-oss": (
        "baseten-admin/gpt-oss120b-generated-magpie-1m-v0.1",
        "baseten_gpt",
    ),
    "jackrong-gpt-oss": ("Jackrong/gpt-oss-120B-distilled-reasoning", "jackrong_gpt"),
    "python-stack": ("bigcode/the-stack-dedup", "python_stack"),
}

# ---- Tokenization helpers ----


def batch_tokenize_and_add(
    writer, texts: List[str], tokenizer: PreTrainedTokenizerBase
):
    if not texts:
        return
    token_lists = tokenizer.batch_encode_plus(texts, add_special_tokens=True)[
        "input_ids"
    ]
    for token_list in token_lists:
        writer.add_entry(token_list)


def write_texts(
    writer,
    text_iter: Iterable[str],
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
):
    batch = []
    for t in text_iter:
        if t is None:
            continue
        batch.append(t)
        if len(batch) >= batch_size:
            batch_tokenize_and_add(writer, batch, tokenizer)
            batch = []
    if batch:
        batch_tokenize_and_add(writer, batch, tokenizer)


def write_texts_capped(
    writer,
    text_iter: Iterable[str],
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
    cap_tokens: int,
):
    """Write at most cap_tokens into writer (truncate last entry if needed)."""
    if cap_tokens is None or cap_tokens <= 0:
        return
    added = 0
    batch = []

    def add_tokens(tokens: List[int]):
        nonlocal added
        if added >= cap_tokens:
            return True
        room = cap_tokens - added
        if len(tokens) <= room:
            writer.add_entry(tokens)
            added += len(tokens)
            return False
        else:
            # truncate final piece to hit the cap exactly
            writer.add_entry(tokens[:room])
            added += room
            return True

    for t in text_iter:
        if t is None:
            continue
        batch.append(t)
        if len(batch) >= batch_size:
            token_lists = tokenizer.batch_encode_plus(batch, add_special_tokens=True)[
                "input_ids"
            ]
            for tl in token_lists:
                if add_tokens(tl):
                    return
            batch = []
    if batch and (cap_tokens is None or added < cap_tokens):
        token_lists = tokenizer.batch_encode_plus(batch, add_special_tokens=True)[
            "input_ids"
        ]
        for tl in token_lists:
            if add_tokens(tl):
                return


# ---- Readers ----


def iter_magpie(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        for m in row["conversations"]:
            if m.get("from") == "gpt":
                yield m["value"]


def iter_sharegpt(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        for m in row["conversations"]:
            if m.get("from") == "gpt":
                yield m["value"]


def iter_ultrachat(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        msgs = row["data"]
        for i in range(1, len(msgs), 2):
            yield msgs[i]


def iter_pile(repo: str, hf_token: str) -> Iterable[str]:
    for split in ("validation", "test"):
        ds = load_dataset(repo, split=split, streaming=True)
        for ex in ds:
            yield ex["text"]


def iter_deepseek_dolphin(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        for m in row["messages"]:
            if m.get("role") == "assistant":
                yield m["content"]


def iter_deepseek_distill(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        yield f"<think>\n{row['content']}\n</think>\n\n{row['reasoning_content']}"


def iter_deepseek_chinese(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        yield row["output"]


def iter_synthia_german(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        yield row["response"]


def iter_baseten_gpt(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        for m in row["conversations"]:
            if m.get("role") == "assistant":
                yield m["content"]


def iter_jackrong_gpt(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        yield row["output"]


def iter_python_stack(repo: str, hf_token: str) -> Iterable[str]:
    segment = 3
    data_files = []
    for i in range(segment):
        if i >= 100:
            data_files.append(f"data-00{i}-of-00144.parquet")
        elif i >= 10:
            data_files.append(f"data-000{i}-of-00144.parquet")
        else:
            data_files.append(f"data-0000{i}-of-00144.parquet")
    ds = load_dataset(
        repo,
        data_dir="data/python",
        split="train",
        data_files=data_files,
        token=hf_token,
    )
    for row in ds:
        yield row["content"]


def iter_wildchat_en(repo: str, hf_token: str) -> Iterable[str]:
    num_conv = 150_000
    ds = load_dataset(repo)["train"]
    tot_conv = 0

    for row in ds:
        if row.get("language") != "English":
            continue
        if row.get("toxic", False):
            continue

        # Find the first assistant message
        first_assistant_msg = None
        for turn in row.get("conversation", []):
            if turn.get("role") != "assistant":
                continue
            content = (turn.get("content") or "").strip()
            if content:
                first_assistant_msg = content
                break

        # Skip conversations without a usable first assistant message
        if not first_assistant_msg:
            continue

        # Yield the first assistant message from the conversation
        yield first_assistant_msg

        tot_conv += 1
        if num_conv is not None and tot_conv >= num_conv:
            break


def load_npz_sequences(filepath: str) -> Iterable[List[int]]:
    data = np.load(filepath)
    for key in data:
        yield data[key].tolist()


# ---- JSONL helpers ----


def iter_jsonl_texts(path: str) -> Iterable[str]:
    """
    Yields text from a JSONL file.

    Supported row formats:
      1) {"text": "..."}
         -> yields obj["text"]

      2) {
             "assistant_samples": [
                 {"model_output": "..."},
                 {"model_output": "..."},
                 ...
             ],
             ...
         }
         -> yields each sample["model_output"] (if present and a string)

    If both "assistant_samples" and "text" are present, assistant_samples win
    and "text" is ignored for that row.

    Rows missing the relevant fields or with non-string values are skipped.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            # 1) Prefer assistant_samples if present
            if "assistant_samples" in obj and isinstance(
                obj["assistant_samples"], list
            ):
                any_yielded = False
                for sample in obj["assistant_samples"]:
                    if isinstance(sample, dict):
                        out = sample.get("model_output")
                        if isinstance(out, str) and out:
                            any_yielded = True
                            yield out
                if any_yielded:
                    # Don't fall back to "text" if we already emitted outputs
                    continue

            # 2) Fallback: plain {"text": "..."}
            if "text" in obj and isinstance(obj["text"], str):
                yield obj["text"]


def write_jsonl(path: str, entries: Iterable[Union[str, Dict[str, Any]]]):
    """
    Writes entries to JSONL.
    - If entry is a str, it becomes {text_field: entry}.
    - If entry is a dict, it's written as-is.
    """
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            if isinstance(e, str):
                obj = {"text": e}
            elif isinstance(e, dict):
                obj = e
            else:
                continue
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


READERS = {
    "magpie": iter_magpie,
    "sharegpt": iter_sharegpt,
    "ultrachat": iter_ultrachat,
    "pile": iter_pile,
    "deepseek_dolphin": iter_deepseek_dolphin,
    "deepseek_distill": iter_deepseek_distill,
    "deepseek_chinese": iter_deepseek_chinese,
    "synthia_german": iter_synthia_german,
    "baseten_gpt": iter_baseten_gpt,
    "jackrong_gpt": iter_jackrong_gpt,
    "python_stack": iter_python_stack,
    "wildchat": iter_wildchat_en,
}

# ---- Resolver ----


def resolve_item(item: str):
    """
    Returns a spec dict:
      - NPZ: {"kind": "npz", "path": ...}
      - JSONL: {"kind": "jsonl", "path": ...}
      - Known keyword: {"kind":"hf", "repo":..., "reader":...}
      - Direct HF repo (optional): if it's a Magpie repo, reuse the Magpie reader.
    """
    low = item.lower()
    if low.endswith(".npz"):
        return {"kind": "npz", "path": item}
    if low.endswith(".jsonl"):
        return {"kind": "jsonl", "path": item}

    if item in KEYWORD_MAP:
        repo, reader = KEYWORD_MAP[item]
        return {"kind": "hf", "repo": repo, "reader": reader}

    if "/" in item:  # looks like a HF repo id
        reader = "magpie" if item.startswith("Magpie-Align/") else None
        if reader is None:
            raise ValueError(
                f"Unknown HF repo '{item}'. Add a keyword mapping or use a supported repo."
            )
        return {"kind": "hf", "repo": item, "reader": reader}

    raise ValueError(f"Unknown dataset keyword or path: '{item}'")


# ---- Base single-index build ----


def build_index(
    index_file_path: str,
    datasets: List[str],
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
    hf_token: str,
    extend_cap: Optional[int],
):
    writer = sssd_speculator.Writer(
        index_file_path=index_file_path,
        vocab_size=tokenizer.vocab_size + 200,
    )
    for item in datasets:
        spec = resolve_item(item)
        print(f"→ Processing {item} …")
        if spec["kind"] == "npz":
            if extend_cap is None:
                for seq in load_npz_sequences(spec["path"]):
                    writer.add_entry(seq)
            else:
                # Cap-aware for NPZ too
                remaining = extend_cap
                for seq in load_npz_sequences(spec["path"]):
                    if remaining <= 0:
                        break
                    if len(seq) <= remaining:
                        writer.add_entry(seq)
                        remaining -= len(seq)
                    else:
                        writer.add_entry(seq[:remaining])
                        remaining = 0
                        break
        elif spec["kind"] == "jsonl":
            if extend_cap is None:
                write_texts(
                    writer, iter_jsonl_texts(spec["path"]), batch_size, tokenizer
                )
            else:
                write_texts_capped(
                    writer,
                    iter_jsonl_texts(spec["path"]),
                    batch_size,
                    tokenizer,
                    extend_cap,
                )
                extend_cap = 0  # consumed
        else:
            reader_fn = READERS[spec["reader"]]
            if extend_cap is None:
                write_texts(
                    writer, reader_fn(spec["repo"], hf_token), batch_size, tokenizer
                )
            else:
                write_texts_capped(
                    writer,
                    reader_fn(spec["repo"], hf_token),
                    batch_size,
                    tokenizer,
                    extend_cap,
                )
                extend_cap = 0
    writer.finalize()


# ---- Incremental multi-index build ----

DEFAULT_SIZES = [100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]


def parse_size_atom(s: str) -> int:
    s = s.strip().lower()
    if not s:
        raise ValueError("Empty size string.")
    suffix = s[-1]
    mult = 1
    if suffix in ("k", "m", "g"):
        base = s[:-1]
        mult = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}[suffix]
    else:
        base = s
    return int(float(base) * mult)


def parse_sizes_arg(arg: str) -> List[int]:
    return [parse_size_atom(x) for x in arg.split(",") if x.strip()]


def format_size_name(n: int) -> str:
    # for filenames: 100k / 1m / 10m / 100m / 1g
    if n % 1_000_000_000 == 0:
        return f"{n // 1_000_000_000}g"
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}m"
    if n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


def derive_index_path(base_path: str, size: int) -> str:
    name = format_size_name(size)
    if base_path.endswith(".idx"):
        return base_path[:-4] + f".{name}.idx"
    return base_path + f".{name}.idx"


class IncrementalWriters:
    """
    Fan-out writer: feeds each token *chunk* to all writers whose cutoff
    has not yet been reached. Ensures exact token counts per cutoff during a fresh run.

    NOTE on --extend-index:
      We don't attempt to read existing token counts. If derived files exist and you pass
      --extend-index, we append more tokens to them; cutoffs are enforced *within this run*
      (cumulated from zero for this run), not relative to prior contents.
    """

    def __init__(
        self,
        base_path: str,
        sizes: List[int],
        tokenizer: PreTrainedTokenizerBase,
        allow_extend: bool,
        per_run_cap: Optional[int],
    ):
        self.sizes: List[int] = sorted(set(sizes))
        self.tokenizer = tokenizer
        self.cum_tokens: int = 0  # tokens emitted this run
        self.per_run_cap: Optional[int] = per_run_cap
        self.run_added: int = 0
        self.paths: List[str] = [derive_index_path(base_path, s) for s in self.sizes]
        # preflight: existence checks honoring extend flag
        for p in self.paths:
            if os.path.exists(p) and not allow_extend:
                raise FileExistsError(
                    f"Index file {p} already exists. Pass --extend-index to append, "
                    "or delete it to rebuild."
                )
        # create writers
        self.writers: List[Tuple[int, Any]] = []
        for s, p in zip(self.sizes, self.paths):
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            w = sssd_speculator.Writer(
                index_file_path=p, vocab_size=self.tokenizer.vocab_size + 200
            )
            self.writers.append((s, w))

    def _room_run(self) -> Optional[int]:
        if self.per_run_cap is None:
            return None
        return max(0, self.per_run_cap - self.run_added)

    def add_token_list(self, tokens: List[int]):
        if not tokens:
            return
        # Respect per-run cap first
        if self.per_run_cap is not None and self.run_added >= self.per_run_cap:
            return
        idx = 0
        remaining = len(tokens)
        while remaining > 0:
            # compute room w.r.t per-run cap
            if self.per_run_cap is not None:
                room_run = self._room_run()
                if room_run is None or room_run <= 0:
                    return
            # compute how many tokens until next cutoff (within this run)
            gaps = [s - self.cum_tokens for s in self.sizes if self.cum_tokens < s]
            if not gaps:
                # all writers considered full for this run; but if per_run_cap exists,
                # we still need to respect it (keep appending to all writers)
                step_cutoff = remaining
            else:
                step_cutoff = min(remaining, min(gaps))
            step = step_cutoff
            if self.per_run_cap is not None:
                step = min(step, self._room_run())
            if step is None or step <= 0:
                return
            chunk = tokens[idx : idx + step]
            # write this chunk to all writers
            for cutoff, w in self.writers:
                # If within run we already reached the cutoff, skip writing to that writer
                if self.cum_tokens < cutoff:
                    w.add_entry(chunk)
                else:
                    # writer already "past its per-run cutoff"; skip
                    pass
            self.cum_tokens += step
            self.run_added += step
            idx += step
            remaining -= step

    def finalize(self):
        for _, w in self.writers:
            w.finalize()


def write_texts_incremental(
    incr: IncrementalWriters,
    text_iter: Iterable[str],
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
):
    batch = []
    for t in text_iter:
        if t is None:
            continue
        batch.append(t)
        if len(batch) >= batch_size:
            token_lists = tokenizer.batch_encode_plus(batch, add_special_tokens=True)[
                "input_ids"
            ]
            for tl in token_lists:
                incr.add_token_list(tl)
            batch = []
    if batch:
        token_lists = tokenizer.batch_encode_plus(batch, add_special_tokens=True)[
            "input_ids"
        ]
        for tl in token_lists:
            incr.add_token_list(tl)


def build_multi_indexes(
    base_index_path: str,
    datasets: List[str],
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
    hf_token: str,
    sizes: List[int],
    allow_extend: bool,
    per_run_cap: Optional[int],
):
    incr = IncrementalWriters(
        base_index_path, sizes, tokenizer, allow_extend, per_run_cap
    )
    for item in datasets:
        spec = resolve_item(item)
        print(f"→ Processing {item} …")
        if spec["kind"] == "npz":
            for seq in load_npz_sequences(spec["path"]):
                incr.add_token_list(seq)
        elif spec["kind"] == "jsonl":
            write_texts_incremental(
                incr, iter_jsonl_texts(spec["path"]), batch_size, tokenizer
            )
        else:
            reader_fn = READERS[spec["reader"]]
            write_texts_incremental(
                incr, reader_fn(spec["repo"], hf_token), batch_size, tokenizer
            )
    incr.finalize()


# ---- CLI ----


def main():
    parser = argparse.ArgumentParser(
        description="Datastore creation utility (keyword-driven)."
    )
    parser.add_argument(
        "--index_file_path",
        type=str,
        required=True,
        help="Path to the output index file (or base for multi-incremental).",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Tokenizer/model path or name"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help=(
            "Space-separated list of dataset keywords, .npz files, and/or .jsonl files. "
            "Keywords: " + ", ".join(sorted(KEYWORD_MAP.keys()))
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help="Batch size for tokenization",
    )
    parser.add_argument(
        "--extend-index",
        action="store_true",
        help="If set and index file(s) exist, append new entries to the existing index(es).",
    )
    parser.add_argument(
        "--extend-by",
        type=str,
        help="Add only N tokens this run (e.g., '100k', '1m'). Applies to single or multi mode.",
    )
    parser.add_argument(
        "--stack-token",
        type=str,
        help="If not set, cannot use the Stack dataset for code data.",
    )
    parser.add_argument(
        "--multi-incremental",
        action="store_true",
        help="Build cumulative sub-datastores at token cutoffs (default: 100k,1m,10m,100m,1g).",
    )
    parser.add_argument(
        "--multi-sizes",
        type=str,
        help="Comma-separated sizes like '100k,1m,10m'. Implies --multi-incremental.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Determine mode
    do_multi = args.multi_incremental or (args.multi_sizes is not None)
    sizes = (
        DEFAULT_SIZES if args.multi_sizes is None else parse_sizes_arg(args.multi_sizes)
    )

    # Extend cap
    extend_cap = None
    if args.extend_by:
        extend_cap = parse_size_atom(args.extend_by)
        if extend_cap <= 0:
            print("Extend cap (--extend-by) must be positive.")
            return

    # Preflight existence checks for single-index mode
    if not do_multi:
        if os.path.exists(args.index_file_path) and not args.extend_index:
            print(
                f"Index file {args.index_file_path} already exists.\n"
                f"To extend it, pass --extend-index. To rebuild, delete the file first."
            )
            return
        os.makedirs(os.path.dirname(args.index_file_path) or ".", exist_ok=True)

    start = time.time()

    if not do_multi:
        build_index(
            args.index_file_path,
            args.datasets,
            batch_size=args.batch_size,
            tokenizer=tokenizer,
            hf_token=args.stack_token,
            extend_cap=extend_cap,
        )
        print(
            f"Index file {args.index_file_path} "
            f"{'extended' if args.extend_index else 'created and written to disk.'}"
        )
    else:
        base = args.index_file_path
        # For multi, IncrementalWriters handles existence/errors per derived path
        build_multi_indexes(
            base,
            args.datasets,
            batch_size=args.batch_size,
            tokenizer=tokenizer,
            hf_token=args.stack_token,
            sizes=sizes,
            allow_extend=args.extend_index,
            per_run_cap=extend_cap,
        )
        derived = [derive_index_path(base, s) for s in sorted(set(sizes))]
        print("Built/updated incremental indexes:")
        for p in derived:
            print(f"  - {p}")

    minutes = (time.time() - start) / 60.0
    print(f"Time taken: {minutes:.2f} minutes")


if __name__ == "__main__":
    main()
