import json
import os
import sys
import time
from pathlib import Path

import torch
from accelerate import Accelerator 
from termcolor import colored
import argparse


ROOT = Path(__file__).resolve().parents[3]


def validate_eagle3_runtime() -> None:
    """Fail early when the fp16 Llama-3.1 + EAGLE-3 pair cannot fit."""
    if not torch.cuda.is_available():
        raise RuntimeError("Llama-3.1 + EAGLE-3 smoke requires a CUDA GPU")
    minimum_gb = float(os.environ.get("SPECEXTEND_MIN_GPU_MEMORY_GB", "20"))
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total_gb < minimum_gb:
        raise RuntimeError(
            "Llama-3.1-8B-Instruct + EAGLE-3 fp16 needs more VRAM for this "
            f"path: GPU has {total_gb:.1f} GiB, minimum is {minimum_gb:.1f} GiB. "
            "Use a GPU with >=20 GiB or run the classic Vicuna config separately."
        )

def load_texts_from_jsonl(path: str, max_samples: int = None):
    """
    Load up to `max_samples` texts from a JSONL file.
    Each line must be a JSON object with a "text" key.
    """
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warning] JSON decode error at line {i}: {e}")
                continue
            text = obj.get("text")
            if text is None:
                print(f"[Warning] No 'text' field at line {i}, skipping.")
                continue
            texts.append(text)
    return texts

def main():
    parser = argparse.ArgumentParser(
        description="Run SpecExtend inference on a JSONL file of texts."
    )
    parser.add_argument(
        "--input_file", "-i",
        required=True,
        help="Path to input JSONL file (one JSON obj per line, with a 'text' field)."
    )
    parser.add_argument(
        "--max_samples", "-n",
        type=int, default=1,
        help="Maximum number of samples to read (default: 1)."
    )
    parser.add_argument(
        "--model_name", "-m",
        choices=["vicuna_7b", "longchat_7b", "llama3_1_8b"],
        default="vicuna_7b",
        help="Which base model to use (default: vicuna_7b)."
    )
    parser.add_argument(
        "--max_gen_len", "-max",
        type=int, default=256,
        help="Maximum number of tokens to generate(default: 256)."
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=int(os.environ.get("SPECEXTEND_MAX_INPUT_TOKENS", "0")),
        help="Truncate each input to this many tokens; 0 disables truncation."
    )
    parser.add_argument(
        "--use_specextend",
        action="store_true",
        help="Enable SpecExtend speculative decoding (default: False)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging from the model."
    )
    parser.add_argument(
        "--output_result_line",
        action="store_true",
        help="If set, print result line-by-line instead of as a block."
    )
    args = parser.parse_args()

    # The vendored SpecExtend EAGLE module predates EAGLE-3 and expects the
    # old ``layers.*`` checkpoint layout.  Llama-3.1 uses the official
    # EAGLE-3 implementation vendored at externals/EAGLE instead.
    use_eagle3 = args.model_name == "llama3_1_8b"
    if use_eagle3:
        validate_eagle3_runtime()
        sys.path.insert(0, str(ROOT / "externals" / "EAGLE"))
        from eagle.model import ea_model as eagle_module
        from eagle.model.ea_model import EaModel
        from shared.modeling_llama_kv_target import (
            LlamaForCausalLM as SpecExtendLlamaForCausalLM,
        )

        # Keep SpecExtend's target attention implementation (hybrid tree
        # attention and the Llama-3.1 RoPE compatibility patch), while using
        # the official EAGLE-3 draft architecture/checkpoint loader.
        eagle_module.KVLlamaForCausalLM = SpecExtendLlamaForCausalLM
    else:
        from eagle.model_eagle import EaModel

    base_model_map = {
        "vicuna_7b":  os.environ.get("SPECEXTEND_BASE_MODEL", "lmsys/vicuna-7b-v1.5-16k"),
        "longchat_7b": os.environ.get("SPECEXTEND_BASE_MODEL", "lmsys/longchat-7b-16k"),
        "llama3_1_8b": os.environ.get(
            "SPECEXTEND_BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"
        ),
    }
    draft_model_map = {
        "vicuna_7b":  os.environ.get("SPECEXTEND_DRAFT_MODEL", "jycha-98/EAGLE-vicuna-7b-v1.5-16k"),
        "longchat_7b": os.environ.get("SPECEXTEND_DRAFT_MODEL", "jycha-98/EAGLE-longchat-7b-16k"),
        "llama3_1_8b": os.environ.get(
            "SPECEXTEND_DRAFT_MODEL", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
        ),
    }

    base_model_path  = base_model_map[args.model_name]
    draft_model_path = draft_model_map[args.model_name]

    texts = load_texts_from_jsonl(args.input_file, args.max_samples)
    if not texts:
        print("No valid texts loaded; exiting.")
        return

    if use_eagle3:
        model = EaModel.from_pretrained(
            use_eagle3=True,
            base_model_path=base_model_path,
            ea_model_path=draft_model_path,
            total_token=int(os.environ.get("SPECEXTEND_TOTAL_TOKEN", "32")),
            depth=int(os.environ.get("SPECEXTEND_DEPTH", "8")),
            top_k=int(os.environ.get("SPECEXTEND_TOP_K", "4")),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()
        model.base_model.model.specextend_hybrid_tree_attn = args.use_specextend
    else:
        model = EaModel.from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=draft_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        ).eval()
    tokenizer = model.tokenizer
    accelerator = Accelerator()
    model, tokenizer = accelerator.prepare(model, tokenizer)

    def encode(text):
        kwargs = {"return_tensors": "pt", "add_special_tokens": True}
        if args.max_input_tokens > 0:
            kwargs.update({"truncation": True, "max_length": args.max_input_tokens})
        return tokenizer(text, **kwargs)["input_ids"]

    # Warmup GPUs
    print(colored(f'Warming up GPUs...', 'yellow'))
    warmup_runs = int(os.environ.get("SPECEXTEND_WARMUP_RUNS", "3"))
    for idx, text in enumerate(texts[:1]):
        input_ids = encode(text).to(accelerator.device)

        for _ in range(warmup_runs):
            if use_eagle3:
                _ = model.eagenerate(
                    input_ids,
                    temperature=0,
                    max_new_tokens=5,
                    max_length=input_ids.shape[1] + 64,
                    is_llama3=True,
                )
            else:
                _ = model.eagenerate(
                    input_ids,
                    temperature=0,
                    max_new_tokens=5,
                    output_result_line=False,
                    verbose=False,
                    use_specextend=args.use_specextend,
                    retrieval_chunk_size=32,
                    retrieve_top_k=64,
                    retrieve_every_n_steps=4,
                    retrieval_verbose=False
                )
    print(colored(f'Warmup complete!', 'yellow'))

    for idx, text in enumerate(texts):
        print(colored(f"\n=== Sample {idx+1}/{len(texts)} ===", 'yellow'))
        input_ids = encode(text).to(accelerator.device)

        if use_eagle3:
            start = time.perf_counter()
            results = model.eagenerate(
                input_ids,
                temperature=0,
                max_new_tokens=args.max_gen_len,
                max_length=input_ids.shape[1] + args.max_gen_len + 64,
                is_llama3=True,
                log=True,
                return_stats=True,
            )
            _, generated, _, decode_time, acceptance_lengths = results
            elapsed = decode_time if decode_time > 0 else time.perf_counter() - start
            tokens_per_sec = generated / elapsed if elapsed > 0 else 0.0
            print(colored(
                f"\nGenerated {generated} tokens in {elapsed:.2f}s. "
                f"\nToken/sec: {tokens_per_sec:.2f}"
                f"\nAverage acceptance length: "
                f"{(sum(acceptance_lengths) / len(acceptance_lengths)) if acceptance_lengths else 0.0:.3f}",
                "cyan",
            ))
        else:
            results = model.eagenerate(
                input_ids,
                temperature=0,
                max_new_tokens=args.max_gen_len,
                output_result_line=args.output_result_line,
                verbose=args.verbose,
                use_specextend=args.use_specextend,
                retrieval_chunk_size=32,
                retrieve_top_k=64,
                retrieve_every_n_steps=4,
                retrieval_verbose=False
            )

if __name__ == "__main__":
    main()
