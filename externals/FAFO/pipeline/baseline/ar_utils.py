"""Shared helpers for the baseline (normal auto-regressive) evaluations.

The baseline loads the model through pipeline/baseline/ar (our own modeling +
loading path, patched onto Transformers the same way FAFO patches its own) and
runs plain auto-regressive decoding: no lookahead/Jacobi guesses and no KV-cache
compression. Keeping the model identical to FAFO makes the throughput/quality
comparison apples-to-apples: only the decoding algorithm differs.
"""

import time
import logging

import torch
from fastchat.model import get_conversation_template

from pipeline.baseline.ar.utils import load_model, build_prompt as build_model_prompt
from pipeline.baseline.ar import get_device

logger = logging.getLogger("main")

# Llama-2 chat models use a FastChat conversation template; every other model
# uses the tokenizer's own chat template (via build_model_prompt).
_CONV_TEMPLATE_MODELS = {
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
}


def load_ar_model(pipeline_config):
    """Load the model + tokenizer for plain auto-regressive decoding (via ar/)."""
    model, tokenizer = load_model(pipeline_config=pipeline_config)
    model.tokenizer = tokenizer
    return model, tokenizer


@torch.inference_mode()
def run_ar_eval(model_id, questions, model, tokenizer, pipeline_config, eval_config, get_turns):
    """Generic AR eval loop.

    Args:
        get_turns: callable(question) -> list[str] of raw user messages (one per
                   conversation turn) to feed the model.
    """
    max_new_tokens = pipeline_config.get("n_new_tokens", eval_config.get("max_new_tokens", 1024))
    do_sample_cfg = bool(pipeline_config.get("do_sample", 0))
    temperature = pipeline_config.get("temperature", 0.7) if do_sample_cfg else 0.0
    do_sample = temperature >= 1e-4

    overall_time = 0.0
    overall_tp = 0.0
    overall_gen = 0
    count_gen = 0
    step = 1

    for question_idx, question in enumerate(questions):
        torch.manual_seed(step)
        conv = None
        conversation = []
        if pipeline_config["model_name"] in _CONV_TEMPLATE_MODELS:
            conv = get_conversation_template(model_id)

        turns = []
        prompts = []
        for turn_message in get_turns(question):
            if conv is not None:
                conv.append_message(conv.roles[0], turn_message)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
            else:
                prompt, conversation = build_model_prompt(
                    pipeline_config["model_name"], tokenizer, turn_message, conversation
                )
            prompts.append(prompt)
            input_ids = torch.as_tensor(tokenizer([prompt]).input_ids).cuda()
            attention_mask = torch.ones_like(input_ids)

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                pad_token_id=(
                    tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )
            if do_sample:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=1.0)
            else:
                gen_kwargs.update(do_sample=False)

            start_time = time.time()
            # Plain auto-regressive generation (stock HF decode; no lookahead).
            output_ids = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
            gap_time = time.time() - start_time

            tokens = output_ids.numel() - len(input_ids[0])
            # Skip the first question (warmup) so one-time kernel compilation /
            # autotuning is not counted; kept fair vs FAFO which does the same.
            if question_idx > 0 or len(questions) == 1:
                overall_time += gap_time
                overall_gen += tokens
                overall_tp += tokens / gap_time
                count_gen += 1

            if model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(input_ids[0]):]

            # Respect the conversation template's stop tokens (Llama-2 path).
            if conv is not None and conv.stop_token_ids:
                stop_idx = [i for i, tid in enumerate(output_ids) if tid in conv.stop_token_ids]
                if len(stop_idx) > 0:
                    output_ids = output_ids[: stop_idx[0]]

            output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
            if conv is not None and conv.stop_str and output.find(conv.stop_str) > 0:
                output = output[: output.find(conv.stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            if conv is not None:
                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
                conv.messages[-1][-1] = output
            else:
                conversation.append({"role": "assistant", "content": output})

            turns.append(output)

        logger.info(f"question {question_idx}: turns={len(turns)}")

    avg_throughput_1 = overall_tp / count_gen if count_gen else 0.0
    avg_throughput_2 = overall_gen / overall_time if overall_time else 0.0
    if get_device() == 0:
        logger.info(
            f"AVERAGE THROUGHPUT1 {avg_throughput_1} AVERAGE THROUGHPUT2 {avg_throughput_2} "
            f"STAT {[overall_tp, count_gen, overall_gen, overall_time]}"
        )

    processed_result = {
        # No KV-cache compression / lookahead in the baseline -> ratio is 1.
        "avg_compression_ratio": 1.0,
        "avg_throughput_1": avg_throughput_1,
        "avg_throughput_2": avg_throughput_2,
    }
    raw_result = ""
    return processed_result, raw_result
