import os
import time
import logging
from tqdm import tqdm

logger = logging.getLogger("main")

import torch
from torch.nn.attention.flex_attention import flex_attention
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
import pipeline.fafo
from pipeline.fafo import (
    get_device,
    augment_all,
    config_fafo
)
from pipeline.fafo.utils import (
    load_model
)
from pipeline.fafo.flex_masking.inference_mask import INFERENCE_MASKS
from pipeline.fafo.models.utils import build_prompt

flex_attention = torch.compile(flex_attention)


def run_eval(
    model_id,
    questions,
    model,
    tokenizer,
    pipeline_config,
    eval_config
):
    num_gpus_total=pipeline_config['num_gpus_total']
    num_gpus_per_model=pipeline_config['num_gpus_per_model']
    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    processed_result, raw_result = {}, {}
    for i in range(0, len(questions), chunk_size):
        processed_result, raw_result = get_model_answers(
            model_id,
            questions[i : i + chunk_size],
            model=model,
            tokenizer=tokenizer,
            pipeline_config=pipeline_config,
            eval_config=eval_config
        )

    return processed_result, raw_result

@torch.inference_mode()
def get_model_answers(
    model_id,
    questions,
    model,
    tokenizer,
    pipeline_config,
    eval_config
):  
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    
    logger.info(f"configuration: flash attn: {pipeline_config['use_flash']} HF PP: {pipeline_config['use_pp']} DS TP: {pipeline_config['use_tp_ds']} GPUS: {devices}")

    ds_local_rank = pipeline_config['local_rank']
    
    if 'fafo' in pipeline_config and pipeline_config['fafo']: 
        CONFIG_MAP = pipeline.fafo.decoding.CONFIG_MAP
        decoding_mask = INFERENCE_MASKS[CONFIG_MAP['config']['kv_cache_method']](CONFIG_MAP)
        CONFIG_MAP['decoding_mask'] = decoding_mask
        CONFIG_MAP['flex_attention'] = flex_attention

    overall_time = 0
    overall_tp = 0
    overall_gen = 0
    count_gen = 0
    stats = {}
    if not pipeline_config['do_sample']:
        temperature = 0.0 #force greedy
    else:
        temperature = pipeline_config.get('temperature', 0.7)
    step = 1

    for question_idx, question in enumerate(tqdm(questions)):
        stats[question_idx] = {} #
        choices = []
        torch.manual_seed(step)
        conv = None
        if pipeline_config['model_name'] == 'meta-llama/Llama-2-7b-chat-hf' or pipeline_config['model_name'] == 'meta-llama/Llama-2-13b-chat-hf':
            conv = get_conversation_template(model_id)
        else:
            conversation = []
        turns = []
        prompts = []

        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            if conv:
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
            else:
                prompt, conversation = build_prompt(pipeline_config['model_name'], tokenizer, qs, conversation)
            prompts.append(prompt)
            input_ids = tokenizer([prompt]).input_ids
            do_sample = temperature >= 1e-4 
            start_time = time.time()
            if pipeline_config['fafo']:
                output_ids = model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=pipeline_config['n_new_tokens'],
                    top_k=0.0, top_p=1.0,
                    config=pipeline_config
                )
            else:
                output_ids = model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=pipeline_config['n_new_tokens'],
                    top_k=0.0, top_p=1.0
                )
            end_time = time.time()
            gap_time = end_time - start_time 
            tokens = output_ids.numel() - len(input_ids[0])
            # The first question warms up torch.compile / flex-attention kernels;
            # exclude it from the throughput measurement (unless it is the only one).
            if question_idx > 0 or len(questions) == 1:
                overall_time += gap_time
                overall_gen += tokens
                overall_tp += tokens / gap_time
                count_gen += 1

            stats[question_idx][j] = [gap_time, tokens]
            if get_device() == 0 and ds_local_rank == 0:
                logger.info([f"step {step} turn {j} time: ", gap_time, " generated tokens: ", tokens, " throughput: " , tokens / gap_time])
            
            if model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(input_ids[0]) :]

            # be consistent with the template's stop_token_ids
            if conv is not None and conv.stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in conv.stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
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

        choices.append({"index": step, "turns": turns, "prompts" : prompts})



    if get_device() == 0:
        logger.info(f"AVERAGE THROUGHPUT1 {overall_tp / count_gen} AVERAGE THROUGHPUT2 {overall_gen / overall_time} STAT {[overall_tp, count_gen, overall_gen, overall_time]}")
        avg_compression_ratio = pipeline.fafo.log_history()
        avg_throughput_1 = overall_tp / count_gen
        avg_throughput_2 = overall_gen / overall_time

    processed_result = {
        'avg_compression_ratio' : avg_compression_ratio,
        'avg_throughput_1' : avg_throughput_1,
        'avg_throughput_2' : avg_throughput_2
    }
    if 'fafo' in pipeline_config and pipeline_config['fafo']: 
        raw_result = CONFIG_MAP.get('log', "")
    else:
        raw_result = ""

    return processed_result, raw_result

def eval_mtbench(config):
    pipeline_config = config['pipeline_params']
    eval_config = config['eval_params']
    model_id = f"{pipeline_config['model_name']}-level-{pipeline_config['level']}-win-{pipeline_config['window']}-guess-{pipeline_config['num_guesses']}"
    if pipeline_config['fafo']:
        augment_all(pipeline_config)
        config_fafo(config=pipeline_config, LEVEL=pipeline_config['level'], WINDOW_SIZE=pipeline_config['window'], GUESS_SET_SIZE=pipeline_config['num_guesses'], USE_FLASH=pipeline_config['use_flash'], DIST_WORKERS=len(os.environ.get("CUDA_VISIBLE_DEVICES").split(",")))
        logger.info(f"FAFO activated config: {pipeline.fafo.decoding.CONFIG_MAP}")

    # Load data
    questions = load_questions(eval_config['dataset_path'], None, None)
    # Load model
    model, tokenizer = load_model(pipeline_config=pipeline_config)
    model.tokenizer = tokenizer

    processed_result, raw_result = run_eval(
        model_id=model_id,
        questions=questions,
        model=model,
        tokenizer=tokenizer,
        pipeline_config=pipeline_config,
        eval_config=eval_config
    )

    return processed_result, raw_result
