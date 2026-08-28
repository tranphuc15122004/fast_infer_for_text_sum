import random
import numpy as np
import logging
logger = logging.getLogger("main")

from transformers import GenerationMixin, AutoTokenizer
from transformers.models.llama import modeling_llama
from transformers.models.qwen2 import modeling_qwen2
from fastchat.model.model_adapter import Llama2Adapter, Llama3Adapter, QwenChatAdapter

from pipeline.fafo.decoding import greedy_search_proxy, FUNC_MAP, CONFIG_MAP, get_device
from pipeline.fafo.models import modeling_llama as fafo_modeling_llama
from pipeline.fafo.models import modeling_qwen2 as fafo_modeling_qwen2
from pipeline.fafo.models.modeling_llama import LlamaForCausalLM
import torch
import inspect

def config_fafo(config, WINDOW_SIZE=None, LEVEL=None, GUESS_SET_SIZE=None, ALWAYS_FWD_ONE=None, SPLIT_FLAG=None, DIST_WORKERS=None, POOL_FROM_PROMPT=None, backend = 'nccl', USE_FLASH=None):
    CONFIG_MAP['config'] = config
    if WINDOW_SIZE is not None:
        CONFIG_MAP["WINDOW_SIZE"] = WINDOW_SIZE
    if LEVEL is not None:
        CONFIG_MAP["LEVEL"] = LEVEL
    if GUESS_SET_SIZE is not None:
        CONFIG_MAP["GUESS_SET_SIZE"] = GUESS_SET_SIZE
    if ALWAYS_FWD_ONE is not None:
        CONFIG_MAP["ALWAYS_FWD_ONE"] = ALWAYS_FWD_ONE
    if SPLIT_FLAG is not None:
        CONFIG_MAP["SPLIT_FLAG"] = SPLIT_FLAG
    if POOL_FROM_PROMPT is not None:
        CONFIG_MAP["POOL_FROM_PROMPT"] = POOL_FROM_PROMPT
    if USE_FLASH is not None:
        CONFIG_MAP["USE_FLASH"] = USE_FLASH

    CONFIG_MAP["log"] = []


def inject_module(fafo_module, original_module):
    s = {}
    for name, cls in inspect.getmembers(original_module, inspect.isclass):
        s[name] = cls 
    for name, cls in inspect.getmembers(fafo_module, inspect.isclass):
        if str(cls.__module__).startswith("pipeline.fafo") and name in s:
            tc = s[name]
            for method_name in dir(cls):
                if callable(getattr(cls, method_name)):
                    try:
                        setattr(tc, method_name, getattr(cls, method_name))
                    except:
                        pass 


def augment_llama(config):
    inject_module(fafo_modeling_llama, modeling_llama)
def augment_qwen2(config):
    inject_module(fafo_modeling_qwen2, modeling_qwen2)

def augment_generate(config):
    FUNC_MAP["greedy_search"] = GenerationMixin.generate
    GenerationMixin.generate = greedy_search_proxy

def augment_all(config):
    augment_llama(config)
    augment_qwen2(config)
    augment_generate(config)

def log_history(clear=False):
    gen = 0
    step = 0    
    if "log" in CONFIG_MAP:
        for log in CONFIG_MAP["log"]:
            gen += log[0]
            step += log[1]
    if clear:
        CONFIG_MAP["log"] = []
    logger.info(f"FAFO LOG - OVERALL GEN: {gen} STEPS: {step} AVG COMPRESS RATIO: {(gen / step) if step > 0 else 0}")

    return (gen / step) if step > 0 else 0


def lock_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_model(pipeline_config):
    model_path = pipeline_config['model_name']
    device = f"cuda:{get_device()}"
    use_flash = pipeline_config['use_flash']

    # get model adapter
    adapter = None
    if model_path == "meta-llama/Llama-2-7b-chat-hf" or model_path == "meta-llama/Llama-2-13b-chat-hf" or (model_path == "NousResearch/Yarn-Llama-2-7b-128k" and not pipeline_config['fafo']) or model_path == "codellama/CodeLlama-13b-hf":
        adapter = Llama2Adapter()
    elif model_path == "meta-llama/Meta-Llama-3-8B-Instruct" or model_path == "meta-llama/Llama-3.1-8B-Instruct" or model_path == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":
        adapter = Llama3Adapter()
    elif model_path == "Qwen/Qwen2.5-7B-Instruct" or model_path == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" or model_path == "Qwen/Qwen2.5-32B-Instruct":
        adapter = QwenChatAdapter()

    kwargs = {"torch_dtype": torch.float16, "revision": "main"}
    if use_flash:
        kwargs["use_flash_attention_2"] = use_flash

    # Load model
    if adapter is not None:
        model, tokenizer = adapter.load_model(model_path, kwargs)
    else:
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    if pipeline_config['fafo']:
        model.init_kv_cache(pipeline_config)

    model.to(device)
    logger.info(model)

    return model, tokenizer
