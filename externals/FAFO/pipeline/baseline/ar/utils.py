"""Model loading, patching, and prompt helpers for the auto-regressive baseline."""

import os
import inspect
import logging

import torch
from transformers.models.llama import modeling_llama
from transformers.models.qwen2 import modeling_qwen2
from fastchat.model.model_adapter import Llama2Adapter, Llama3Adapter, QwenChatAdapter

from pipeline.baseline.ar.models import modeling_llama as ar_modeling_llama
from pipeline.baseline.ar.models import modeling_qwen2 as ar_modeling_qwen2

logger = logging.getLogger("main")


def get_device():
    """Local rank (0 for the single-GPU baseline)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def build_prompt(model_name, tokenizer, prompt, conversation):
    """Append a user turn and render the prompt via the tokenizer chat template."""
    if len(conversation) == 0:
        conversation.append({"role": "system", "content": "You are a useful assistant."})
    conversation.append({"role": "user", "content": prompt})
    return (
        tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True),
        conversation,
    )


def inject_module(ar_module, original_module):
    """Copy the AR modeling classes' methods onto the stock Transformers classes.

    Mirrors FAFO's ``inject_module``: for every class defined in ``ar_module``
    (guarded by the ``pipeline.baseline.ar`` module prefix) that also exists in
    the Transformers module, overwrite the Transformers class' callables with
    ours. This makes an adapter-loaded stock model run our AR forward.
    """
    s = {}
    for name, cls in inspect.getmembers(original_module, inspect.isclass):
        s[name] = cls
    for name, cls in inspect.getmembers(ar_module, inspect.isclass):
        if str(cls.__module__).startswith("pipeline.baseline.ar") and name in s:
            tc = s[name]
            for method_name in dir(cls):
                if callable(getattr(cls, method_name)):
                    try:
                        setattr(tc, method_name, getattr(cls, method_name))
                    except Exception:
                        pass


def augment_ar():
    """Patch Transformers' Llama/Qwen2 modeling with our plain-AR forward.

    Copies our modeling classes' methods (including the inherited ``from_pretrained``
    classmethod, which stays bound to our class) onto the stock Transformers classes,
    so an adapter-loaded model is built as our class and runs our AR forward on the
    legacy tuple kv-cache. Unlike FAFO we do NOT replace ``GenerationMixin.generate``
    -- plain AR uses the stock decode loop.
    """
    inject_module(ar_modeling_llama, modeling_llama)
    inject_module(ar_modeling_qwen2, modeling_qwen2)


def load_model(pipeline_config):
    """Load a model + tokenizer for plain auto-regressive decoding.

    Uses the same FastChat adapters as FAFO so the model/tokenizer setup matches,
    then patches Transformers (``augment_ar``) so the loaded model runs OUR AR
    forward on the legacy tuple kv-cache.
    """
    model_path = pipeline_config["model_name"]
    use_flash = pipeline_config.get("use_flash", False)
    device = f"cuda:{get_device()}"

    # Patch Transformers BEFORE instantiation so from_pretrained builds the modules
    # with OUR __init__ (sets num_heads/head_dim/rotary_emb/... that our forward uses),
    # mirroring FAFO which augments before load_model.
    augment_ar()

    adapter = None
    if model_path in (
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "codellama/CodeLlama-13b-hf",
        "NousResearch/Yarn-Llama-2-7b-128k",
    ):
        adapter = Llama2Adapter()
    elif model_path in (
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    ):
        adapter = Llama3Adapter()
    elif model_path in (
        "Qwen/Qwen2.5-7B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "Qwen/Qwen2.5-32B-Instruct",
    ):
        adapter = QwenChatAdapter()

    kwargs = {"torch_dtype": torch.float16, "revision": "main"}
    # Our modeling only implements the eager attention path, so force it: this
    # guarantees the loaded model uses the (LlamaAttention / Qwen2Attention)
    # classes we patch, rather than the stock SDPA/FlashAttention2 forwards
    # (which expect a `Cache` object instead of the legacy tuple).
    kwargs["attn_implementation"] = "eager"
    if use_flash:
        logger.warning(
            "use_flash is set but the AR baseline only implements the eager path; "
            "falling back to eager attention."
        )

    if adapter is not None:
        model, tokenizer = adapter.load_model(model_path, kwargs)
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.to(device)
    logger.info(model)
    return model, tokenizer
