import logging

logger = logging.getLogger("main")

from eval.gsm8k_utils.gsm8k_utils import load_gsm8k, build_prompt
from ar_utils import run_ar_eval, load_ar_model


def eval_gsm8k(config):
    pipeline_config = config["pipeline_params"]
    eval_config = config["eval_params"]
    model_id = pipeline_config["model_name"]

    # Load data (few-shot CoT prompt, identical to the FAFO eval).
    questions = load_gsm8k(eval_config["dataset_path"])

    # Load a plain HF baseline model.
    model, tokenizer = load_ar_model(pipeline_config)

    return run_ar_eval(
        model_id,
        questions,
        model,
        tokenizer,
        pipeline_config,
        eval_config,
        get_turns=lambda q: [build_prompt(q["instruction"])],
    )
