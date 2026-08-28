import logging

logger = logging.getLogger("main")

from fastchat.llm_judge.common import load_questions
from ar_utils import run_ar_eval, load_ar_model


def eval_mtbench(config):
    pipeline_config = config["pipeline_params"]
    eval_config = config["eval_params"]
    model_id = pipeline_config["model_name"]

    # Load data (multi-turn MT-Bench questions).
    questions = load_questions(eval_config["dataset_path"], None, None)

    # Load a plain HF baseline model.
    model, tokenizer = load_ar_model(pipeline_config)

    return run_ar_eval(
        model_id,
        questions,
        model,
        tokenizer,
        pipeline_config,
        eval_config,
        get_turns=lambda q: q["turns"],
    )
