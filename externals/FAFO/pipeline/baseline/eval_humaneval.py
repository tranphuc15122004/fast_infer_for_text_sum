import logging

logger = logging.getLogger("main")

from human_eval.data import read_problems
from ar_utils import run_ar_eval, load_ar_model


def generate_prompt(input):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


    ### Instruction:
    Create a Python script for this problem:
    {input}

    ### Response:"""
    return INSTRUCTION


def eval_humaneval(config):
    pipeline_config = config["pipeline_params"]
    eval_config = config["eval_params"]
    model_id = pipeline_config["model_name"]

    # Load data (HumanEval problems, identical prompt to the FAFO eval).
    questions = list(read_problems().values())

    # Load a plain HF baseline model.
    model, tokenizer = load_ar_model(pipeline_config)

    return run_ar_eval(
        model_id,
        questions,
        model,
        tokenizer,
        pipeline_config,
        eval_config,
        get_turns=lambda q: [generate_prompt(q["prompt"])],
    )
