import argparse
import itertools
import logging
import multiprocessing as mp
import os
import statistics
from dataclasses import asdict
from functools import partial

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from sglang.bench_offline_throughput import BenchArgs, throughput_test
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.sssd_utils import default_branch_func
from sglang.utils import get_timestamp_str, read_json, save_json


def objective(
    trial: optuna.trial.Trial,
    base_server_args: ServerArgs,
    bench_args: BenchArgs,
    search_space: dict,
    k_runs: int,
    batch_size: int,
    max_draft_tokens_in_batch: int,
) -> float:
    """
    The objective function for Optuna to optimize.
    A single trial consists of:
    1. Suggesting a set of hyper-parameters.
    2. Running the benchmark k_runs times for these parameters.
    3. Returning the median latency.
    """
    # 1. Suggest hyper-parameters from the search space

    params = {
        "speculative_eagle_topk": trial.suggest_int(
            "speculative_eagle_topk",
            search_space["speculative_eagle_topk"][0],
            search_space["speculative_eagle_topk"][-1],
        ),
        "speculative_num_draft_tokens": trial.suggest_int(
            "speculative_num_draft_tokens",
            search_space["speculative_num_draft_tokens"][0],
            search_space["speculative_num_draft_tokens"][-1],
        ),
    }

    # For SSSD we derive speculative_num_steps from the number of draft tokens
    if base_server_args.speculative_algorithm == "SSSD":
        params["speculative_num_steps"], params["speculative_eagle_topk"] = (
            default_branch_func(params["speculative_num_draft_tokens"])
        )
    else:
        params["speculative_num_steps"] = trial.suggest_int(
            "speculative_num_steps",
            search_space["speculative_num_steps"][0],
            search_space["speculative_num_steps"][-1],
        )

    # 2. Handle constraints by pruning invalid trials.
    if params["speculative_num_steps"] >= params["speculative_num_draft_tokens"]:
        raise optuna.exceptions.TrialPruned(
            f"Pruning trial: speculative_num_steps cannot be >= speculative_num_draft_tokens. Params: {params}"
        )

    if params["speculative_eagle_topk"] >= params["speculative_num_draft_tokens"]:
        raise optuna.exceptions.TrialPruned(
            f"Pruning trial: speculative_eagle_topk cannot be >= than speculative_num_draft_tokens. Params: {params}"
        )

    if params["speculative_num_draft_tokens"] * batch_size > max_draft_tokens_in_batch:
        raise optuna.exceptions.TrialPruned(
            f"Pruning trial: Total draft tokens per batch exceeds the limit of {max_draft_tokens_in_batch}. Params: {params}"
        )

    # EAGLE3 specific constraints
    if (
        base_server_args.speculative_algorithm == "EAGLE3"
        or base_server_args.speculative_algorithm == "EAGLE"
    ):
        if (
            params["speculative_eagle_topk"] * (params["speculative_num_steps"] - 1) + 1
            < params["speculative_num_draft_tokens"]
        ):
            raise optuna.exceptions.TrialPruned(
                f"Pruning trial: Total tree size is smaller than draft tokens in batch. Params: {params}"
            )

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Testing with: {params}")

    latencies_for_trial = []
    acceptance_lengths_for_trial = []
    for k in range(k_runs):
        print(f"  Run {k+1}/{k_runs}...")

        # Set up output directory for results
        result_dir = "/tmp/sssd"
        result_path = os.path.join(result_dir, "hyperparameter_result.json")
        os.makedirs(result_dir, exist_ok=True)
        bench_args.result_filename = result_path
        if os.path.exists(result_path):
            os.remove(result_path)

        try:
            # Run benchmark as separate process
            current_args = ServerArgs(**asdict(base_server_args) | params)
            p = mp.Process(target=throughput_test, args=(current_args, bench_args))
            p.start()
            p.join()

            result = read_json(result_path)

            latency = result["total_latency"]
            latencies_for_trial.append(latency)

            # Also record the acceptance length
            if "avg_acceptance_length" in result["extra_metrics"]:
                acceptance_lengths_for_trial.append(
                    result["extra_metrics"]["avg_acceptance_length"]
                )
            print(f"    Latency for this run: {latency:.4f}")

        except Exception as e:
            # Handle runtime errors (e.g., CUDA out of memory).
            # Returning a large value tells Optuna this was a bad trial.
            print(f"    Run {k+1}/{k_runs} failed with an error: {e}")
            print(f"  Trial {trial.number} failed. Reporting high latency to Optuna.")
            return float("inf")

    if not latencies_for_trial:
        print(f"  Skipping trial {trial.number} as all runs failed.")
        return float("inf")

    # Calculate the median latency for the current trial
    median_latency = statistics.median(latencies_for_trial)

    if acceptance_lengths_for_trial:
        median_idx = latencies_for_trial.index(median_latency)
        median_acceptance_length = acceptance_lengths_for_trial[median_idx]
        trial.set_user_attr("acceptance_length", median_acceptance_length)

    print(
        f"  Median Latency for trial {trial.number}: {median_latency:.4f} (from runs: {latencies_for_trial})"
    )

    return median_latency


def run_optuna_search(
    algorithm_name,
    base_server_args,
    bench_args,
    search_space,
    k_runs,
    batch_size,
    max_draft_tokens_in_batch,
    num_successful_trials,
) -> optuna.Study:
    """
    Sets up and runs the Optuna optimization study.
    """
    print(
        f"\n--- Starting Optuna Search for {algorithm_name} ({num_successful_trials} successful trials, {k_runs} runs each) ---"
    )

    # Use functools.partial to pass the static arguments to the objective function
    objective_with_args = partial(
        objective,
        base_server_args=base_server_args,
        bench_args=bench_args,
        search_space=search_space,
        k_runs=k_runs,
        batch_size=batch_size,
        max_draft_tokens_in_batch=max_draft_tokens_in_batch,
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective_with_args,
        callbacks=[
            MaxTrialsCallback(num_successful_trials, states=(TrialState.COMPLETE,))
        ],
    )

    print(f"\n--- {algorithm_name} Optuna Search Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")

    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )
    fail_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.FAIL]
    )

    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Failed trials: {len(fail_trials)}")
    print(f"  Successful trials: {len(complete_trials)}")

    if study.best_trial is None or study.best_trial.value == float("inf"):
        print("\nNo successful trials were completed to determine the best parameters.")
    else:
        print("\n--- Best Result ---")
        print(f"Best Median Latency: {study.best_trial.value:.4f}")
        if "acceptance_length" in study.best_trial.user_attrs:
            print(
                f"Acceptance Length {study.best_trial.user_attrs['acceptance_length']}"
            )

        print("Best Hyper-parameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")

    return study


def run_hyperparameter_search(
    server_args: ServerArgs,
    bench_args: BenchArgs,
    batch_sizes: list[int],
    max_draft_tokens_in_batch: int,
    sample_size: int,
    num_successful_trials: int,
    out_dir: str,
):
    # TODO
    # Alternative datasets for datastore:
    # https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.1-Pro-1M-v0.1
    # https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.1-Pro-500K-Filtered

    mp.set_start_method("spawn", force=True)
    speculative_algorithm = server_args.speculative_algorithm
    print(f"\n--- Running Hyperparameter Search for {speculative_algorithm} ---")
    for batch_size in batch_sizes:
        num_prompts = max(50, 2 * batch_size)

        out_path = os.path.join(
            out_dir, f"{speculative_algorithm}_{batch_size}_results.json"
        )
        if os.path.exists(out_path):
            print(
                f"Results for {speculative_algorithm} with batch size {batch_size} already exist. Skipping..."
            )
            continue

        extra_server_args = {
            "cuda_graph_max_bs": batch_size,
            "max_running_requests": batch_size,
        }
        extra_bench_args = {
            "num_prompts": num_prompts,
        }

        if speculative_algorithm == "EAGLE3" or speculative_algorithm == "EAGLE":
            base_args = ServerArgs(**asdict(server_args) | extra_server_args)
            # Define the search space for Optuna (min, max values)
            search_space = {
                "speculative_num_steps": [1, 7],
                "speculative_eagle_topk": [1, 5],
                "speculative_num_draft_tokens": [1, 32],
            }
        elif speculative_algorithm == "SSSD":
            base_args = ServerArgs(**asdict(server_args) | extra_server_args)
            # Define the search space for Optuna (min, max values)
            search_space = {
                "speculative_eagle_topk": [3, 5],
                "speculative_num_draft_tokens": [1, 32],
            }
        else:
            raise ValueError(
                "Unsupported speculative algorithm. Choose either 'EAGLE3', 'EAGLE' or 'SSSD'."
            )

        logging.basicConfig(
            level=getattr(logging, base_args.log_level.upper()),
            format="%(message)s",
        )

        # Common Benchmarking Arguments
        bench_args = BenchArgs(**asdict(bench_args) | extra_bench_args)

        # Run the optimization
        study_result = run_optuna_search(
            algorithm_name=speculative_algorithm,
            base_server_args=base_args,
            bench_args=bench_args,
            search_space=search_space,
            k_runs=sample_size,
            batch_size=batch_size,
            max_draft_tokens_in_batch=max_draft_tokens_in_batch,
            num_successful_trials=num_successful_trials,
        )

        assert study_result.best_trial is not None, "No successful trials found."

        d = {
            "algorithm": speculative_algorithm,
            "search_space": search_space,
            "best_latency": study_result.best_trial.value,
            "batch_size": batch_size,
            "best_parameters": study_result.best_params,
            "timestamp": get_timestamp_str(),
        }

        if "acceptance_length" in study_result.best_trial.user_attrs:
            d["acceptance_length"] = study_result.best_trial.user_attrs[
                "acceptance_length"
            ]

        os.makedirs(out_dir, exist_ok=True)
        save_json(out_path, d)


def add_hyperparameter_search_args(parser: argparse.ArgumentParser):
    """Adds arguments for the hyperparameter search script."""
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 48, 64],
        help="A comma-separated list of batch sizes to run the search for. (e.g., 1 4 8 16)",
    )
    parser.add_argument(
        "--max-draft-tokens-in-batch",
        type=int,
        default=512,
        help="The maximum number of draft tokens allowed in a single batch.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Number of samples to take for each trial in the hyperparameter search. (TODO: set to 3)",
    )
    parser.add_argument(
        "--num-successful-trials",
        type=int,
        default=15,
        help="Total number of successful trials for Optuna to run. (TODO: set to 15)",
    )
    parser.add_argument(
        "--hparam-output-dir",
        type=str,
        default="data/hyperparameter_search",
        help="Directory to save the results JSON file.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    add_hyperparameter_search_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    # SGLang tends to overallocate the KVCache, leading to occasional CUDA OOM errors
    server_args.mem_fraction_static *= 0.95

    run_hyperparameter_search(
        server_args,
        bench_args,
        batch_sizes=args.batch_sizes,
        max_draft_tokens_in_batch=args.max_draft_tokens_in_batch,
        sample_size=args.sample_size,
        num_successful_trials=args.num_successful_trials,
        out_dir=args.hparam_output_dir,
    )
