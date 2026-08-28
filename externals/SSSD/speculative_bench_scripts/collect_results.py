"""Gathers all benchmark results and compacts them into a single JSON file."""

import argparse
import json
import os
import re

from sglang.utils import get_timestamp_str, read_json, save_json


def read_all_json_files(dir_path: str, sorting_fn=None) -> tuple[list[dict], list[str]]:
    """
    Reads all JSON files from a specified directory and returns a list of their contents.

    Args:
        dir_path (str): The path to the directory containing the JSON files.

    Returns:
        list: A list containing the parsed JSON data from each file.
    """
    json_data = []
    filenames = [
        f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
    ]
    if sorting_fn:
        filenames = sorted(filenames, key=sorting_fn)

    # Iterate over all files in the directory
    for filename in filenames:
        if filename.endswith(".json"):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    json_data.append(data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")

    return json_data, filenames


def get_name_and_batch_size(filename: str) -> tuple[str, int]:
    m = re.match(r"([a-zA-Z0-9\-]+)_([0-9]+)(?:_.*)?", filename)
    base = m.group(1)
    batch_size = int(m.group(2))

    return (base, batch_size)


def process_evaluation_results(
    eval_results: list[dict], eval_filenames: list[str]
) -> dict:
    results = {}
    for filename, result in zip(eval_filenames, eval_results):
        method_name, batch_size = get_name_and_batch_size(filename)
        if method_name not in results:
            results[method_name] = {}
        results[method_name][f"batch_{batch_size}"] = result
    return results


def parse_args() -> dict:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gather benchmark results into a single JSON file."
    )
    parser.add_argument(
        "--hparam-benchmark",
        type=str,
        default="sharegpt",
        help="The benchmark to use for hyperparameter search (default: sharegpt).",
    )
    parser.add_argument(
        "--eval-benchmark",
        type=str,
        default="mt-bench",
        help="The benchmark to use for evaluation (default: mt-bench).",
    )
    parser.add_argument(
        "--submission-url",
        type=str,
        default="",
        help="The URL to submit the results to.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the directory containing the data files.",
    )
    args = parser.parse_args()
    return vars(args)


def gather_results():
    results_dict = {}
    args = parse_args()
    benchmark = args["eval_benchmark"]
    hparam_benchmark = args["hparam_benchmark"]
    data_path = args["data_path"]

    # evaluation results
    eval_results, eval_filenames = read_all_json_files(
        f"{data_path}/evaluation/{benchmark}", get_name_and_batch_size
    )
    results_dict["evaluation"] = {}
    results_dict["evaluation"][benchmark] = process_evaluation_results(
        eval_results, eval_filenames
    )

    # hyperparameters
    folder = f"{data_path}/hyperparameter_search/{hparam_benchmark}"
    if os.path.exists(folder):
        hyperparam_results, _ = read_all_json_files(folder, get_name_and_batch_size)
        hyperparam_d = {}

        for r in hyperparam_results:
            alg = r.pop("algorithm")
            batch = f"batch_{r.pop('batch_size')}"
            if alg not in hyperparam_d:
                hyperparam_d[alg] = {}
            hyperparam_d[alg][batch] = r

        if "hyperparameter_search" not in results_dict:
            results_dict["hyperparameter_search"] = {}
        results_dict["hyperparameter_search"][hparam_benchmark] = hyperparam_d
    else:
        print(f"Skipping hyperparameter saving, folder does not exist.")

    # machine specs
    results_dict["machine_specs"] = (read_json(f"{data_path}/machine_specs.json"),)

    # saving
    results_dict["submission_timestamp"] = get_timestamp_str()
    results_dict["submission_url"] = args["submission_url"]
    output_path = f"{data_path}/collected_results.json"
    save_json(output_path, results_dict, sort_keys=False)
    print(f"All Results saved to {output_path}.")


if __name__ == "__main__":
    gather_results()
