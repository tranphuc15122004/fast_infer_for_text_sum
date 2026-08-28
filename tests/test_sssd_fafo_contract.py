from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_vendored_repositories_are_not_nested_git_repositories():
    assert not (ROOT / "externals/SSSD/.git").exists()
    assert not (ROOT / "externals/SSSD/sssd_speculator/.git").exists()
    assert not (ROOT / "externals/FAFO/.git").exists()


def test_dispatcher_and_launchers_register_sssd_and_fafo():
    dispatcher = (ROOT / "scripts/run.sh").read_text(encoding="utf-8")
    assert 'sssd)       WRAPPER="scripts/run_sssd.sh"' in dispatcher
    assert 'fafo)       WRAPPER="scripts/run_fafo.sh"' in dispatcher
    assert "sssd" in dispatcher.split("Available:", 1)[1]
    assert "fafo" in dispatcher.split("Available:", 1)[1]
    for name in ("run_sssd.sh", "run_fafo.sh"):
        launcher = ROOT / "scripts" / name
        assert launcher.is_file()
        text = launcher.read_text(encoding="utf-8")
        assert 'source "$ROOT/scripts/common/config.sh"' in text
        assert 'source "$ROOT/scripts/common/runtime.sh"' in text
        assert '"$FAST_INFER_PYTHON"' in text


def test_shared_config_exposes_sssd_and_fafo_namespaces():
    config = (ROOT / "scripts/common/config.sh").read_text(encoding="utf-8")
    for marker in (
        "fast_infer__load_sssd()",
        "fast_infer__load_fafo()",
        "sssd) fast_infer__load_sssd",
        "fafo) fast_infer__load_fafo",
    ):
        assert marker in config


def test_sssd_command_uses_the_forked_sglang_entrypoint():
    module = _load_script("infer_sssd.py")
    command = module.build_command(
        python="/venv/bin/python",
        model="/models/llama",
        dataset="/tmp/custom.jsonl",
        result_file="/tmp/result.json",
        max_new_tokens=16,
        datastore_path="/tmp/sssd.idx",
        num_draft_tokens=8,
        num_steps=5,
        topk=5,
        adaptive=True,
    )
    assert command[:3] == ["/venv/bin/python", "-m", "sglang.bench_offline_throughput"]
    assert "--speculative-algorithm" in command
    assert command[command.index("--speculative-algorithm") + 1] == "SSSD"
    assert command[command.index("--model-path") + 1] == "/models/llama"
    assert command[command.index("--dataset-name") + 1] == "custom"
    assert "--speculative-adaptive" in command


def test_fafo_command_uses_upstream_main_and_single_sample_configs():
    module = _load_script("infer_fafo.py")
    command = module.build_command(
        python="/venv/bin/python",
        pipeline_config="/tmp/pipeline.json",
        eval_config="/tmp/eval.json",
        output_dir="/tmp/fafo-out",
        exp_desc="smoke",
    )
    assert command[:2] == ["/venv/bin/python", "pipeline/fafo/main.py"]
    assert command[command.index("--pipeline_config_dir") + 1] == "/tmp/pipeline.json"
    assert command[command.index("--eval_config_dir") + 1] == "/tmp/eval.json"
    assert command[command.index("--output_folder_dir") + 1] == "/tmp/fafo-out"

    pipeline = module.build_pipeline_config("meta-llama/Llama-3.1-8B-Instruct", 16, "stream-llm")
    assert pipeline["pipeline_params"]["fafo"] is True
    assert pipeline["pipeline_params"]["n_new_tokens"] == 16
    assert pipeline["pipeline_params"]["kv_cache_method"] == "stream-llm"

    evaluation = module.build_eval_config("/tmp/one-sample.jsonl")
    assert evaluation["eval_params"]["dataset"] == "gsm8k"
    assert evaluation["eval_params"]["dataset_path"] == "/tmp/one-sample.jsonl"


def test_master_example_documents_llama_sssd_fafo_defaults():
    config = (ROOT / "docs/fast_infer_master.example.env").read_text(encoding="utf-8")
    for marker in (
        "SSSD_DATASTORE_PATH",
        "SSSD_NUM_DRAFT_TOKENS",
        "FAFO_KV_METHOD",
        "FAFO_MAX_NEW_TOKENS",
    ):
        assert marker in config


def test_baseline_docs_record_upstream_revision_and_gpu_constraints():
    for baseline in ("sssd", "fafo"):
        text = (ROOT / f"docs/baselines/{baseline}.md").read_text(encoding="utf-8")
        assert "commit" in text.lower()
        assert "Llama 3.1" in text
        assert "GPU" in text
