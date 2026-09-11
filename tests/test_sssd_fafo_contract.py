from __future__ import annotations

import ast
import importlib.util
import sys
import warnings
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
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--random-seed") + 1] == "42"


def test_sssd_runtime_does_not_shadow_installed_native_extension(monkeypatch):
    module = _load_script("infer_sssd.py")
    import os

    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join((str(module.SSSD_SPECULATOR), "/tmp/keep-this-path")),
    )
    entries = module._runtime_env()["PYTHONPATH"].split(os.pathsep)

    assert str(module.SSSD_PYTHON) in entries
    assert str(module.SSSD_SPECULATOR) not in entries
    assert "/tmp/keep-this-path" in entries


def test_sssd_native_kernel_is_declared_for_the_server_runtime():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    sglang_kernel_lines = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip().startswith("sglang-kernel")
    ]
    assert any("0.4.2" in line for line in sglang_kernel_lines)
    assert "\ngguf" in f"\n{requirements}"


def test_sssd_preflight_rejects_broken_native_kernel(monkeypatch):
    import common.longbench_adapter as adapter

    monkeypatch.setattr(
        adapter,
        "_module_importable",
        lambda name: (False, "ImportError: libnvrtc.so.12 is missing"),
    )

    result = adapter.preflight_baseline(
        "sssd",
        config={"model": "/models/llama"},
        cuda_available=True,
    )

    assert result["status"] == "missing_dependency"
    assert "libnvrtc.so.12" in result["reason"]


def test_sssd_preflight_accepts_inplace_speculator_build(monkeypatch, tmp_path):
    """Preflight mirrors infer_sssd: an in-tree .so build must not be reported
    as missing_dependency just because it is not pip-installed in the shared
    runtime."""
    import common.longbench_adapter as adapter

    speculator_dir = tmp_path / "sssd_speculator"
    package_dir = speculator_dir / "sssd_speculator"
    package_dir.mkdir(parents=True)
    (package_dir / "sssd_speculator.cpython-312-x86_64-linux-gnu.so").write_text("x")

    def fake_module_importable(name):
        if name == "sgl_kernel":
            return True, None
        return False, "sssd_speculator is not installed"

    monkeypatch.setattr(adapter, "_module_importable", fake_module_importable)
    monkeypatch.setattr(adapter, "SSSD_SPECULATOR", speculator_dir)

    from types import SimpleNamespace

    monkeypatch.setattr(
        adapter,
        "subprocess",
        SimpleNamespace(
            run=lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )

    result = adapter.preflight_baseline(
        "sssd",
        config={"model": "/models/llama", "python": "/venv/bin/python"},
        cuda_available=True,
    )

    # An empty datastore turns a ready SSSD cell into ``aggregate_only``, which
    # the runner treats the same as ``ready`` for launching the child.
    assert result["status"] in ("ready", "aggregate_only")
    assert result["requirements"]["sssd_speculator"]["available"] is True


def test_sssd_preflight_rejects_broken_inplace_speculator_build(monkeypatch, tmp_path):
    import common.longbench_adapter as adapter

    speculator_dir = tmp_path / "sssd_speculator"
    package_dir = speculator_dir / "sssd_speculator"
    package_dir.mkdir(parents=True)
    (package_dir / "sssd_speculator.cpython-312-x86_64-linux-gnu.so").write_text("x")

    def fake_module_importable(name):
        if name == "sgl_kernel":
            return True, None
        return False, "sssd_speculator is not installed"

    monkeypatch.setattr(adapter, "_module_importable", fake_module_importable)
    monkeypatch.setattr(adapter, "SSSD_SPECULATOR", speculator_dir)

    from types import SimpleNamespace

    monkeypatch.setattr(
        adapter,
        "subprocess",
        SimpleNamespace(
            run=lambda *a, **k: SimpleNamespace(
                returncode=1, stdout="", stderr="ImportError: wrong ELF class"
            )
        ),
    )

    result = adapter.preflight_baseline(
        "sssd",
        config={"model": "/models/llama", "python": "/venv/bin/python"},
        cuda_available=True,
    )

    assert result["status"] == "missing_dependency"
    assert "wrong ELF class" in result["reason"]
    assert result["requirements"]["sssd_speculator"]["available"] is False


def test_fafo_mask_cache_builds_contexts_beyond_static_limit():
    fafo_root = ROOT / "externals" / "FAFO"
    if str(fafo_root) not in sys.path:
        sys.path.insert(0, str(fafo_root))

    from pipeline.fafo.flex_masking.mask_cache import LazyBlockMaskCache

    calls = []
    masks = LazyBlockMaskCache(
        lambda kv_len: calls.append(kv_len) or f"mask-{kv_len}",
    )

    assert masks[84] == "mask-10880"
    assert masks[84] == "mask-10880"
    assert calls == [10880]


def test_fafo_mask_cache_builds_exact_non_block_aligned_kv_length():
    fafo_root = ROOT / "externals" / "FAFO"
    if str(fafo_root) not in sys.path:
        sys.path.insert(0, str(fafo_root))

    from pipeline.fafo.flex_masking.mask_cache import LazyBlockMaskCache

    calls = []
    masks = LazyBlockMaskCache(
        lambda kv_len: calls.append(kv_len) or f"mask-{kv_len}",
    )

    assert masks.for_length(11508) == "mask-11508"
    assert masks.for_length(11508) == "mask-11508"
    assert calls == [11508]


def test_fafo_expand_mask_has_no_transformers_deprecation_warning():
    fafo_root = ROOT / "externals" / "FAFO"
    if str(fafo_root) not in sys.path:
        sys.path.insert(0, str(fafo_root))

    import torch
    from pipeline.fafo.models import modeling_llama, modeling_qwen2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for module in (modeling_llama, modeling_qwen2):
            expanded = module._expand_mask(
                torch.ones((1, 3), dtype=torch.bool), torch.float32
            )
            assert expanded.shape == (1, 1, 3, 3)

    assert not any("deprecated" in str(item.message).lower() for item in caught)


def test_fafo_custom_generate_calls_return_tensors():
    """FAFO's patched decoder returns a tensor, not a GenerateOutput object."""
    for name in ("eval_gsm8k.py", "eval_mtbench.py", "eval_humaneval.py"):
        path = ROOT / "externals/FAFO/pipeline/fafo" / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        generate_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate"
        ]
        assert generate_calls, f"no model.generate call found in {path}"
        for call in generate_calls:
            assert any(
                keyword.arg == "return_dict_in_generate"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
                for keyword in call.keywords
            ), f"{path}:{call.lineno} must request tensor output"


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

    pipeline = module.build_pipeline_config(
        "meta-llama/Llama-3.1-8B-Instruct", 16, "stream-llm", max_input_tokens=1024
    )
    assert pipeline["pipeline_params"]["fafo"] is True
    assert pipeline["pipeline_params"]["n_new_tokens"] == 16
    assert pipeline["pipeline_params"]["kv_cache_method"] == "stream-llm"
    assert pipeline["pipeline_params"]["max_input_tokens"] == 1024

    evaluation = module.build_eval_config("/tmp/one-sample.jsonl", max_new_tokens=16)
    assert evaluation["eval_params"]["dataset"] == "gsm8k"
    assert evaluation["eval_params"]["dataset_path"] == "/tmp/one-sample.jsonl"
    assert evaluation["eval_params"]["max_new_tokens"] == 16


def test_fafo_parser_forwards_smoke_context_limit():
    module = _load_script("infer_fafo.py")
    args = module._parser().parse_args(
        ["--max-input-tokens", "1024", "--max-new-tokens", "8", "--smoke", "--output", "x"]
    )

    assert args.max_input_tokens == 1024


def test_fafo_smoke_budget_matches_repository_smoke_budget():
    module = _load_script("infer_fafo.py")

    assert module.resolve_smoke_budget(32) == 8
    assert module.resolve_smoke_budget(4) == 4


def test_fafo_parser_accepts_aggregate_summary_log():
    module = _load_script("infer_fafo.py")
    parsed = module._parse_log(
        """
AGE THROUGHPUT1 1.415505116185281 AVERAGE THROUGHPUT2 1.415505116185281 STAT [1.415505116185281, 1, 16, 11.303385496139526]
FAFO LOG - OVERALL GEN: 16 STEPS: 9 AVG COMPRESS RATIO: 1.7777777777777777
"""
    )

    assert parsed["output_tokens"] == 16
    assert parsed["e2e_s"] == 11.303385496139526
    assert parsed["throughput"] == 1.415505116185281


def test_fafo_parser_ignores_hidden_warmup_in_overall_token_count():
    module = _load_script("infer_fafo.py")
    parsed = module._parse_log(
        ""
        "AVERAGE THROUGHPUT2 1.4 STAT [1.4, 1, 8, 5.7]\n"
        "FAFO LOG - OVERALL GEN: 32 STEPS: 18 AVG COMPRESS RATIO: 1.7\n"
    )

    assert parsed["output_tokens"] == 8
    assert parsed["e2e_s"] == 5.7


def test_fafo_budget_validation_rejects_lookahead_tokens():
    module = _load_script("infer_fafo.py")

    assert module.generated_tokens_within_budget(8, 8)
    assert not module.generated_tokens_within_budget(16, 8)


def test_fafo_smoke_adds_hidden_compile_warmup_record():
    module = _load_script("infer_fafo.py")

    records = [{"id": "sample-1", "prompt": "hello", "reference": None}]
    runtime_records = module.prepare_fafo_records(records, smoke=True)

    assert len(runtime_records) == 2
    assert runtime_records[0]["id"].startswith("__fafo_warmup__")
    assert runtime_records[1]["id"] == "sample-1"


def test_fafo_representative_single_sample_also_adds_hidden_compile_warmup():
    module = _load_script("infer_fafo.py")

    records = [{"id": "sample-1", "prompt": "hello", "reference": None}]
    runtime_records = module.prepare_fafo_records(records, smoke=False)

    assert len(runtime_records) == 2
    assert runtime_records[1]["id"] == "sample-1"


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
