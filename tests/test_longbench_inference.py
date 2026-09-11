import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_longbench_loader_maps_master_values(tmp_path):
    master = tmp_path / "master.env"
    master.write_text(
        'LONG_BENCH_DATA_DIR="data/fixture"\n'
        'LONG_BENCH_OUTPUT_DIR="outputs/fixture"\n'
        'LONG_BENCH_MODEL="/models/llama"\n'
        'LONG_BENCH_BASELINES="vanilla_hf vanilla_fa"\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/common/config.sh; "
            "fast_infer_load_config longbench; "
            "printf '%s\\n' \"$LONG_BENCH_MODEL\" \"$LONG_BENCH_BASELINES\"",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "FAST_INFER_MASTER_CONFIG": str(master),
            "FAST_INFER_PYTHON": sys.executable,
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "/models/llama" in result.stdout
    assert "vanilla_hf vanilla_fa" in result.stdout


def test_longbench_launcher_uses_master_and_shared_runtime():
    text = (ROOT / "scripts/run_longbench_200.sh").read_text(encoding="utf-8")
    assert "fast_infer_load_config longbench" in text
    assert "scripts/common/runtime.sh" in text
    assert "run_longbench_200.py" in text


def test_evaluation_defaults_use_longbench_100_14k_profile(monkeypatch):
    monkeypatch.delenv("LONG_BENCH_DATA_DIR", raising=False)
    monkeypatch.delenv("LONG_BENCH_OUTPUT_DIR", raising=False)

    from run_longbench_200 import _parser

    args = _parser().parse_args([])
    assert str(args.data_dir) == "data/longbench_100_14k"
    assert str(args.output_dir) == "outputs/longbench_100_14k"

    expected_defaults = {
        ROOT / "scripts/common/config.sh": (
            'fast_infer_default LONG_BENCH_DATA_DIR "data/longbench_100_14k"',
            'fast_infer_default LONG_BENCH_OUTPUT_DIR "outputs/longbench_100_14k"',
        ),
        ROOT / "scripts/collect_metrics.py": (
            'ROOT / "outputs" / "longbench_100_14k"',
            'ROOT / "data" / "longbench_100_14k"',
        ),
        ROOT / "scripts/show_longbench_200.py": (
            'ROOT / "data" / "longbench_100_14k"',
        ),
        ROOT / "scripts/modal_longbench.py": (
            'REMOTE_ROOT / "data" / "longbench_100_14k"',
            'VOLUME_MOUNT / "outputs" / "longbench_100_14k"',
        ),
    }
    for path, snippets in expected_defaults.items():
        text = path.read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet in text, f"missing new default in {path}: {snippet}"


def test_child_log_is_streamed_before_baseline_exits(tmp_path):
    from run_longbench_200 import _run_child

    marker = tmp_path / "child-ready"
    release = tmp_path / "release-child"
    log_path = tmp_path / "baseline.log"
    code = (
        "from pathlib import Path; import time; "
        "print('first runtime line'); "
        f"Path({str(marker)!r}).write_text('ready'); "
        f"release=Path({str(release)!r}); "
        "\nwhile not release.exists(): time.sleep(0.01); "
        "print('second runtime line')"
    )
    result_holder = {}

    def run_child():
        result_holder["value"] = _run_child(
            [sys.executable, "-c", code],
            output=tmp_path / "child-output.jsonl",
            log_path=log_path,
            timeout_seconds=10,
        )

    thread = threading.Thread(target=run_child)
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), "child did not reach its streaming checkpoint"

        deadline = time.monotonic() + 2
        streamed = False
        while time.monotonic() < deadline:
            if log_path.is_file() and "first runtime line" in log_path.read_text(
                encoding="utf-8"
            ):
                streamed = True
                break
            time.sleep(0.01)
        assert streamed, "baseline log was not updated while child was running"
    finally:
        release.write_text("release", encoding="utf-8")
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert result_holder["value"]["status"] == "success"
    assert "second runtime line" in log_path.read_text(encoding="utf-8")


def test_measure_call_returns_elapsed_and_output():
    from common.benchmark_runtime import measure_call

    value, timing = measure_call(lambda: "ok", device=torch.device("cpu"))

    assert value == "ok"
    assert timing["e2e_ms"] >= 0
    assert timing["device"] == "cpu"


def test_build_status_record_never_invents_performance_metrics():
    from common.benchmark_runtime import build_status_record

    row = build_status_record(
        method="vanilla_fa",
        dataset="lcc",
        sample_id="x",
        status="unsupported_cpu",
        reason="CUDA unavailable",
    )

    assert row["status"] == "unsupported_cpu"
    assert row["e2e_ms"] is None
    assert row["throughput_tok_s"] is None


def test_vanilla_parser_exposes_distinct_attention_defaults():
    from infer_vanilla_fa import build_parser as fa_parser
    from infer_vanilla_hf import build_parser as hf_parser

    assert hf_parser().parse_args(["--output", "x"]).attention_backend == "eager"
    assert (
        fa_parser().parse_args(["--output", "x"]).attention_backend
        == "flash_attention_2"
    )


def test_vanilla_record_contains_shared_timing_fields():
    from common.benchmark_runtime import build_sample_record

    record = build_sample_record(
        method="vanilla_hf",
        dataset="lcc",
        sample_id="id",
        model="m",
        input_tokens=10,
        output_tokens=2,
        timing={"e2e_ms": 4.0, "prefill_ms": 1.0, "decode_ms": 3.0},
        config={"attention_backend": "eager"},
        text="x",
        reference_output="y",
    )

    assert record["throughput_tok_s"] == 500.0
    assert record["attention_backend"] == "eager"


def test_smoke_defaults_to_a_bounded_context_but_full_keeps_unlimited_inputs(monkeypatch):
    import run_longbench_200

    monkeypatch.delenv("LONG_BENCH_SMOKE_MAX_INPUT_TOKENS", raising=False)
    monkeypatch.delenv("LONG_BENCH_MAX_INPUT_TOKENS", raising=False)

    assert run_longbench_200.resolve_max_input_tokens("smoke", None) == 4096
    assert run_longbench_200.resolve_max_input_tokens("full", None) == 0
    assert run_longbench_200.resolve_max_input_tokens("smoke", 0) == 0


def test_prompt_cap_preserves_both_context_head_and_instruction_suffix():
    from common.input_utils import truncate_input_ids

    ids = torch.arange(20).reshape(1, 20)
    capped = truncate_input_ids(ids, 8, suffix_tokens=3)

    assert capped.tolist() == [[0, 1, 2, 3, 4, 17, 18, 19]]
    assert capped.shape == (1, 8)


def test_prompt_cap_keeps_short_inputs_unchanged():
    from common.input_utils import truncate_input_ids

    ids = torch.arange(5).reshape(1, 5)

    assert torch.equal(truncate_input_ids(ids, 8), ids)


def test_vanilla_prompt_batch_preserves_instruction_suffix_when_capped():
    from types import SimpleNamespace

    from common.vanilla_inference import _prompt_batch

    class FakeTokenizer:
        def __call__(self, prompt, **kwargs):
            return SimpleNamespace(input_ids=torch.arange(20).reshape(1, 20))

    result = _prompt_batch(FakeTokenizer(), "document ... Summary:", max_input_tokens=8)

    assert result.tolist() == [[0, 1, 2, 3, 16, 17, 18, 19]]


def test_timed_generation_returns_output_and_synchronized_wall_time():
    from common.paired_generation import timed_generate

    class FakeModel:
        def generate(self, input_ids, **kwargs):
            extra = torch.full(
                (input_ids.shape[0], kwargs["max_new_tokens"]),
                7,
                dtype=input_ids.dtype,
            )
            return torch.cat((input_ids, extra), dim=1)

    ids = torch.tensor([[1, 2]])
    output, elapsed_ms = timed_generate(
        FakeModel(), ids, device=torch.device("cpu"), max_new_tokens=3
    )

    assert output.tolist() == [[1, 2, 7, 7, 7]]
    assert elapsed_ms >= 0.0


def test_speedup_ignores_pairs_with_different_output_lengths():
    from common.metrics import aggregate_speedup

    records = [
        {
            "dense_e2e_ms": 100.0,
            "e2e_ms": 10.0,
            "speedup_valid": False,
        },
        {
            "dense_e2e_ms": 120.0,
            "e2e_ms": 60.0,
            "speedup_valid": True,
        },
    ]

    assert aggregate_speedup(records)["esr"] == 2.0


def test_eagle_help_does_not_import_heavy_model_modules():
    result = subprocess.run(
        [sys.executable, "scripts/eagle3_infer_qwen3.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0
    assert "--base-model" in result.stdout


def test_magicdec_parser_has_unique_options():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/infer_magicdec.py",
            "--model-pth",
            "checkpoint.pth",
            "--model-name",
            "model",
            "--output",
            "out.jsonl",
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_eagle_llama3_prompt_matches_upstream_chat_contract():
    from eagle3_infer_qwen3 import EAGLE_LLAMA3_SYSTEM_PROMPT, build_eagle_messages

    messages = build_eagle_messages("Summarize this document.")

    assert messages[0] == {
        "role": "system",
        "content": EAGLE_LLAMA3_SYSTEM_PROMPT,
    }
    assert messages[1] == {
        "role": "user",
        "content": "Summarize this document.",
    }


def test_eagle_normalizes_transformers5_llama3_rope_parameters():
    from types import SimpleNamespace

    sys.path.insert(0, str(ROOT / "externals" / "EAGLE"))
    from eagle.model.ea_model import normalize_llama3_rope_config

    config = SimpleNamespace(
        rope_scaling=None,
        rope_parameters={"rope_type": "llama3", "factor": 8.0},
    )

    normalize_llama3_rope_config(config)

    assert config.rope_scaling == config.rope_parameters


def test_eagle_refreshes_zero_filled_llama3_rope_buffer():
    from transformers import LlamaConfig

    from eagle.model.modeling_llama_kv import LlamaAttention

    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
    )
    attention = LlamaAttention(config)
    assert attention._uses_hf_llama3_rope

    with torch.no_grad():
        attention.rotary_emb.inv_freq.zero_()
    attention._refresh_llama3_rope_buffers()

    assert attention.rotary_emb.inv_freq[0].item() == 1.0
    assert attention.rotary_emb.inv_freq.abs().max().item() > 0.0


def test_longbench_uses_long_form_output_and_profile_timeout_defaults(monkeypatch):
    import run_longbench_200

    monkeypatch.delenv("LONG_BENCH_MAX_NEW_TOKENS", raising=False)
    monkeypatch.delenv("LONG_BENCH_REPRESENTATIVE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LONG_BENCH_FULL_TIMEOUT_SECONDS", raising=False)

    profile = run_longbench_200.resolve_profile(
        mode="representative", cuda_available=True
    )
    assert profile["max_new_tokens"] == 2048
    assert run_longbench_200.resolve_timeout_seconds("representative") == 3600
    assert run_longbench_200.resolve_timeout_seconds("full") == 21600


def test_low_free_gpu_is_rejected_before_model_children_are_spawned():
    from run_longbench_200 import gpu_memory_guard_reason

    report = {
        "requested_ids": [0],
        "host_gpus": [
            {"index": 0, "name": "NVIDIA B200", "free_memory_gb": 9.1}
        ],
    }

    reason = gpu_memory_guard_reason(report, min_free_gb=32.0)

    assert reason is not None
    assert "GPU 0" in reason
    assert "9.1" in reason


def test_sssd_command_forwards_context_limit():
    from common.longbench_adapter import build_adapter_command

    command = build_adapter_command(
        "sssd",
        config={
            "python": "/usr/bin/python3",
            "model": "/models/llama",
            "max_input_tokens": 4096,
            "smoke": True,
        },
        data_file=ROOT / "data/longbench_200/gov_report.jsonl",
        output=ROOT / "outputs/test-sssd.jsonl",
        max_samples=1,
        max_new_tokens=8,
    )

    assert "--max-input-tokens" in command
    assert command[command.index("--max-input-tokens") + 1] == "4096"


@pytest.mark.parametrize(
    "baseline",
    [
        "vanilla_hf",
        "vanilla_fa",
        "magicdec",
        "longspec",
        "eagle3",
        "dflash",
        "specextend",
        "sssd",
        "fafo",
    ],
)
def test_all_longbench_adapters_forward_the_shared_seed(baseline):
    from common.longbench_adapter import build_adapter_command

    command = build_adapter_command(
        baseline,
        config={
            "python": "/usr/bin/python3",
            "model": "/models/llama",
            "eagle_model": "/models/eagle",
            "dflash_model": "/models/dflash",
            "longspec_target_model": "/models/llama",
            "longspec_draft_model": "/models/longspec",
            "specextend_draft_model": "/models/eagle",
            "sssd_datastore_path": "",
            "seed": 37,
            "max_input_tokens": 4096,
            "smoke": True,
        },
        data_file=ROOT / "data/longbench_200/gov_report.jsonl",
        output=ROOT / f"outputs/test-{baseline}.jsonl",
        max_samples=1,
        max_new_tokens=8,
    )

    assert "--seed" in command
    assert command[command.index("--seed") + 1] == "37"


def test_speculative_adapters_skip_internal_reference_when_external_reference_exists():
    from common.longbench_adapter import build_adapter_command

    common = {
        "python": "/usr/bin/python3",
        "model": "/models/llama",
        "eagle_model": "/models/eagle",
        "dflash_model": "/models/dflash",
        "seed": 37,
        "max_input_tokens": 4096,
        "smoke": True,
        "skip_reference": True,
    }
    data_file = ROOT / "data/longbench_200/gov_report.jsonl"
    eagle = build_adapter_command(
        "eagle3", config=common, data_file=data_file,
        output=ROOT / "outputs/test-eagle-reference.jsonl", max_samples=1,
        max_new_tokens=8,
    )
    dflash = build_adapter_command(
        "dflash", config=common, data_file=data_file,
        output=ROOT / "outputs/test-dflash-reference.jsonl", max_samples=1,
        max_new_tokens=8,
    )

    assert "--skip-naive" in eagle
    assert "--skip-reference" in dflash


def test_vanilla_generate_passes_attention_mask_to_avoid_pad_eos_ambiguity():
    from types import SimpleNamespace

    import torch

    from common.vanilla_inference import _generate

    class FakeModel:
        generation_config = SimpleNamespace(pad_token_id=2)

        def generate(self, input_ids, **kwargs):
            self.kwargs = kwargs
            return input_ids

    model = FakeModel()
    input_ids = torch.tensor([[5, 6, 2]])

    _generate(
        model,
        input_ids,
        SimpleNamespace(max_new_tokens=2, temperature=0.0),
    )

    assert torch.equal(model.kwargs["attention_mask"], torch.ones_like(input_ids))


def test_vanilla_warmup_is_bounded_without_changing_benchmark_budget():
    from types import SimpleNamespace

    from common.vanilla_inference import _warmup_args

    args = SimpleNamespace(max_new_tokens=2048, temperature=0.0)
    warmup = _warmup_args(args)

    assert args.max_new_tokens == 2048
    assert warmup.max_new_tokens == 8
    assert warmup.temperature == args.temperature


def test_magicdec_warmup_is_bounded_without_changing_benchmark_budget():
    from infer_magicdec import resolve_generation_budget

    assert resolve_generation_budget(2048) == 2048
    assert resolve_generation_budget(2048, warmup=True) == 8
    assert resolve_generation_budget(4, warmup=True) == 4


def test_dflash_repairs_invalid_generation_token_ids_from_tokenizer():
    from types import SimpleNamespace

    from infer_dflash import normalize_generation_token_ids

    config = SimpleNamespace(
        vocab_size=128256,
        bos_token_id=151643,
        eos_token_id=151645,
    )
    tokenizer = SimpleNamespace(bos_token_id=128000, eos_token_id=128001)

    changed = normalize_generation_token_ids(config, tokenizer)

    assert changed == {
        "bos_token_id": (151643, 128000),
        "eos_token_id": (151645, 128001),
    }
    assert config.bos_token_id == 128000
    assert config.eos_token_id == 128001


def test_dflash_external_reference_keeps_optional_timings_null():
    from infer_dflash import round_optional

    assert round_optional(None) is None
    assert round_optional(1.23456) == 1.235


def test_magicdec_preflight_requires_checkpoint_before_launch(monkeypatch):
    import common.longbench_adapter as adapter

    monkeypatch.setattr(adapter.importlib.util, "find_spec", lambda name: object())
    result = adapter.preflight_baseline(
        "magicdec",
        config={"model": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
        cuda_available=True,
    )

    assert result["status"] == "missing_checkpoint"
    assert "checkpoint" in result["reason"]


def test_magicdec_preflight_rejects_unimportable_flashinfer(tmp_path, monkeypatch):
    import common.longbench_adapter as adapter

    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        adapter,
        "_module_importable",
        lambda name: (False, "flashinfer native extension failed to import")
        if name == "flashinfer"
        else (True, None),
    )
    result = adapter.preflight_baseline(
        "magicdec",
        config={
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "magicdec_model_pth": str(checkpoint),
        },
        cuda_available=True,
    )

    assert result["status"] == "missing_dependency"
    assert "flashinfer" in result["reason"]


def test_specextend_preflight_rejects_slow_attention_fallback(monkeypatch):
    import common.longbench_adapter as adapter

    monkeypatch.setattr(
        adapter,
        "_module_importable",
        lambda name: (False, "flash_attn is not installed")
        if name == "flash_attn"
        else (True, None),
    )
    result = adapter.preflight_baseline(
        "specextend",
        config={"model": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
        cuda_available=True,
    )

    assert result["status"] == "missing_dependency"
    assert "fallback" in result["reason"]


def test_specextend_cache_capacity_uses_longest_prompt_and_generation_budget():
    sys.path.insert(0, str(ROOT / "externals" / "SpecExtend" / "specextend"))
    from run_eagle import resolve_cache_max_length

    assert resolve_cache_max_length(9662, 2048) == 11774


def test_vanilla_decode_attention_mask_is_preallocated_and_sliced():
    from common.vanilla_inference import _build_decode_attention_mask

    input_ids = torch.tensor([[5, 6, 7]])
    mask = _build_decode_attention_mask(input_ids, max_new_tokens=4)

    assert mask.shape == (1, 7)
    assert mask.dtype == input_ids.dtype
    assert torch.equal(mask[:, :3], torch.ones((1, 3), dtype=torch.long))
    assert torch.equal(mask[:, :4], torch.ones((1, 4), dtype=torch.long))
    # Slicing must be a view into one allocation, not a newly concatenated mask.
    assert mask[:, :4].untyped_storage().data_ptr() == mask.untyped_storage().data_ptr()


def test_vanilla_builds_static_cache_for_supported_transformers():
    from common.vanilla_inference import _build_static_cache
    from transformers import LlamaConfig

    model = torch.nn.Module()
    model.config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model.register_parameter("weight", torch.nn.Parameter(torch.empty(0)))

    cache, backend = _build_static_cache(
        model,
        max_cache_len=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert cache is not None
    assert type(cache).__name__ == "StaticCache"
    assert backend == "static"


def test_vanilla_timed_decode_reuses_static_cache_and_grows_mask_by_view():
    from types import SimpleNamespace

    from common.vanilla_inference import _timed_generate
    from transformers import LlamaConfig

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = LlamaConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
            self.register_parameter("weight", torch.nn.Parameter(torch.empty(0)))
            self.calls = []

        def forward(self, **kwargs):
            self.calls.append(kwargs)
            seq_len = int(kwargs["input_ids"].shape[1])
            logits = torch.full((1, seq_len, 11), -1.0)
            logits[..., 3] = 1.0
            return SimpleNamespace(
                logits=logits,
                past_key_values=kwargs.get("past_key_values"),
            )

    class FakeTokenizer:
        eos_token_id = None

    model = FakeModel()
    input_ids = torch.tensor([[5, 6, 7]])
    output_ids, timing = _timed_generate(
        model,
        input_ids,
        FakeTokenizer(),
        SimpleNamespace(max_new_tokens=3, temperature=0.0, dtype="float32"),
        torch.device("cpu"),
    )

    assert output_ids.shape == (1, 6)
    assert timing["kv_cache_backend"] == "static"
    assert timing["attention_mask_strategy"] == "preallocated_slice"
    assert [tuple(call["attention_mask"].shape) for call in model.calls] == [
        (1, 3),
        (1, 4),
        (1, 5),
    ]
    cache = model.calls[0]["past_key_values"]
    assert all(call["past_key_values"] is cache for call in model.calls)


def test_registry_contains_exactly_requested_baselines():
    from common.longbench_adapter import BASELINES

    assert BASELINES == (
        "vanilla_hf",
        "vanilla_fa",
        "magicdec",
        "eagle3",
        "dflash",
        "specextend",
        "fafo",
    )


def test_longbench_filters_disabled_legacy_baselines_from_old_master_defaults():
    from run_longbench_200 import _filter_matrix_baselines

    selected, skipped = _filter_matrix_baselines(
        ["vanilla_hf", "longspec", "sssd", "dflash"]
    )

    assert selected == ["vanilla_hf", "dflash"]
    assert skipped == ["longspec", "sssd"]


def test_longbench_selects_vanilla_fa_then_hf_as_external_reference(tmp_path):
    from run_longbench_200 import _select_external_reference

    vanilla_hf = tmp_path / "vanilla_hf" / "gov_report.jsonl"
    vanilla_fa = tmp_path / "vanilla_fa" / "gov_report.jsonl"
    vanilla_hf.parent.mkdir()
    vanilla_fa.parent.mkdir()
    vanilla_hf.write_text("{}\n", encoding="utf-8")

    assert _select_external_reference(
        tmp_path, "gov_report", ["vanilla_hf", "vanilla_fa"]
    ) == vanilla_hf

    vanilla_fa.write_text("{}\n", encoding="utf-8")
    assert _select_external_reference(
        tmp_path, "gov_report", ["vanilla_hf", "vanilla_fa"]
    ) == vanilla_fa


def test_longbench_joins_external_reference_metrics_by_sample_id(tmp_path):
    from run_longbench_200 import _attach_external_reference_metrics

    reference = tmp_path / "vanilla_fa.jsonl"
    speculative = tmp_path / "eagle.jsonl"
    reference.write_text(
        '{"sample_id":"x","decode_ms":100.0,"e2e_ms":200.0,"output_tokens":32}\n'
        '{"type":"summary","method":"vanilla_fa"}\n',
        encoding="utf-8",
    )
    speculative.write_text(
        '{"sample_id":"x","decode_ms":50.0,"e2e_ms":80.0,"output_tokens":32}\n'
        '{"type":"summary","method":"eagle3"}\n',
        encoding="utf-8",
    )

    attached = _attach_external_reference_metrics(
        speculative, reference, reference_baseline="vanilla_fa"
    )

    assert attached == 1
    row = json.loads(speculative.read_text(encoding="utf-8").splitlines()[0])
    assert row["dense_decode_ms"] == 100.0
    assert row["dense_e2e_ms"] == 200.0
    assert row["external_decode_speedup"] == 2.0
    assert row["external_e2e_speedup"] == 2.5
    assert row["speedup_scope"] == "external_reference"
    assert row["speedup_valid"] is True


def test_longbench_rejects_external_speedup_when_output_budgets_differ(tmp_path):
    from run_longbench_200 import _attach_external_reference_metrics

    reference = tmp_path / "vanilla_fa.jsonl"
    speculative = tmp_path / "fafo.jsonl"
    reference.write_text(
        '{"sample_id":"x","decode_ms":100.0,"e2e_ms":200.0,"output_tokens":8}\n'
        '{"type":"summary","method":"vanilla_fa"}\n',
        encoding="utf-8",
    )
    speculative.write_text(
        '{"sample_id":"x","decode_ms":50.0,"e2e_ms":80.0,"output_tokens":32}\n'
        '{"type":"summary","method":"fafo"}\n',
        encoding="utf-8",
    )

    attached = _attach_external_reference_metrics(
        speculative, reference, reference_baseline="vanilla_fa"
    )

    assert attached == 1
    row = json.loads(speculative.read_text(encoding="utf-8").splitlines()[0])
    assert row["speedup_valid"] is False
    assert "external_decode_speedup" not in row
    assert "external_e2e_speedup" not in row


def test_eagle_converter_preserves_canonical_id_and_reference(tmp_path):
    from common.longbench_adapter import convert_records_for_baseline

    output = tmp_path / "eagle.jsonl"
    convert_records_for_baseline(
        "eagle3",
        [{"id": "lcc_1", "prompt": "code", "reference": "next"}],
        output,
    )

    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["question_id"] == "lcc_1"
    assert row["turns"] == ["code"]
    assert row["answer"] == "next"


def test_orchestrator_smoke_preflight_writes_manifest_without_loading_model(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_longbench_200.py",
            "--mode",
            "smoke",
            "--preflight-only",
            "--baselines",
            "vanilla_hf",
            "--datasets",
            "lcc",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Run directory:" in result.stdout
    manifests = list(tmp_path.glob("*/run_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["mode"] == "smoke"
    assert manifest["preflight_only"] is True
    # Preflight-only runs must not attempt metric aggregation.
    assert manifest["aggregate"]["status"] == "skipped"


def test_run_collector_forwards_strict_completeness(monkeypatch):
    import run_longbench_200

    calls: dict = {}

    def fake_run_child(command, **kwargs):
        calls["command"] = list(command)
        calls["kwargs"] = kwargs
        return {
            "status": "success",
            "returncode": 0,
            "elapsed_ms": 3.0,
            "output_exists": True,
            "log": "ok",
            "log_tail": "",
            "command": list(command),
        }

    monkeypatch.setattr(run_longbench_200, "_run_child", fake_run_child)

    result = run_longbench_200._run_collector(
        Path("/out/run1"),
        Path("/data/longbench_200"),
        baselines=["vanilla_hf", "fafo"],
        datasets=["lcc", "gov_report"],
        expected_samples=200,
        strict=True,
        timeout_seconds=60,
    )

    cmd = calls["command"]
    assert cmd[1].endswith("collect_metrics.py")
    assert cmd[cmd.index("--outputs-dir") + 1] == "/out/run1"
    assert cmd[cmd.index("--data-dir") + 1] == "/data/longbench_200"
    assert "--strict" in cmd
    assert (
        cmd[cmd.index("--expected-baselines") + 1] == "vanilla_hf fafo"
    )
    assert (
        cmd[cmd.index("--expected-datasets") + 1] == "lcc gov_report"
    )
    assert cmd[cmd.index("--expected-samples") + 1] == "200"
    assert result["strict"] is True
    assert result["output_files"]["json"].endswith("metrics_summary.json")
    assert result["output_files"]["csv"].endswith("metrics_summary.csv")
    assert result["output_files"]["md"].endswith("metrics_summary.md")


def test_run_collector_is_best_effort_without_strict(monkeypatch):
    import run_longbench_200

    calls: dict = {}

    def fake_run_child(command, **kwargs):
        calls["command"] = list(command)
        return {
            "status": "success",
            "returncode": 0,
            "elapsed_ms": 2.0,
            "output_exists": True,
            "log": "ok",
            "log_tail": "",
            "command": list(command),
        }

    monkeypatch.setattr(run_longbench_200, "_run_child", fake_run_child)

    result = run_longbench_200._run_collector(
        Path("/out/run2"),
        Path("/data/longbench_200"),
        baselines=["vanilla_hf"],
        datasets=["lcc"],
        expected_samples=200,
        strict=False,
        timeout_seconds=60,
    )

    assert "--strict" not in calls["command"]
    assert result["strict"] is False


def test_full_profile_requires_cuda_unless_unsupported_is_allowed():
    from run_longbench_200 import resolve_profile

    with pytest.raises(SystemExit):
        resolve_profile(mode="full", cuda_available=False, allow_unsupported=False)


def test_collector_ignores_preflight_records_for_speed_aggregates(tmp_path):
    from collect_metrics import load_run_records

    path = tmp_path / "vanilla_fa" / "lcc.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {
            "status": "unsupported_cpu",
            "method": "vanilla_fa",
            "dataset": "lcc",
            "e2e_ms": None,
        },
        {"type": "summary", "status": "preflight_only"},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = load_run_records(tmp_path)

    assert result["coverage"]["success"] == 0


def test_code_completion_aggregate_excludes_rouge_keys():
    from collect_metrics import aggregate_run_group

    result = aggregate_run_group(
        [
            {
                "status": "success",
                "task_type": "code_completion",
                "text": "return x",
                "reference_output": "return x",
            }
        ]
    )

    assert "rouge1_f" not in result["quality"]
    assert result["quality"]["code_exact_match"] == 1.0
