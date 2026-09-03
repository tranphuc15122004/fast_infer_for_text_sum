from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_trajectory_train_and_infer_cli(tmp_path: Path) -> None:
    cache = tmp_path / "trajectory.jsonl"
    train_dir = tmp_path / "train"
    output = tmp_path / "infer.jsonl"
    build = subprocess.run(
        [sys.executable, "scripts/build_syncspec_trajectories.py", "--backend", "synthetic",
         "--output", str(cache), "--samples", "2", "--max-new-tokens", "5",
         "--include-logits"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    trajectory = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    assert trajectory[0]["metadata"]["dtype"] == "bfloat16"
    assert trajectory[0]["metadata"]["seed"] == 42
    assert trajectory[0]["metadata"]["target_artifact_fingerprint"]
    train = subprocess.run(
        [sys.executable, "scripts/train_syncspec.py", "--stage", "diffusion",
         "--data", str(cache), "--output-dir", str(train_dir), "--steps", "1",
         "--device", "cpu", "--vocab-size", "256", "--hidden-size", "8", "--layers", "1", "--heads", "2",
         "--groups", "2", "--kd", "4", "--kl-weight", "0.1",
         "--rank-weight", "0.1", "--rank-margin", "0.1", "--rank-top-m", "4",
         "--train-batch-size", "1"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert train.returncode == 0, train.stdout + train.stderr
    assert (train_dir / "pytorch_model.bin").is_file()
    assert json.loads(train.stdout.strip().splitlines()[-1])["batch_size"] == 1
    infer = subprocess.run(
        [sys.executable, "scripts/infer_syncspec.py", "--backend", "synthetic",
         "--smoke", "--output", str(output), "--device", "cpu"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert infer.returncode == 0, infer.stdout + infer.stderr
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[0]["method"] == "syncspec"
    assert records[0]["status"] == "ok"
    sys.path.insert(0, str(ROOT / "scripts"))
    from common.io_util import validate_schema
    assert validate_schema(records[0], spec=True) == []
    assert records[-1]["record_count"] == 1


def test_synthetic_torch_trajectory_cache_reaches_training_cli(tmp_path: Path) -> None:
    cache = tmp_path / "trajectory.pt"
    train_dir = tmp_path / "train"
    build = subprocess.run(
        [sys.executable, "scripts/build_syncspec_trajectories.py", "--backend", "synthetic",
         "--output", str(cache), "--samples", "1", "--max-new-tokens", "3"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    train = subprocess.run(
        [sys.executable, "scripts/train_syncspec.py", "--stage", "diffusion",
         "--data", str(cache), "--output-dir", str(train_dir), "--steps", "1",
         "--device", "cpu", "--vocab-size", "256", "--hidden-size", "8",
         "--layers", "1", "--heads", "2", "--groups", "2", "--kd", "2"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert train.returncode == 0, train.stdout + train.stderr
    assert (train_dir / "pytorch_model.bin").is_file()


def test_synthetic_infer_cli_runs_real_microbatch(tmp_path: Path) -> None:
    output = tmp_path / "batch-output.jsonl"
    proc = subprocess.run(
        [sys.executable, "scripts/infer_syncspec.py", "--backend", "synthetic",
         "--device", "cpu", "--max-samples", "2", "--batch-size", "2",
         "--max-new-tokens", "3", "--output", str(output)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["batch_size"] for row in rows[:2]] == [2, 2]
    assert rows[-1]["record_count"] == 2


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available on this host")
def test_synthetic_infer_cli_rejects_unavailable_cuda(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/infer_syncspec.py", "--backend", "synthetic",
         "--device", "cuda", "--output", str(tmp_path / "cuda.jsonl")],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode != 0
    assert "cuda" in (proc.stdout + proc.stderr).lower()


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available on this host")
def test_trajectory_cli_rejects_unavailable_cuda(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/build_syncspec_trajectories.py", "--backend", "synthetic",
         "--device", "cuda", "--output", str(tmp_path / "cuda.pt")],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode != 0
    assert "cuda" in (proc.stdout + proc.stderr).lower()


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available on this host")
def test_train_cli_rejects_unavailable_cuda(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/train_syncspec.py", "--stage", "diffusion",
         "--data", str(tmp_path / "missing.pt"), "--output-dir", str(tmp_path / "train"),
         "--device", "cuda"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode != 0
    assert "cuda" in (proc.stdout + proc.stderr).lower()


def test_synthetic_infer_cli_honors_explicit_budget(tmp_path: Path) -> None:
    output = tmp_path / "budget-output.jsonl"
    proc = subprocess.run(
        [sys.executable, "scripts/infer_syncspec.py", "--backend", "synthetic",
         "--device", "cpu", "--max-samples", "1", "--max-new-tokens", "3",
         "--kd", "6", "--kv", "3", "--output", str(output)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["budgets"][0] == {"kd": 6, "kv": 3}


def test_infer_budget_profiles_expose_adaptive_transformers_set() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from infer_syncspec import _budget_profiles

    profiles = _budget_profiles(
        "transformers", None, None,
        "8:4,8:8,16:4,16:8,16:12,16:16",
    )
    assert [(profile.kd, profile.kv) for profile in profiles] == [
        (0, 0), (8, 4), (8, 8), (16, 4), (16, 8), (16, 12), (16, 16),
    ]


def test_synthetic_infer_cli_exactness_check_records_vanilla_match(tmp_path: Path) -> None:
    output = tmp_path / "exactness-output.jsonl"
    proc = subprocess.run(
        [sys.executable, "scripts/infer_syncspec.py", "--backend", "synthetic",
         "--device", "cpu", "--max-samples", "1", "--max-new-tokens", "3",
         "--check-exactness", "--output", str(output)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["exactness_checked"] is True
    assert rows[0]["exact_match_vanilla_ar"] is True
    assert rows[-1]["exactness_failures"] == 0


def test_synthetic_infer_cli_rejects_stochastic_exactness_check(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/infer_syncspec.py", "--backend", "synthetic",
         "--device", "cpu", "--stochastic", "--check-exactness",
         "--output", str(tmp_path / "out.jsonl")],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode != 0
    assert "stochastic" in (proc.stdout + proc.stderr).lower()


def test_trajectory_cli_honors_sample_limit_for_input_jsonl(tmp_path: Path) -> None:
    data = tmp_path / "two.jsonl"
    data.write_text(
        '{"id":"one","source_ids":[1,2]}\n'
        '{"id":"two","source_ids":[3,4]}\n',
        encoding="utf-8",
    )
    output = tmp_path / "limited.jsonl"
    proc = subprocess.run(
        [sys.executable, "scripts/build_syncspec_trajectories.py", "--backend", "synthetic",
         "--input", str(data), "--samples", "1", "--max-new-tokens", "1",
         "--output", str(output)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["sample_id"] == "one"


def test_infer_cli_rejects_network_backend_without_local_model(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/infer_syncspec.py", "--backend", "transformers",
         "--target-model", str(tmp_path / "missing"), "--output", str(tmp_path / "out.jsonl"),
         "--device", "cpu"], cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode != 0
    assert "not found" in (proc.stdout + proc.stderr).lower()


def test_transformers_infer_requires_trained_selector_and_survival(tmp_path: Path) -> None:
    (tmp_path / "target").mkdir()
    (tmp_path / "drafter").mkdir()
    proc = subprocess.run(
        [sys.executable, "scripts/infer_syncspec.py", "--backend", "transformers",
         "--target-model", str(tmp_path / "target"),
         "--drafter-checkpoint", str(tmp_path / "drafter"),
         "--output", str(tmp_path / "out.jsonl"), "--device", "cpu"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode != 0
    assert "selector-checkpoint" in (proc.stdout + proc.stderr)


def test_selector_training_requires_diffusion_checkpoint(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/train_syncspec.py", "--stage", "selector",
         "--data", str(tmp_path / "missing.jsonl"),
         "--output-dir", str(tmp_path / "selector"), "--device", "cpu"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode != 0
    assert "init-checkpoint" in (proc.stdout + proc.stderr)


def test_survival_training_requires_selector_checkpoint(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/train_syncspec.py", "--stage", "survival",
         "--init-checkpoint", str(tmp_path / "drafter"),
         "--data", str(tmp_path / "missing.jsonl"),
         "--output-dir", str(tmp_path / "survival"), "--device", "cpu"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode != 0
    assert "selector-checkpoint" in (proc.stdout + proc.stderr)


def test_selector_checkpoint_rejects_target_vocabulary_mismatch(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from SyncSpec.config import SyncSpecConfig
    from SyncSpec.selector import SourceCoherentSelector
    from infer_syncspec import _load_trained_components

    selector_dir = tmp_path / "selector"
    selector_dir.mkdir()
    selector = SourceCoherentSelector(4, rank=2, ngram_dim=6, vocab_size=8)
    torch.save(selector.state_dict(), selector_dir / "selector.pt")
    (selector_dir / "selector_config.json").write_text(
        json.dumps({"hidden_size": 4, "rank": 2, "ngram_dim": 6, "vocab_size": 8}),
        encoding="utf-8",
    )
    args = SimpleNamespace(selector_checkpoint=str(selector_dir), survival_checkpoint=None)
    with pytest.raises(ValueError, match="vocabulary"):
        _load_trained_components(args, SyncSpecConfig(vocab_size=9, hidden_size=4))


def test_syncspec_record_adds_rouge_for_reference() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from infer_syncspec import _record
    from SyncSpec.schema import InferenceResult
    import torch

    class Tokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            del token_ids, skip_special_tokens
            return "a concise summary"

    result = InferenceResult(
        token_ids=torch.tensor([1, 2]), committed_tokens=2,
        budgets=[{"kd": 4, "kv": 2}], accepted_lengths=[2],
        timing_ms={"e2e": 1.0, "prefill": 0.1, "draft": 0.1, "verify": 0.1},
    )
    record = _record(
        result, torch.tensor([1, 2, 3]), "toy", Tokenizer(),
        {"dataset": "xsum", "reference": "a concise summary"},
    )
    assert record["dataset"] == "xsum"
    assert record["rouge1"] == 1.0
    assert record["rouge2"] == 1.0
    assert record["rougeL"] == 1.0


def test_transformers_backend_runs_with_local_tiny_llama_and_tokenizer(tmp_path: Path) -> None:
    transformers = __import__("transformers")
    tokenizers = __import__("tokenizers")
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from SyncSpec.model import SyncSpecDrafter, SyncSpecDrafterConfig

    target_dir = tmp_path / "target"
    drafter_dir = tmp_path / "drafter"
    tokenizer_core = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "Summarize": 4,
         "the": 5, "following": 6, "document": 7, "tiny": 8}, unk_token="<unk>"
    ))
    tokenizer_core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_core, pad_token="<pad>", bos_token="<bos>",
        eos_token="<eos>", unk_token="<unk>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        "{% if add_generation_prompt %}{% endif %}"
    )
    tokenizer.save_pretrained(target_dir)
    target = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=9, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    ))
    target.save_pretrained(target_dir)
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({
        "id": "tiny", "dataset": "xsum", "document": "tiny document",
        "reference": "tiny document",
    }) + "\n", encoding="utf-8")
    trajectory = tmp_path / "trajectory.jsonl"
    build = subprocess.run(
        [sys.executable, "scripts/build_syncspec_trajectories.py", "--backend", "transformers",
         "--target-model", str(target_dir), "--input", str(data), "--output", str(trajectory),
         "--device", "cpu", "--dtype", "float32", "--max-new-tokens", "2",
         "--include-target-features", "--include-source-memory", "--local-files-only"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    trajectory_rows = [json.loads(line) for line in trajectory.read_text(encoding="utf-8").splitlines()]
    assert trajectory_rows[0]["source_memory"]
    assert trajectory_rows[0]["metadata"]["source_memory_source"] == "target_final_hidden"
    train = subprocess.run(
        [sys.executable, "scripts/train_syncspec.py", "--stage", "diffusion",
         "--target-model", str(target_dir), "--data", str(trajectory),
         "--output-dir", str(drafter_dir), "--device", "cpu", "--dtype", "float32",
         "--kd", "2", "--steps", "1", "--local-files-only"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert train.returncode == 0, train.stdout + train.stderr
    assert (drafter_dir / "pytorch_model.bin").is_file()
    drafter_config = json.loads((drafter_dir / "config.json").read_text(encoding="utf-8"))
    assert drafter_config["max_positions"] == 64
    selector_dir = tmp_path / "selector"
    selector_train = subprocess.run(
        [sys.executable, "scripts/train_syncspec.py", "--stage", "selector",
         "--target-model", str(target_dir), "--init-checkpoint", str(drafter_dir),
         "--data", str(trajectory), "--output-dir", str(selector_dir),
         "--device", "cpu", "--dtype", "float32", "--kd", "2", "--steps", "1",
         "--local-files-only"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert selector_train.returncode == 0, selector_train.stdout + selector_train.stderr
    assert (selector_dir / "selector.pt").is_file()
    survival_dir = tmp_path / "survival"
    survival_train = subprocess.run(
        [sys.executable, "scripts/train_syncspec.py", "--stage", "survival",
         "--target-model", str(target_dir), "--init-checkpoint", str(drafter_dir),
         "--data", str(trajectory), "--output-dir", str(survival_dir),
         "--device", "cpu", "--dtype", "float32", "--kd", "2", "--steps", "1",
         "--selector-checkpoint", str(selector_dir),
         "--local-files-only"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert survival_train.returncode == 0, survival_train.stdout + survival_train.stderr
    assert (survival_dir / "survival.pt").is_file()
    output = tmp_path / "output.jsonl"
    proc = subprocess.run(
        [sys.executable, "scripts/infer_syncspec.py", "--backend", "transformers",
         "--target-model", str(target_dir), "--drafter-checkpoint", str(drafter_dir),
         "--input", str(data), "--output", str(output), "--device", "cpu",
         "--dtype", "float32", "--max-samples", "1", "--max-new-tokens", "2",
         "--selector-checkpoint", str(selector_dir),
         "--survival-checkpoint", str(survival_dir), "--check-exactness",
         "--local-files-only"], cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["status"] == "ok"
    assert rows[0]["dataset"] == "xsum"
    assert rows[0]["exactness_checked"] is True
    assert rows[0]["exact_match_vanilla_ar"] is True
    assert rows[-1]["exactness_failures"] == 0
    assert rows[-1]["record_count"] == 1
    profile = tmp_path / "profile.json"
    profiled = subprocess.run(
        [sys.executable, "scripts/profile_syncspec.py", "--backend", "transformers",
         "--target-model", str(target_dir), "--drafter-checkpoint", str(drafter_dir),
         "--selector-checkpoint", str(selector_dir), "--survival-checkpoint", str(survival_dir),
         "--input", str(data), "--output", str(profile), "--device", "cpu",
         "--dtype", "float32", "--repeats", "1", "--warmup-runs", "0",
         "--kd", "2", "--kv", "1", "--local-files-only"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert profiled.returncode == 0, profiled.stdout + profiled.stderr
    profile_payload = json.loads(profile.read_text(encoding="utf-8"))
    assert profile_payload["key"]["model"] == str(target_dir)
    assert profile_payload["key"]["selector_checkpoint"] == str(selector_dir)
    assert profile_payload["key"]["survival_checkpoint"] == str(survival_dir)
    assert "verify" in profile_payload["measurements_ms"]
