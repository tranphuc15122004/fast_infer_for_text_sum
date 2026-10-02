from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "scripts" / "mr_dflash"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


class _ToyTokenizer:
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [10 + len(word) for word in str(text).split()]}

    def convert_tokens_to_ids(self, token):
        return self.eos_token_id if token == "<|im_end|>" else -1

    def apply_chat_template(
        self,
        conversation,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    ):
        del tokenize, return_dict
        role_ids = {"system": 1, "user": 2, "assistant": 3, "tool": 4}
        values = []
        for message in conversation:
            values.append(role_ids[message["role"]])
            values.extend(self(message["content"])["input_ids"])
            values.append(self.eos_token_id)
        if add_generation_prompt:
            values.append(role_ids["assistant"])
        return values


def _features(input_ids):
    ids = torch.as_tensor(input_ids, dtype=torch.float32)
    return torch.stack((ids, ids * 2 + 3), dim=-1).unsqueeze(0)


class _ToyCapture:
    def __init__(self, offset=0.0):
        self.offset = offset

    def capture_one(self, input_ids):
        return _features(input_ids) + self.offset

    def close(self):
        pass


def _row(sample_id, source, response, reference=None):
    metadata = {"generation_model": "tiny-target"}
    if reference is not None:
        metadata["reference_summary"] = reference
    return {
        "id": sample_id,
        "source": source,
        "conversations": [
            {"role": "user", "content": f"Summarize document {sample_id}."},
            {"role": "assistant", "content": response},
        ],
        "metadata": metadata,
    }


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_run_root(tmp_path: Path):
    from MR_DFlash.data import build_sample
    from MR_DFlash.offline_features import (
        write_feature_shard,
        write_sharded_feature_manifest,
    )

    root = tmp_path / "run"
    rows = [
        _row(
            "arxiv_a",
            "arxiv",
            "A generated summary with enough distinct words.",
            "A generated summary with enough distinct words.",
        ),
        _row("sharegpt_b", "sharegpt", "Another useful answer with several words."),
    ]
    data_path = root / "regenerated_full" / "train.jsonl"
    _write_jsonl(data_path, rows)
    tokenizer = _ToyTokenizer()
    samples = []
    for row in rows:
        encoded = build_sample(
            row,
            tokenizer,
            64,
            supervision_mode="last_assistant",
        )
        assert encoded is not None
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        samples.append(
            {
                "id": row["id"],
                "input_ids": input_ids,
                "loss_mask": torch.tensor(encoded["loss_mask"], dtype=torch.float32),
                "hidden_states": _features(input_ids)[0],
                "length": len(input_ids),
            }
        )
    cache_path = root / "target_features_qwen3_4b_full" / "train"
    shard = write_feature_shard(cache_path / "shard_00000.pt", samples)
    write_sharded_feature_manifest(
        cache_path,
        target_model_path="tiny-target",
        feature_layer_ids=[1],
        hidden_size=2,
        feature_width=2,
        max_length=64,
        requested_torch_dtype="float32",
        shards=[shard],
        sample_ids=[row["id"] for row in rows],
        capture_backend="hf_backbone",
        attention_backend="sdpa",
    )
    return root, rows, cache_path


def test_quality_audit_reports_response_warnings_and_reference_rouge() -> None:
    from audit_regenerated_cache_quality import analyze_regenerated_quality

    rows = [
        _row(
            "good",
            "arxiv",
            "The paper proposes a useful method for robust summarization.",
            "The paper proposes a useful method for robust summarization.",
        ),
        _row("short", "sharegpt", "Yes."),
        _row(
            "repeat-a",
            "arxiv",
            "alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta",
        ),
        _row(
            "repeat-b",
            "arxiv",
            "alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta",
        ),
    ]

    report = analyze_regenerated_quality(rows)

    assert report["num_samples"] == 4
    assert report["reference_metrics"]["num_samples"] == 1
    assert report["reference_metrics"]["rougeL"] == 1.0
    assert report["warning_counts"]["short_response"] == 1
    assert report["warning_counts"]["duplicate_response"] == 1
    assert report["warning_counts"]["repetitive_4gram"] == 2
    assert report["flagged_samples_truncated"] is False


def test_regenerated_audit_rejects_sample_without_user_prompt() -> None:
    from audit_regenerated_cache_quality import _validate_regenerated_rows

    row = {
        "id": "assistant-only",
        "conversations": [{"role": "assistant", "content": "A generated response."}],
        "metadata": {"generation_model": "tiny-target"},
    }

    with pytest.raises(ValueError, match="user prompt"):
        _validate_regenerated_rows([row], expected_generation_model="tiny-target")


def test_hidden_sample_selection_is_reproducible_and_covers_sources() -> None:
    from audit_regenerated_cache_quality import select_stratified_sample_ids

    rows = [
        {"id": f"a{i}", "source": "arxiv"} for i in range(8)
    ] + [
        {"id": f"s{i}", "source": "sharegpt"} for i in range(8)
    ]
    lengths = {row["id"]: (int(row["id"][1:]) + 1) * 100 for row in rows}

    selected_a = select_stratified_sample_ids(rows, lengths, sample_count=8, seed=42)
    selected_b = select_stratified_sample_ids(rows, lengths, sample_count=8, seed=42)

    assert selected_a == selected_b
    assert len(selected_a) == 8
    assert {sample_id[0] for sample_id in selected_a} == {"a", "s"}
    assert len({lengths[sample_id] for sample_id in selected_a}) >= 4


def test_hidden_comparison_flags_numeric_mismatch() -> None:
    from audit_regenerated_cache_quality import compare_hidden_values

    cached = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    matching = compare_hidden_values(cached, cached.clone(), atol=0.05, rtol=0.05)
    mismatching = compare_hidden_values(
        cached,
        cached + 1.0,
        atol=0.05,
        rtol=0.05,
    )

    assert matching["allclose"] is True
    assert matching["cosine_similarity"] == pytest.approx(1.0)
    assert mismatching["allclose"] is False
    assert mismatching["max_abs_error"] == 1.0


def test_run_quality_audit_checks_cache_and_recomputed_hidden(tmp_path: Path) -> None:
    from audit_regenerated_cache_quality import audit_run_root

    root, _rows, _cache_path = _make_run_root(tmp_path)
    progress = []
    report = audit_run_root(
        root,
        splits=("train",),
        hidden_samples_per_split=2,
        device="cpu",
        tokenizer=_ToyTokenizer(),
        capture_factory=lambda _manifest, _device: _ToyCapture(),
        progress_callback=progress.append,
    )

    assert report["status"] == "pass"
    assert report["splits"]["train"]["cache"]["num_samples"] == 2
    assert report["splits"]["train"]["hidden_recompute"]["checked_samples"] == 2
    assert report["splits"]["train"]["quality"]["reference_metrics"]["num_samples"] == 1
    assert any("cache train: 2/2" in message for message in progress)
    assert any("hidden train: 2/2" in message for message in progress)


def test_run_quality_audit_fails_when_recomputed_hidden_differs(tmp_path: Path) -> None:
    from audit_regenerated_cache_quality import audit_run_root

    root, _rows, _cache_path = _make_run_root(tmp_path)
    report = audit_run_root(
        root,
        splits=("train",),
        hidden_samples_per_split=1,
        device="cpu",
        tokenizer=_ToyTokenizer(),
        capture_factory=lambda _manifest, _device: _ToyCapture(offset=100.0),
    )

    assert report["status"] == "fail"
    assert report["splits"]["train"]["hidden_recompute"]["failed_samples"] == 1


def test_run_quality_audit_rejects_nonfinite_cached_hidden(tmp_path: Path) -> None:
    from audit_regenerated_cache_quality import audit_run_root

    root, _rows, cache_path = _make_run_root(tmp_path)
    shard_path = cache_path / "shard_00000.pt"
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    payload["samples"][0]["hidden_states"][0, 0] = float("nan")
    torch.save(payload, shard_path)

    with pytest.raises(ValueError, match="NaN/Inf"):
        audit_run_root(
            root,
            splits=("train",),
            hidden_samples_per_split=1,
            device="cpu",
            tokenizer=_ToyTokenizer(),
            capture_factory=lambda _manifest, _device: _ToyCapture(),
        )


def test_run_quality_audit_rejects_token_cache_mismatch(tmp_path: Path) -> None:
    from audit_regenerated_cache_quality import audit_run_root

    root, _rows, cache_path = _make_run_root(tmp_path)
    shard_path = cache_path / "shard_00000.pt"
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    payload["samples"][0]["input_ids"][0] += 1
    torch.save(payload, shard_path)

    with pytest.raises(ValueError, match="input_ids/loss_mask lệch regenerated"):
        audit_run_root(
            root,
            splits=("train",),
            hidden_samples_per_split=1,
            device="cpu",
            tokenizer=_ToyTokenizer(),
            capture_factory=lambda _manifest, _device: _ToyCapture(),
        )


def test_run_quality_audit_rejects_missing_shard(tmp_path: Path) -> None:
    from audit_regenerated_cache_quality import audit_run_root

    root, _rows, cache_path = _make_run_root(tmp_path)
    (cache_path / "shard_00000.pt").unlink()

    with pytest.raises(FileNotFoundError, match="thiếu shard"):
        audit_run_root(
            root,
            splits=("train",),
            hidden_samples_per_split=1,
            device="cpu",
            tokenizer=_ToyTokenizer(),
            capture_factory=lambda _manifest, _device: _ToyCapture(),
        )
