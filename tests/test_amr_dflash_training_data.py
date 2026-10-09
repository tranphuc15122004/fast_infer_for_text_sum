"""Regenerated corpora must become reproducible prompts, without target leakage."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts", ROOT / "externals/dflash"):
    sys.path.insert(0, str(path))

from AMR_DFlash.artifacts import read_jsonl, sha256_file, write_jsonl
from AMR_DFlash.pipeline import _prompt_for, _source_content_hash, _tokenize_prompt, validate_document_records
from AMR_DFlash.training_data import convert_regenerated_record, prepare_manifest, training_signal_summary
from common.data_loader import load_records, normalize


@pytest.fixture
def tokenizer_path(tmp_path):
    path = tmp_path / "tokenizer"
    backend = Tokenizer(WordLevel({"[UNK]": 0, "a": 1, "b": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}"
        "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    )
    tokenizer.save_pretrained(path)
    return path


def regenerated(sample_id, source="sharegpt", *, prompt=None):
    return {"id": sample_id, "source": source,
            "conversations": [{"role": "user", "content": prompt or f"a b topic {sample_id}"},
                              {"role": "assistant", "content": "GENERATED ANSWER SECRET"}],
            "metadata": {"generation_model": "/local/target", "generation_temperature": 0.0}}


def prepare(tmp_path, tokenizer_path, train, validation, **overrides):
    train_path = write_jsonl(tmp_path / "train.jsonl", train)
    val_path = write_jsonl(tmp_path / "val.jsonl", validation)
    options = dict(inputs={"train": [train_path], "validation": [val_path]},
                   output_dir=tmp_path / "prepared", tokenizer_path=tokenizer_path,
                   limits={"train": 4, "validation": 2, "holdout": 0},
                   min_input_tokens=1, max_input_tokens=64, seed=17, progress=False)
    options.update(overrides)
    return prepare_manifest(**options)


def test_history_is_rendered_once_and_only_final_regenerated_answer_is_removed(tokenizer_path):
    row = regenerated("chat")
    row["conversations"] = [
        {"from": "system", "value": "system rules"},
        {"from": "human", "value": "first question"},
        {"from": "gpt", "value": "previous assistant answer"},
        {"from": "human", "value": "last question"},
        {"from": "gpt", "value": "GENERATED ANSWER SECRET"},
    ]
    converted = convert_regenerated_record(row, split="train")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    prompt = _prompt_for(tokenizer, normalize(converted, 0))
    assert prompt == "system: system rules\nuser: first question\nassistant: previous assistant answer\nuser: last question\nassistant: "
    assert converted["messages"][-1] == {"role": "user", "content": "last question"}
    assert "GENERATED ANSWER SECRET" not in json.dumps(converted)
    assert "reference" not in converted


def test_arxiv_uses_original_summary_as_reference():
    row = regenerated("paper", "arxiv")
    row["metadata"]["reference_summary"] = "Human reference summary"
    converted = convert_regenerated_record(row, split="validation")
    assert converted["reference"] == "Human reference summary"
    assert converted["split"] == "validation"
    assert converted["original_id"] == "paper"
    assert converted["id"] == "arxiv:paper"


@pytest.mark.parametrize("change", ["empty_answer", "missing_answer", "unknown_role", "assistant_prompt"])
def test_malformed_regeneration_fails_with_sample_identity(change):
    row = regenerated("broken")
    if change == "empty_answer":
        row["conversations"][-1]["content"] = " "
    elif change == "missing_answer":
        row["conversations"].pop()
    elif change == "unknown_role":
        row["conversations"][0]["role"] = "alien"
    else:
        row["conversations"].insert(-1, {"role": "assistant", "content": "extra"})
    with pytest.raises(ValueError, match="broken"):
        convert_regenerated_record(row, split="train")


def test_balanced_sampling_preserves_splits_and_loader_contract(tmp_path, tokenizer_path):
    train = [regenerated(f"sg-{i}") for i in range(8)] + [regenerated(f"ar-{i}", "arxiv") for i in range(8)]
    validation = [regenerated("val-sg"), regenerated("val-ar", "arxiv")]
    report = prepare(tmp_path, tokenizer_path, train, validation)
    records = load_records(Path(report["manifest"]))
    validate_document_records(records, {"data": {}})
    assert len(records) == 6
    assert report["selected_by_split_source"] == {"train": {"arxiv": 2, "sharegpt": 2},
                                                   "validation": {"arxiv": 1, "sharegpt": 1}}
    assert all(record["prompt"] and record["raw"]["messages"] for record in records)
    assert {record["raw"]["split"] for record in records} == {"train", "validation"}
    assert sha256_file(tmp_path / "train.jsonl") == report["contract"]["inputs"][0]["sha256"]
    repeated = prepare(tmp_path, tokenizer_path, train, validation, output_dir=tmp_path / "prepared-again")
    assert Path(report["manifest"]).read_bytes() == Path(repeated["manifest"]).read_bytes()


def test_token_filter_uses_same_template_and_truncation_as_capture(tmp_path, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    short = regenerated("short", prompt="a")
    long = regenerated("long", prompt="a b " * 30)
    validation = regenerated("val", "arxiv", prompt="b a " * 30)
    report = prepare(tmp_path, tokenizer_path, [short, long], [validation],
                     min_input_tokens=8, max_input_tokens=16)
    assert report["skipped"]["too_short"] == 1
    for sample in load_records(Path(report["manifest"])):
        _, ids = _tokenize_prompt(tokenizer, sample, max_input_tokens=16)
        assert ids.shape[1] == sample["raw"]["input_tokens_after_truncation"] == 16


def test_overlap_outside_selection_limit_is_rejected(tmp_path, tokenizer_path):
    train = [regenerated("first"), regenerated("duplicate", prompt="shared content a b")]
    val = [regenerated("val", "arxiv", prompt="shared content a b")]
    with pytest.raises(ValueError, match="overlap"):
        prepare(tmp_path, tokenizer_path, train, val, limits={"train": 1, "validation": 1, "holdout": 0})
    assert not (tmp_path / "prepared/manifest.jsonl").exists()


def test_document_identity_prevents_different_windows_crossing_splits(tmp_path, tokenizer_path):
    train, val = regenerated("window-a"), regenerated("window-b")
    for row in (train, val):
        row["metadata"]["original_id"] = "same-chat"
    with pytest.raises(ValueError, match="overlap"):
        prepare(tmp_path, tokenizer_path, [train], [val])


def test_full_history_participates_in_overlap_hash(tokenizer_path):
    first = regenerated("first")
    second = regenerated("second")
    for row, history in ((first, "context A"), (second, "context B")):
        row["conversations"] = [{"role": "user", "content": history},
                                {"role": "assistant", "content": "prior answer"},
                                {"role": "user", "content": "same last question"},
                                {"role": "assistant", "content": "generated"}]
    a, b = convert_regenerated_record(first, split="train"), convert_regenerated_record(second, split="validation")
    assert _source_content_hash(a, a["prompt"]) != _source_content_hash(b, b["prompt"])
    validate_document_records([normalize(a, 0), normalize(b, 1)], {"data": {}})


def test_resume_requires_same_contract_and_valid_completed_outputs(tmp_path, tokenizer_path):
    train, val = [regenerated("train")], [regenerated("val", "arxiv")]
    report = prepare(tmp_path, tokenizer_path, train, val)
    with pytest.raises(FileExistsError):
        prepare(tmp_path, tokenizer_path, train, val)
    assert prepare(tmp_path, tokenizer_path, train, val, resume=True) == report
    with pytest.raises(ValueError, match="contract"):
        prepare(tmp_path, tokenizer_path, train, val, resume=True, seed=99)
    Path(report["manifest"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash|modified"):
        prepare(tmp_path, tokenizer_path, train, val, resume=True)


def test_interrupted_tokenization_can_resume_without_reencoding_completed_prompts(tmp_path, tokenizer_path, monkeypatch):
    original = PreTrainedTokenizerFast.__call__
    calls = []

    def interrupt(tokenizer, *args, **kwargs):
        calls.append(args[0])
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return original(tokenizer, *args, **kwargs)

    train, val = [regenerated("train")], [regenerated("val", "arxiv")]
    monkeypatch.setattr(PreTrainedTokenizerFast, "__call__", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        prepare(tmp_path, tokenizer_path, train, val)
    assert not (tmp_path / "prepared/manifest.jsonl").exists()
    resumed_calls = []

    def count_resumed(tokenizer, *args, **kwargs):
        resumed_calls.append(args[0])
        return original(tokenizer, *args, **kwargs)

    monkeypatch.setattr(PreTrainedTokenizerFast, "__call__", count_resumed)
    report = prepare(tmp_path, tokenizer_path, train, val, resume=True)
    assert report["selected"] == 2
    assert len(resumed_calls) == 1


def test_duplicate_content_is_removed_within_split(tmp_path, tokenizer_path):
    report = prepare(tmp_path, tokenizer_path,
                     [regenerated("a", prompt="same a b"), regenerated("b", prompt="same a b")],
                     [regenerated("validation", "arxiv")])
    assert report["selected"] == 2
    assert report["skipped"]["duplicate_content_or_document"] == 1


def test_small_scale_120_rows_has_balanced_sources_and_no_response_leakage(tmp_path, tokenizer_path):
    train = [regenerated(f"train-{i}", "arxiv" if i % 2 else "sharegpt",
                         prompt=f"train topic {i} " + "a b " * (5 + i % 10)) for i in range(100)]
    validation = [regenerated(f"val-{i}", "arxiv" if i % 2 else "sharegpt",
                              prompt=f"validation topic {i} " + "b a " * (5 + i % 10)) for i in range(20)]
    for row in train + validation:
        if row["source"] == "arxiv":
            row["metadata"]["reference_summary"] = "Original paper abstract"
    report = prepare(tmp_path, tokenizer_path, train, validation,
                     limits={"train": 20, "validation": 10, "holdout": 0})
    assert sum(sum(counts.values()) for counts in report["scanned_by_split_source"].values()) == 120
    assert report["selected_by_split_source"] == {"train": {"arxiv": 10, "sharegpt": 10},
                                                   "validation": {"arxiv": 5, "sharegpt": 5}}
    assert report["with_reference"] == 15
    assert report["training_labels_ready"] is False
    records = load_records(Path(report["manifest"]))
    assert all(record["prompt"].strip() and record["raw"]["source_document_id"] for record in records)
    assert "GENERATED ANSWER SECRET" not in Path(report["manifest"]).read_text()
    assert len({record["id"] for record in records}) == 30


def test_label_summary_requires_train_preferences_and_uncensored_train_teachers():
    labels = [{"split": "train", "state_id": "t", "status": "success", "censored": False,
               "teacher_logits_ref": "teacher/t.pt"},
              {"split": "train", "state_id": "c", "status": "success", "censored": True,
               "teacher_logits_ref": "teacher/c.pt"},
              {"split": "validation", "state_id": "v", "status": "success", "censored": False}]
    preferences = [{"split": "validation", "state_id": "v"}]
    report = training_signal_summary(labels, preferences)
    assert report["by_split"]["train"]["uncensored_teacher_rows"] == 1
    assert report["train_compressor_signal_present"] is True
    assert report["train_selector_signal_present"] is False
    assert report["train_signal_present"] is False
    preferences.append({"split": "train", "state_id": "t"})
    assert training_signal_summary(labels, preferences)["train_signal_present"] is True


def test_empty_validation_after_length_filter_fails_instead_of_training_on_train_only(tmp_path, tokenizer_path):
    with pytest.raises(ValueError, match="validation"):
        prepare(tmp_path, tokenizer_path, [regenerated("long", prompt="a " * 30)],
                [regenerated("short", prompt="a")], min_input_tokens=12)


def test_multi_turn_prompt_requires_target_chat_template(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer.chat_template = None
    converted = convert_regenerated_record(regenerated("no-template"), split="train")
    with pytest.raises(ValueError, match="chat_template"):
        _prompt_for(tokenizer, normalize(converted, 0))


def test_run_sh_prepare_data_needs_only_local_tokenizer(tmp_path, tokenizer_path):
    train_path = write_jsonl(tmp_path / "train.jsonl", [regenerated("train")])
    val_path = write_jsonl(tmp_path / "val.jsonl", [regenerated("val", "arxiv")])
    master = tmp_path / "master.env"
    master.write_text("FI_DEVICE=cpu\nFI_OFFLINE=1\n", encoding="utf-8")
    environment = {**os.environ, "FAST_INFER_MASTER_CONFIG": str(master),
                   "FAST_INFER_PYTHON": str(ROOT / ".venv/bin/python"), "AMR_DEVICE": "cpu"}
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run.sh"), "amr_dflash", "prepare-data",
         "--train-input", str(train_path), "--validation-input", str(val_path),
         "--tokenizer-path", str(tokenizer_path), "--output-dir", str(tmp_path / "prepared"),
         "--min-input-tokens", "1", "--max-input-tokens", "64"],
        cwd=ROOT, env=environment, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert len(read_jsonl(tmp_path / "prepared/manifest.jsonl")) == 2
