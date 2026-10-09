from __future__ import annotations

import json
from pathlib import Path

import pytest

from Finetuning.data import iter_summary_jsonl
from scripts.prepare_finetuning_phase1_inputs import prepare_inputs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(sample_id: str, source: str, *, split: str, reference: str = "") -> dict:
    messages = (
        [{"role": "user", "content": f"Summarize the scientific document:\n\nPaper body {sample_id}"}]
        if source == "arxiv"
        else [
            {"role": "user", "content": f"Question one {sample_id}"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": f"Question two {sample_id}"},
        ]
    )
    metadata = {"prompt_length_bin": 2, "prompt_token_length": 5000, "split": split}
    if reference:
        metadata["reference_summary"] = reference
    return {
        "id": sample_id,
        "source": source,
        "conversations": messages,
        "metadata": metadata,
    }


def test_prepare_inputs_preserves_splits_and_only_uses_real_arxiv_references(tmp_path: Path) -> None:
    source = tmp_path / "stratified"
    output = tmp_path / "phase1_inputs"
    _write_jsonl(
        source / "train_prompts.jsonl",
        [
            _row("a1", "arxiv", split="train", reference="Gold paper abstract."),
            _row("s1", "sharegpt", split="train"),
            {"record_type": "summary", "count": 2},
        ],
    )
    _write_jsonl(
        source / "val_prompts.jsonl",
        [_row("a2", "arxiv", split="val", reference="Validation abstract.")],
    )
    _write_jsonl(source / "test_prompts.jsonl", [_row("heldout", "sharegpt", split="test")])

    report = prepare_inputs(source, output)

    train = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    evaluation = [json.loads(line) for line in (output / "eval.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in train] == ["arxiv:a1", "sharegpt:s1"]
    assert [row["id"] for row in evaluation] == ["arxiv:a2"]
    assert train[0]["summary"] == "Gold paper abstract."
    assert train[1]["summary"] == ""
    assert train[1]["metadata"]["reference_kind"] == "none"
    assert train[1]["document"].endswith("ASSISTANT:")
    assert train[0]["document"] == "Summarize the scientific document:\n\nPaper body a1"
    assert report["splits"]["train"]["records"] == 2
    assert report["splits"]["validation"]["records"] == 1
    assert report["splits"]["train"]["by_source"] == {"arxiv": 1, "sharegpt": 1}
    assert not (output / "test.jsonl").exists()
    loaded = list(iter_summary_jsonl(output / "train.jsonl"))
    assert len(loaded) == 2
    assert loaded[1].summary == ""


def test_prepare_inputs_rejects_duplicate_identity_across_splits(tmp_path: Path) -> None:
    source = tmp_path / "stratified"
    _write_jsonl(source / "train_prompts.jsonl", [_row("same", "sharegpt", split="train")])
    _write_jsonl(source / "val_prompts.jsonl", [_row("same", "sharegpt", split="val")])

    with pytest.raises(ValueError, match="overlap between train and validation"):
        prepare_inputs(source, tmp_path / "output")


def test_prepare_inputs_deduplicates_identical_prompts_inside_one_split(tmp_path: Path) -> None:
    source = tmp_path / "stratified"
    first = _row("sharegpt:first", "sharegpt", split="train")
    duplicate = _row("sharegpt:duplicate", "sharegpt", split="train")
    duplicate["conversations"] = first["conversations"]
    _write_jsonl(source / "train_prompts.jsonl", [first, duplicate])
    _write_jsonl(source / "val_prompts.jsonl", [_row("val", "arxiv", split="val")])

    report = prepare_inputs(source, tmp_path / "output")

    train = (tmp_path / "output" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train) == 1
    assert report["splits"]["train"]["records"] == 1
    assert report["splits"]["train"]["deduplicated_same_split_prompts"] == 1


def test_prepare_inputs_rejects_same_prompt_across_splits_even_with_different_ids(tmp_path: Path) -> None:
    source = tmp_path / "stratified"
    train = _row("train-id", "sharegpt", split="train")
    validation = _row("validation-id", "sharegpt", split="val")
    validation["conversations"] = train["conversations"]
    _write_jsonl(source / "train_prompts.jsonl", [train])
    _write_jsonl(source / "val_prompts.jsonl", [validation])

    with pytest.raises(ValueError, match="prompt content overlap between train and validation"):
        prepare_inputs(source, tmp_path / "output")


def test_prepare_inputs_allows_same_source_document_inside_one_split(tmp_path: Path) -> None:
    source = tmp_path / "stratified"
    first = _row("first", "arxiv", split="train")
    second = _row("second", "arxiv", split="train")
    first["metadata"]["original_id"] = "same-paper"
    second["metadata"]["original_id"] = "same-paper"
    _write_jsonl(source / "train_prompts.jsonl", [first, second])
    _write_jsonl(source / "val_prompts.jsonl", [_row("val", "sharegpt", split="val")])

    report = prepare_inputs(source, tmp_path / "output")

    assert report["splits"]["train"]["records"] == 2


def test_prepare_inputs_rejects_source_document_overlap_across_splits(tmp_path: Path) -> None:
    source = tmp_path / "stratified"
    train = _row("train-id", "arxiv", split="train")
    validation = _row("validation-id", "arxiv", split="val")
    train["metadata"]["original_id"] = "same-paper"
    validation["metadata"]["original_id"] = "same-paper"
    _write_jsonl(source / "train_prompts.jsonl", [train])
    _write_jsonl(source / "val_prompts.jsonl", [validation])

    with pytest.raises(ValueError, match="source-document overlap between train and validation"):
        prepare_inputs(source, tmp_path / "output")


def test_prepare_inputs_refuses_to_overwrite_output(tmp_path: Path) -> None:
    source = tmp_path / "stratified"
    _write_jsonl(source / "train_prompts.jsonl", [_row("a", "arxiv", split="train")])
    _write_jsonl(source / "val_prompts.jsonl", [_row("b", "arxiv", split="val")])
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(FileExistsError, match="output directory already exists"):
        prepare_inputs(source, output)
