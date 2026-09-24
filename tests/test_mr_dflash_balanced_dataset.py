from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "mr_dflash"))


class TinyTokenizer:
    def __init__(self):
        self.calls = 0

    def apply_chat_template(
        self, conversations, *, tokenize=False, add_generation_prompt=False, enable_thinking=False
    ):
        rendered = " ".join(
            f"<{message['role']}> {message['content']}" for message in conversations
        )
        if add_generation_prompt:
            rendered += " <assistant>"
        return rendered

    def __call__(self, texts, **_kwargs):
        self.calls += 1
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [text.split() for text in texts]}


def _candidate(sample_id: str, tokens: int) -> dict:
    return {"id": sample_id, "prompt_tokens": tokens}


def test_default_length_bins_are_five_two_kibibyte_intervals() -> None:
    from build_balanced_dataset import DEFAULT_MAX_PROMPT_TOKENS, length_bin

    assert DEFAULT_MAX_PROMPT_TOKENS == 10 * 1024
    assert [length_bin(value) for value in (0, 2048, 2049, 4096, 4097, 6144, 6145, 8192, 8193, 10240)] == [
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4
    ]
    assert length_bin(10241) is None


def test_prompt_length_counts_rendered_chat_template_and_generation_prefix() -> None:
    from build_balanced_dataset import prompt_token_counts

    tokenizer = TinyTokenizer()
    rows = [
        {
            "conversations": [
                {"role": "user", "content": "summarize paper"},
            ]
        }
    ]

    assert prompt_token_counts(rows, tokenizer) == [4]


def test_quota_plan_meets_source_and_bin_totals_without_forcing_half_each_bin() -> None:
    from build_balanced_dataset import BIN_TARGETS, SOURCE_TARGETS, plan_source_bin_quotas

    available = {
        "sharegpt": {0: 8000, 1: 4000, 2: 7000, 3: 5000, 4: 6000},
        "arxiv": {0: 4000, 1: 9000, 2: 6000, 3: 5000, 4: 4000},
    }
    plan = plan_source_bin_quotas(available)

    assert plan["feasible"] is True
    quotas = plan["source_bin_quotas"]
    assert {source: sum(bins.values()) for source, bins in quotas.items()} == SOURCE_TARGETS
    assert {
        bucket: sum(quotas[source][bucket] for source in SOURCE_TARGETS)
        for bucket in BIN_TARGETS
    } == BIN_TARGETS
    assert all(quotas[source][bucket] <= available[source][bucket] for source in SOURCE_TARGETS for bucket in BIN_TARGETS)
    assert len({quotas["sharegpt"][bucket] for bucket in BIN_TARGETS}) > 1

    from build_balanced_dataset import _allocate_five_percent

    val = _allocate_five_percent(quotas)
    test = _allocate_five_percent(quotas, val)
    for source in SOURCE_TARGETS:
        assert sum(val[source].values()) == 1250
        assert sum(test[source].values()) == 1250
    for bucket in BIN_TARGETS:
        assert sum(val[source][bucket] for source in SOURCE_TARGETS) == 500
        assert sum(test[source][bucket] for source in SOURCE_TARGETS) == 500


def test_quota_plan_reports_shortage_instead_of_changing_targets() -> None:
    from build_balanced_dataset import plan_source_bin_quotas

    plan = plan_source_bin_quotas(
        {
            "sharegpt": {0: 9000, 1: 9000, 2: 9000, 3: 0, 4: 0},
            "arxiv": {0: 5000, 1: 5000, 2: 5000, 3: 5000, 4: 5000},
        }
    )

    assert plan["feasible"] is False
    assert plan["source_shortages"]["sharegpt"] == 0
    assert plan["bin_shortages"][3] == 5000
    assert plan["bin_shortages"][4] == 5000
    assert plan["source_bin_quotas"] == {}


def test_selection_obeys_each_source_bin_quota_and_is_order_independent() -> None:
    from build_balanced_dataset import select_candidates_by_bin_quota

    candidates = [
        *(_candidate(f"s-{index}", 100) for index in range(10)),
        *(_candidate(f"a-{index}", 3000) for index in range(8)),
        *(_candidate(f"l-{index}", 9000) for index in range(6)),
        _candidate("too-long", 10241),
    ]
    quotas = {0: 3, 1: 2, 2: 0, 3: 0, 4: 2}
    first = select_candidates_by_bin_quota(
        candidates, source="sharegpt", bin_quotas=quotas, seed=17
    )
    repeated = select_candidates_by_bin_quota(
        reversed(candidates), source="sharegpt", bin_quotas=quotas, seed=17
    )

    assert first == repeated
    assert Counter(item["length_bin"] for item in first.values()) == Counter({0: 3, 1: 2, 4: 2})
    assert "too-long" not in first


def test_splits_are_exact_and_stratified_by_source_and_bin() -> None:
    from build_balanced_dataset import assign_stratified_splits

    selections = {
        source: {
            f"{source}-{bucket}-{index}": {
                "prompt_tokens": bucket * 2048 + index,
                "length_bin": bucket,
            }
            for bucket in range(5)
            for index in range(100)
        }
        for source in ("sharegpt", "arxiv")
    }
    source_bin_quotas = {source: {bucket: 100 for bucket in range(5)} for source in selections}

    assignments = assign_stratified_splits(
        selections, source_bin_quotas=source_bin_quotas, seed=9
    )
    assert assignments == assign_stratified_splits(
        selections, source_bin_quotas=source_bin_quotas, seed=9
    )
    for source in selections:
        assert Counter(assignments[source].values()) == Counter(train=450, val=25, test=25)
        for bucket in range(5):
            counts = Counter(
                assignments[source][sample_id]
                for sample_id, item in selections[source].items()
                if item["length_bin"] == bucket
            )
            assert counts == Counter(train=90, val=5, test=5)


def test_scan_uses_full_cap_deduplicates_and_resumes_cache(tmp_path: Path) -> None:
    from build_balanced_dataset import scan_normalized_source

    source = tmp_path / "prompts.jsonl"
    rows = [
        {"id": "short", "conversations": [{"role": "user", "content": "one two"}]},
        {"id": "long", "conversations": [{"role": "user", "content": "one two three four five six"}]},
        {"id": "short", "conversations": [{"role": "user", "content": "duplicate"}]},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    cache, info, tokenizer = tmp_path / "lengths.jsonl", tmp_path / "info.json", TinyTokenizer()

    def run(resume: bool):
        return scan_normalized_source(
            source,
            cache,
            info,
            tokenizer,
            source="sharegpt",
            tokenizer_ref="tiny",
            batch_size=2,
            resume=resume,
        )

    first = run(False)
    calls = tokenizer.calls
    assert run(True) == first
    assert tokenizer.calls == calls
    assert first["unique_rows"] == 2
    assert first["duplicate_ids"] == 1
    assert first["eligible_rows"] == 2
    assert first["over_cap"] == 0
    assert len(first["eligible_bin_counts"]) == 5


def test_preview_scans_every_normalized_row_and_writes_reports_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import SimpleNamespace

    import build_balanced_dataset as builder

    share_raw = tmp_path / "sharegpt.json"
    arxiv_raw = tmp_path / "arxiv.jsonl"
    share_raw.write_text("[]", encoding="utf-8")
    arxiv_raw.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "preview"
    normalized = {
        "prepare_sharegpt.py": [
            {"id": "sharegpt-a", "source": "sharegpt", "conversations": [{"role": "user", "content": "one two"}], "metadata": {}},
            {"id": "sharegpt-a", "source": "sharegpt", "conversations": [{"role": "user", "content": "duplicate"}], "metadata": {}},
        ],
        "prepare_arxiv.py": [
            {"id": "arxiv-a", "source": "arxiv", "conversations": [{"role": "user", "content": "one two three"}], "metadata": {}},
        ],
    }

    def fake_prepare(script_name, _input_path, output_path, *, resume):
        rows = normalized[script_name]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_name, **kwargs):
            assert kwargs == {"local_files_only": True, "use_fast": True}
            return TinyTokenizer()

    monkeypatch.setattr(builder, "_prepare_source", fake_prepare)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeAutoTokenizer))

    assert builder.main(
        [
            "--sharegpt-source", str(share_raw),
            "--arxiv-source", str(arxiv_raw),
            "--tokenizer", "qwen3-local",
            "--output-root", str(output),
            "--batch-size", "2",
            "--preview-only",
        ]
    ) == 0

    report = json.loads((output / "reports" / "length_distribution.json").read_text())
    assert report["global"]["unique_rows"] == 2
    assert report["global"]["duplicate_ids"] == 1
    assert report["quota_feasible"] is False
    assert (output / "reports" / "length_distribution.csv").is_file()
    assert (output / "reports" / "length_distribution.png").stat().st_size > 0
    assert (output / "manifests" / "preview_manifest.json").is_file()
    assert not (output / "normalized" / "train_prompts.jsonl").exists()
    with pytest.raises(ValueError, match="quota không khả thi"):
        builder.main(
            [
                "--sharegpt-source", str(share_raw),
                "--arxiv-source", str(arxiv_raw),
                "--tokenizer", "qwen3-local",
                "--output-root", str(output),
                "--batch-size", "2",
                "--resume",
                "--confirm-build",
            ]
        )
    assert not (output / "normalized" / "train_prompts.jsonl").exists()


def test_full_build_requires_explicit_confirmation() -> None:
    from build_balanced_dataset import main

    with pytest.raises(SystemExit, match="--confirm-build"):
        main(["--tokenizer", "qwen3-local"])



def test_scan_accepts_empty_normalized_source(tmp_path: Path) -> None:
    from build_balanced_dataset import scan_normalized_source

    source = tmp_path / "empty.jsonl"
    source.touch()
    summary = scan_normalized_source(
        source,
        tmp_path / "lengths.jsonl",
        tmp_path / "info.json",
        TinyTokenizer(),
        source="arxiv",
        tokenizer_ref="tiny",
        batch_size=2,
        resume=False,
    )

    assert summary["unique_rows"] == 0
    assert summary["eligible_rows"] == 0
    assert summary["over_cap"] == 0
    assert set(summary["eligible_bin_counts"].values()) == {0}
