"""Tests cho profile scheduler cache theo token budget."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


SRC_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = SRC_DIR.parent / "scripts" / "mr_dflash"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


SCHEMA_VERSION = "mr_dflash_cache_throughput_profile_v1"


def _candidate_payload() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "version": 1,
        "buckets": [
            {"min_length": 1, "max_length": 4096, "batch_size": 64, "token_budget": 262144},
            {"min_length": 4097, "max_length": 8192, "batch_size": 32, "token_budget": 262144},
            {"min_length": 8193, "max_length": 16384, "batch_size": 16, "token_budget": 262144},
            {"min_length": 16385, "max_length": 32768, "batch_size": 4, "token_budget": 131072},
        ],
    }


def test_profile_selects_first_matching_bucket_and_token_budget() -> None:
    from MR_DFlash.cache_throughput import CacheThroughputProfile

    profile = CacheThroughputProfile.from_payload(_candidate_payload())

    assert profile.batch_for_length(1) == 64
    assert profile.batch_for_length(4096) == 64
    assert profile.batch_for_length(4097) == 32
    assert profile.batch_for_length(8192) == 32
    assert profile.batch_for_length(8193) == 16
    assert profile.batch_for_length(16384) == 16
    assert profile.batch_for_length(16385) == 4
    assert profile.batch_for_length(32768) == 4
    assert profile.token_budget_for_length(4096) == 262144
    assert profile.token_budget_for_length(16385) == 131072

    with pytest.raises(ValueError, match="vượt bucket cuối"):
        profile.batch_for_length(32769)
    with pytest.raises(ValueError, match="vượt bucket cuối"):
        profile.token_budget_for_length(32769)


@pytest.mark.parametrize(
    ("buckets", "message"),
    [
        ([], "không được rỗng"),
        (
            [
                {"min_length": 1, "max_length": 8192, "batch_size": 1, "token_budget": 8},
                {"min_length": 8193, "max_length": 4096, "batch_size": 1, "token_budget": 8},
            ],
            "tăng dần",
        ),
        (
            [
                {"min_length": 1, "max_length": 4096, "batch_size": 1, "token_budget": 8},
                {"min_length": 4096, "max_length": 8192, "batch_size": 1, "token_budget": 8},
            ],
            "liên tục",
        ),
        (
            [
                {"min_length": 1, "max_length": 4096, "batch_size": 1, "token_budget": 8},
                {"min_length": 4098, "max_length": 8192, "batch_size": 1, "token_budget": 8},
            ],
            "liên tục",
        ),
        ([{"min_length": -1, "max_length": 4096, "batch_size": 1, "token_budget": 8}], "âm"),
        ([{"min_length": 1, "max_length": 4096, "batch_size": 0, "token_budget": 8}], "batch_size"),
        ([{"min_length": 1, "max_length": 4096, "batch_size": 1, "token_budget": 0}], "token_budget"),
    ],
)
def test_profile_rejects_malformed_bucket_payloads(
    buckets: list[dict[str, int]], message: str
) -> None:
    from MR_DFlash.cache_throughput import CacheThroughputProfile

    payload = {"schema_version": SCHEMA_VERSION, "version": 1, "buckets": buckets}
    with pytest.raises(ValueError, match=message):
        CacheThroughputProfile.from_payload(payload)


def test_profile_validation_requires_coverage_and_batch_limit() -> None:
    from MR_DFlash.cache_throughput import CacheThroughputProfile

    profile = CacheThroughputProfile.from_payload(_candidate_payload())
    profile.validate(max_length=8192, batch_limit=64)

    with pytest.raises(ValueError, match="batch_limit"):
        profile.validate(max_length=8192, batch_limit=32)
    with pytest.raises(ValueError, match="bao phủ"):
        profile.validate(max_length=32769, batch_limit=64)


def test_profile_serialization_and_hash_are_canonical() -> None:
    from MR_DFlash.cache_throughput import CacheThroughputProfile

    payload = _candidate_payload()
    reordered = {
        "buckets": [dict(reversed(list(bucket.items()))) for bucket in payload["buckets"]],
        "version": payload["version"],
        "schema_version": payload["schema_version"],
    }
    first = CacheThroughputProfile.from_payload(payload)
    second = CacheThroughputProfile.from_payload(reordered)

    expected_serialized = (
        '{"buckets":[{"batch_size":64,"max_length":4096,"min_length":1,"token_budget":262144},'
        '{"batch_size":32,"max_length":8192,"min_length":4097,"token_budget":262144},'
        '{"batch_size":16,"max_length":16384,"min_length":8193,"token_budget":262144},'
        '{"batch_size":4,"max_length":32768,"min_length":16385,"token_budget":131072}],'
        '"schema_version":"mr_dflash_cache_throughput_profile_v1","version":1}'
    )
    assert first.serialized_payload == expected_serialized
    assert first.profile_sha256 == second.profile_sha256
    assert len(first.profile_sha256) == 64
    assert json.loads(first.serialized_payload) == first.to_payload()


def test_profile_caps_batch_by_actual_padded_token_count() -> None:
    from MR_DFlash.cache_throughput import CacheThroughputProfile

    profile = CacheThroughputProfile.from_payload(_candidate_payload())

    assert profile.cap_batch_size(padded_length=4096, requested_batch_size=64) == 64
    assert profile.cap_batch_size(padded_length=8192, requested_batch_size=64) == 32
    assert profile.cap_batch_size(padded_length=32768, requested_batch_size=64) == 4
    assert profile.batch_for_lengths([100, 200, 4096], requested_batch_size=64) == 64


def test_cache_worker_consumes_throughput_profile(tmp_path: Path, monkeypatch) -> None:
    import cache_target_features
    from MR_DFlash.tokenized_data import write_tokenized_manifest

    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 2

    class FakeCapturer:
        instances = []

        def __init__(self, *_args, **_kwargs):
            self.tokenizer = TinyTokenizer()
            self.device = torch.device("cpu")
            self.layer_ids = [0]
            self.context_feature_dim = 2
            self.model = type("Model", (), {"config": type("Config", (), {"hidden_size": 2})()})()
            self.batch_sizes = []
            self.__class__.instances.append(self)

        def capture_batch(self, input_ids, attention_mask):
            self.batch_sizes.append(int(input_ids.shape[0]))
            return [
                torch.ones(int(length), 2, dtype=torch.bfloat16)
                for length in attention_mask.sum(dim=-1).tolist()
            ]

        def close(self):
            return None

    monkeypatch.setattr(cache_target_features, "HFTargetCapture", FakeCapturer)
    tokenized = tmp_path / "tokenized"
    tokenized.mkdir()
    torch.save(
        {
            "samples": [
                {"id": "a", "input_ids": torch.arange(4), "loss_mask": torch.ones(4), "length": 4},
                {"id": "b", "input_ids": torch.arange(4), "loss_mask": torch.ones(4), "length": 4},
                {"id": "c", "input_ids": torch.arange(4), "loss_mask": torch.ones(4), "length": 4},
            ]
        },
        tokenized / "shard_00000.pt",
    )
    write_tokenized_manifest(
        tokenized,
        shards=[{"path": "shard_00000.pt", "count": 3}],
        num_samples=3,
        target_model="tiny-target",
        feature_layer_ids=[0],
        chat_template="tiny",
        max_length=8,
        supervision_mode="last_assistant",
    )
    profile = tmp_path / "throughput_profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "version": 1,
                "buckets": [
                    {"min_length": 1, "max_length": 8, "batch_size": 2, "token_budget": 16}
                ],
            }
        ),
        encoding="utf-8",
    )

    cache_target_features.cache_dataset(
        target_model_path="tiny-target",
        tokenized_path=str(tokenized),
        output_path=str(tmp_path / "cache"),
        max_length=8,
        batch_size=1,
        bucket_buffer_size=2,
        shard_size=4,
        layer_ids=[0],
        device="cpu",
        throughput_profile=str(profile),
    )

    assert FakeCapturer.instances[0].batch_sizes == [2, 1]
