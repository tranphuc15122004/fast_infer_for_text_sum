"""CPU contract tests for the lazy SpecForge capture adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "mr_dflash"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def test_specforge_capture_trims_ragged_rows_and_preserves_lengths() -> None:
    from specforge_capture import SpecForgeTargetCapture

    class FakeReq:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeOutput:
        def __init__(self, hidden):
            self.aux_hidden_states = hidden

    class FakeBackend:
        def __init__(self):
            self.cleared = 0
            self.seen_ids = []

        def _forward_extend(self, reqs):
            self.seen_ids = [req.origin_input_ids for req in reqs]
            rows = [
                torch.full((len(req.origin_input_ids), 6), float(index + 1))
                for index, req in enumerate(reqs)
            ]
            return FakeOutput(torch.cat(rows, dim=0))

        def _clear_pools(self):
            self.cleared += 1

    class FakeTarget:
        def __init__(self, backend):
            self._backend = backend

        def set_capture_layers(self, layers, *, capture_method):
            assert layers == [1, 9, 17, 25, 33]
            assert capture_method == "dflash"

    backend = FakeBackend()
    target = SpecForgeTargetCapture(
        FakeTarget(backend),
        layer_ids=[1, 9, 17, 25, 33],
        request_cls=FakeReq,
        sampling_params_cls=FakeSamplingParams,
    )
    input_ids = torch.tensor([[10, 11, 12, 0], [20, 21, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    loss_mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 0]])

    rows = target.capture_batch(input_ids, attention_mask, loss_mask)

    assert [tuple(row.shape) for row in rows] == [(3, 6), (2, 6)]
    assert backend.seen_ids == [[10, 11, 12], [20, 21]]
    assert backend.cleared == 1
    assert rows[0].tolist() == [[1.0] * 6] * 3
    assert rows[1].tolist() == [[2.0] * 6] * 2


def test_specforge_capture_does_not_import_sglang_for_fake_backend() -> None:
    from specforge_capture import SpecForgeTargetCapture

    class FakeTarget:
        _backend = object()

    target = SpecForgeTargetCapture(
        FakeTarget(),
        layer_ids=[1],
        request_cls=object,
        sampling_params_cls=object,
    )

    assert target.layer_ids == [1]


def test_cache_backend_dispatches_specforge_with_b200_limits(monkeypatch) -> None:
    import cache_target_features
    import specforge_capture
    from MR_DFlash.cache_throughput import CacheThroughputProfile

    class FakeSpecForgeTarget:
        calls = []

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))
            return object()

    monkeypatch.setattr(
        specforge_capture, "SpecForgeTargetCapture", FakeSpecForgeTarget
    )
    profile = CacheThroughputProfile.from_payload(
        {
            "schema_version": "mr_dflash_cache_throughput_profile_v1",
            "version": 1,
            "buckets": [
                {
                    "min_length": 1,
                    "max_length": 8192,
                    "batch_size": 32,
                    "token_budget": 262144,
                }
            ],
        }
    )

    target = cache_target_features._build_capturer(
        target_model_path="tiny-target",
        layer_ids=[1, 9],
        cache_backend="specforge_sglang",
        cache_dir="unused",
        trust_remote_code=False,
        torch_dtype="bfloat16",
        device="cuda:0",
        local_files_only=True,
        target_revision=None,
        attention_backend="flashinfer",
        max_length=8192,
        batch_size=1,
        cache_concurrency=64,
        cache_max_total_tokens=262144,
        cache_memory_fraction=0.99,
        throughput_profile=profile,
    )

    assert target is not None
    args, kwargs = FakeSpecForgeTarget.calls[0]
    assert args[:2] == ("tiny-target", [1, 9])
    assert kwargs["max_running_requests"] == 64
    assert kwargs["max_total_tokens"] == 262144
    assert kwargs["mem_fraction_static"] == 0.99
    assert kwargs["attention_backend"] == "flashinfer"


def test_specforge_maps_hf_default_attention_to_registered_backend() -> None:
    import cache_target_features

    assert (
        cache_target_features._effective_attention_backend(
            "specforge_sglang", "sdpa"
        )
        == "flashinfer"
    )
    assert (
        cache_target_features._effective_attention_backend(
            "specforge_sglang", "torch_native"
        )
        == "torch_native"
    )


def test_vendored_specforge_capture_supports_legacy_sglang_scheduler_module() -> None:
    capture_source = (
        Path(__file__).resolve().parents[3]
        / "externals"
        / "SpecForge"
        / "specforge"
        / "offline_capture"
        / "sglang_backend"
        / "capture.py"
    ).read_text(encoding="utf-8")
    assert "scheduler_dp_attn_mixin" in capture_source


def test_specforge_capture_does_not_prefer_sssd_sglang_source(monkeypatch) -> None:
    import specforge_capture

    specforge_root = specforge_capture._specforge_root()
    sssd_root = specforge_root.parent / "SSSD" / "python"
    monkeypatch.setattr(sys, "path", [str(sssd_root), str(specforge_root)])

    specforge_capture._ensure_specforge_importable()

    assert str(sssd_root) not in sys.path
    assert str(specforge_root) in sys.path


def test_specforge_capture_places_kernel_caches_in_writable_runtime_root(
    monkeypatch, tmp_path: Path
) -> None:
    import specforge_capture

    runtime_root = tmp_path / "runtime-cache"
    for name in (
        "FAST_INFER_CACHE_ROOT",
        "FLASHINFER_WORKSPACE_BASE",
        "TRITON_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FAST_INFER_CACHE_ROOT", str(runtime_root))

    paths = specforge_capture._prepare_runtime_caches()

    assert paths["FAST_INFER_CACHE_ROOT"] == str(runtime_root)
    for name, suffix in (
        ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
    ):
        assert paths[name] == str(runtime_root / suffix)
        assert (runtime_root / suffix).is_dir()


def test_specforge_capture_falls_back_when_configured_kernel_cache_is_read_only(
    monkeypatch, tmp_path: Path
) -> None:
    import specforge_capture

    runtime_root = tmp_path / "runtime-cache"
    read_only = tmp_path / "read-only"
    read_only.mkdir()
    read_only.chmod(0o500)
    try:
        monkeypatch.setenv("FAST_INFER_CACHE_ROOT", str(runtime_root))
        monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(read_only))
        monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
        monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)

        paths = specforge_capture._prepare_runtime_caches()

        assert paths["FLASHINFER_WORKSPACE_BASE"] == str(runtime_root / "flashinfer")
        assert (runtime_root / "flashinfer").is_dir()
    finally:
        read_only.chmod(0o700)
