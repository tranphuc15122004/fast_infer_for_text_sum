"""Audit chất lượng regenerated samples và MR-DFlash hidden cache.

Script quét đầy đủ JSONL/cache để kiểm tra cấu trúc, coverage, token/mask
parity và các dấu hiệu response đáng ngờ. Sau đó nó chạy lại target backbone
cho một mẫu phân tầng nhỏ trên một GPU và so sánh hidden với cache.

Ví dụ trên B200::

    python3 scripts/mr_dflash/audit_regenerated_cache_quality.py \
      --run-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/run \
      --device cuda:0 --hidden-samples-per-split 32

Các cảnh báo nội dung chỉ nhằm hỗ trợ review; lỗi cấu trúc hoặc hidden
recompute vượt tolerance làm lệnh trả mã khác 0. Script không sửa dữ liệu.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_path in (SOURCE_ROOT, SCRIPT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from _common import read_jsonl, write_json  # noqa: E402
from verify_feature_cache import _load_manifest, validate_feature_cache  # noqa: E402


_MANIFEST_COMPATIBILITY_KEYS = (
    "target_model_path",
    "target_revision",
    "feature_layer_ids",
    "hidden_size",
    "feature_width",
    "max_length",
    "requested_torch_dtype",
    "capture_backend",
    "attention_backend",
)
_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
_WHITESPACE = re.compile(r"\s+")


def _last_assistant_text(row: Mapping[str, Any]) -> str:
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"sample {row.get('id')!r}: conversations rỗng/không hợp lệ")
    last = conversations[-1]
    if not isinstance(last, dict) or str(last.get("role", "")).strip().lower() != "assistant":
        raise ValueError(f"sample {row.get('id')!r}: message cuối không phải assistant")
    response = last.get("content")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"sample {row.get('id')!r}: assistant response rỗng")
    return response.strip()


def _ngram_repetition_ratio(tokens: Sequence[str], n: int) -> float:
    if n < 1:
        raise ValueError("repetition n-gram size phải >= 1")
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    if len(grams) < 2:
        return 0.0
    return (len(grams) - len(set(grams))) / len(grams)


def _percentile(sorted_values: Sequence[int], fraction: float) -> Optional[int]:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * fraction)))
    return int(sorted_values[index])


def analyze_regenerated_quality(
    rows: Iterable[Mapping[str, Any]],
    *,
    min_response_tokens: int = 4,
    repetition_ngram_size: int = 4,
    repetition_warning_ratio: float = 0.30,
) -> Dict[str, Any]:
    """Tính thống kê response, cảnh báo heuristic và ROUGE khi có reference."""
    if min_response_tokens < 0:
        raise ValueError("min_response_tokens phải >= 0")
    if not 0.0 <= repetition_warning_ratio <= 1.0:
        raise ValueError("repetition_warning_ratio phải thuộc [0,1]")

    from Benchmark.common.rouge import add_rouge, aggregate_rouge

    seen_ids: set[str] = set()
    response_owner: Dict[str, str] = {}
    response_word_counts: list[int] = []
    response_char_counts: list[int] = []
    warning_counts: Counter[str] = Counter()
    flagged_samples: list[Dict[str, Any]] = []
    flagged_sample_count = 0
    rouge_rows: list[Dict[str, Any]] = []
    num_samples = 0
    num_clipped = 0
    num_template_clipped = 0
    generation_models: Counter[str] = Counter()

    for row in rows:
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"regenerated có id rỗng/trùng: {sample_id!r}")
        seen_ids.add(sample_id)
        response = _last_assistant_text(row)
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"sample {sample_id!r}: metadata không hợp lệ")

        words = _WHITESPACE.split(response.strip()) if response.strip() else []
        normalized = " ".join(response.casefold().split())
        flags: list[str] = []
        if len(words) < min_response_tokens:
            flags.append("short_response")
        repeated_ratio = _ngram_repetition_ratio(words, repetition_ngram_size)
        if (
            len(words) >= repetition_ngram_size * 2
            and repeated_ratio >= repetition_warning_ratio
        ):
            flags.append(f"repetitive_{repetition_ngram_size}gram")
        if normalized in response_owner:
            flags.append("duplicate_response")
        else:
            response_owner[normalized] = sample_id

        clipped = bool(metadata.get("generation_budget_clipped", False))
        template_clipped = bool(metadata.get("response_template_clipped", False))
        if clipped:
            flags.append("generation_budget_clipped")
            num_clipped += 1
        if template_clipped:
            flags.append("response_template_clipped")
            num_template_clipped += 1

        for flag in flags:
            warning_counts[flag] += 1
        if flags and len(flagged_samples) < 200:
            flagged_samples.append(
                {
                    "id": sample_id,
                    "source": str(row.get("source", "unknown")),
                    "flags": flags,
                    "response_preview": response[:500],
                }
            )
        if flags:
            flagged_sample_count += 1

        reference = str(metadata.get("reference_summary", "")).strip()
        if reference:
            score: Dict[str, Any] = {"id": sample_id}
            add_rouge(score, response, reference)
            rouge_rows.append(score)
        generation_models[str(metadata.get("generation_model", ""))] += 1
        response_word_counts.append(len(words))
        response_char_counts.append(len(response))
        num_samples += 1

    sorted_word_counts = sorted(response_word_counts)
    sorted_char_counts = sorted(response_char_counts)
    rouge_summary: Dict[str, Any] = {
        "num_samples": len(rouge_rows),
        **aggregate_rouge(rouge_rows),
    }
    return {
        "num_samples": num_samples,
        "generation_model_counts": dict(sorted(generation_models.items())),
        "response_words": {
            "min": min(sorted_word_counts) if sorted_word_counts else None,
            "p50": _percentile(sorted_word_counts, 0.50),
            "p90": _percentile(sorted_word_counts, 0.90),
            "max": max(sorted_word_counts) if sorted_word_counts else None,
        },
        "response_characters": {
            "min": min(sorted_char_counts) if sorted_char_counts else None,
            "p50": _percentile(sorted_char_counts, 0.50),
            "p90": _percentile(sorted_char_counts, 0.90),
            "max": max(sorted_char_counts) if sorted_char_counts else None,
        },
        "reference_metrics": rouge_summary,
        "clipped_response_count": num_clipped,
        "template_clipped_response_count": num_template_clipped,
        "warning_counts": dict(sorted(warning_counts.items())),
        "flagged_samples": flagged_samples,
        "flagged_sample_count": flagged_sample_count,
        "flagged_samples_truncated": flagged_sample_count > len(flagged_samples),
    }


def select_stratified_sample_ids(
    rows: Sequence[Mapping[str, Any]],
    sample_lengths: Mapping[str, int],
    *,
    sample_count: int,
    seed: int = 42,
) -> list[str]:
    """Chọn xác định theo source và bốn quantile độ dài trong từng source."""
    if sample_count < 0:
        raise ValueError("sample_count phải >= 0")
    if sample_count == 0 or not rows:
        return []

    by_source: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row.get("id", ""))
        if sample_id not in sample_lengths:
            raise ValueError(f"không có cache length cho sample {sample_id!r}")
        by_source[str(row.get("source", "unknown"))].append(row)

    strata: Dict[tuple[str, int], list[str]] = {}
    for source, source_rows in sorted(by_source.items()):
        ordered = sorted(
            source_rows,
            key=lambda row: (int(sample_lengths[str(row["id"])]), str(row["id"])),
        )
        for rank, row in enumerate(ordered):
            bucket = min(3, (rank * 4) // len(ordered))
            strata.setdefault((source, bucket), []).append(str(row["id"]))

    rng = random.Random(seed)
    keys = sorted(strata)
    rng.shuffle(keys)
    for ids in strata.values():
        rng.shuffle(ids)

    selected: list[str] = []
    while len(selected) < min(sample_count, len(rows)):
        made_progress = False
        for key in keys:
            if strata[key]:
                selected.append(strata[key].pop())
                made_progress = True
                if len(selected) >= sample_count:
                    break
        if not made_progress:
            break
    return selected


def compare_hidden_values(
    cached: torch.Tensor,
    recomputed: torch.Tensor,
    *,
    atol: float = 0.05,
    rtol: float = 0.05,
) -> Dict[str, Any]:
    """So sánh mọi feature ở mọi offset cho một sample."""
    cached = torch.as_tensor(cached).detach().cpu()
    recomputed = torch.as_tensor(recomputed).detach().cpu()
    if tuple(cached.shape) != tuple(recomputed.shape):
        return {
            "allclose": False,
            "cached_shape": list(cached.shape),
            "recomputed_shape": list(recomputed.shape),
            "max_abs_error": None,
            "max_relative_error": None,
            "cosine_similarity": None,
        }
    if cached.numel() == 0:
        return {
            "allclose": False,
            "cached_shape": list(cached.shape),
            "recomputed_shape": list(recomputed.shape),
            "max_abs_error": None,
            "max_relative_error": None,
            "cosine_similarity": None,
        }
    if not torch.isfinite(cached).all().item() or not torch.isfinite(recomputed).all().item():
        return {
            "allclose": False,
            "cached_shape": list(cached.shape),
            "recomputed_shape": list(recomputed.shape),
            "max_abs_error": None,
            "max_relative_error": None,
            "cosine_similarity": None,
            "non_finite": True,
        }

    cached_flat = cached.reshape(-1)
    recomputed_flat = recomputed.reshape(-1)
    max_abs_error = 0.0
    max_relative_error = 0.0
    allclose = True
    dot = 0.0
    cached_norm_squared = 0.0
    recomputed_norm_squared = 0.0
    chunk_size = 1_000_000
    for start in range(0, cached_flat.numel(), chunk_size):
        cached_chunk = cached_flat[start : start + chunk_size].float()
        recomputed_chunk = recomputed_flat[start : start + chunk_size].float()
        difference = (cached_chunk - recomputed_chunk).abs()
        relative = difference / cached_chunk.abs().clamp_min(1e-8)
        max_abs_error = max(max_abs_error, float(difference.max()))
        max_relative_error = max(max_relative_error, float(relative.max()))
        allclose = allclose and bool(
            torch.allclose(cached_chunk, recomputed_chunk, atol=atol, rtol=rtol)
        )
        cached_double = cached_chunk.double()
        recomputed_double = recomputed_chunk.double()
        dot += float(torch.dot(cached_double, recomputed_double))
        cached_norm_squared += float(torch.dot(cached_double, cached_double))
        recomputed_norm_squared += float(torch.dot(recomputed_double, recomputed_double))
    if cached_norm_squared == 0 or recomputed_norm_squared == 0:
        cosine = 1.0 if torch.equal(cached_flat, recomputed_flat) else 0.0
    else:
        cosine = dot / math.sqrt(cached_norm_squared * recomputed_norm_squared)
    return {
        "allclose": allclose,
        "cached_shape": list(cached.shape),
        "recomputed_shape": list(recomputed.shape),
        "max_abs_error": max_abs_error,
        "max_relative_error": max_relative_error,
        "cosine_similarity": cosine,
    }


def _validate_manifest_compatibility(manifests: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    first_split = next(iter(manifests))
    baseline = dict(manifests[first_split])
    for split, manifest in manifests.items():
        for key in _MANIFEST_COMPATIBILITY_KEYS:
            if manifest.get(key) != baseline.get(key):
                raise ValueError(
                    f"manifest {split} không khớp {first_split} ở {key}: "
                    f"{manifest.get(key)!r} != {baseline.get(key)!r}"
                )
    target_path = str(baseline.get("target_model_path") or "")
    layer_ids = [int(value) for value in baseline.get("feature_layer_ids", [])]
    if not target_path or not layer_ids:
        raise ValueError("manifest cần target_model_path và feature_layer_ids")
    if int(baseline.get("feature_width", 0)) != int(baseline.get("hidden_size", 0)) * len(layer_ids):
        raise ValueError("manifest feature_width không bằng hidden_size * số feature layers")
    if str(baseline.get("requested_torch_dtype", "")) not in _DTYPES:
        raise ValueError(
            "requested_torch_dtype không hỗ trợ: "
            f"{baseline.get('requested_torch_dtype')!r}"
        )
    if str(baseline.get("capture_backend", "hf_backbone")) not in {
        "hf_backbone",
        "hf",
        "legacy",
    }:
        raise ValueError(
            "audit hidden trực tiếp hiện chỉ hỗ trợ HF capture backend; "
            f"manifest capture_backend={baseline.get('capture_backend')!r}"
        )
    return baseline


def _cache_lengths(manifest: Mapping[str, Any], cache_root: Path) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for descriptor in manifest.get("shards", []):
        ids = [str(value) for value in descriptor.get("ids", [])]
        lengths = [int(value) for value in descriptor.get("lengths", [])]
        if len(ids) == int(descriptor.get("count", -1)) and len(lengths) == len(ids):
            samples = zip(ids, lengths)
        else:
            path = cache_root / str(descriptor["path"])
            payload = torch.load(path, map_location="cpu", weights_only=False)
            shard_samples = payload.get("samples") if isinstance(payload, dict) else payload
            samples = (
                (str(sample.get("id", "")), int(sample.get("length", len(sample["input_ids"]))))
                for sample in shard_samples
            )
        for sample_id, length in samples:
            if not sample_id or sample_id in result:
                raise ValueError(f"manifest cache có ID rỗng/trùng: {sample_id!r}")
            result[sample_id] = length
    return result


def _validate_regenerated_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_generation_model: Optional[str],
) -> None:
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in seen:
            raise ValueError(f"regenerated có id rỗng/trùng: {sample_id!r}")
        seen.add(sample_id)
        _last_assistant_text(row)
        conversations = row.get("conversations") or []
        if not any(
            isinstance(message, dict)
            and str(message.get("role", "")).strip().lower() == "user"
            and str(message.get("content", "")).strip()
            for message in conversations[:-1]
        ):
            raise ValueError(f"sample {sample_id!r}: thiếu user prompt không rỗng")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or not metadata.get("generation_model"):
            raise ValueError(f"sample {sample_id!r}: thiếu metadata.generation_model")
        actual_model = str(metadata["generation_model"])
        if expected_generation_model and actual_model != expected_generation_model:
            raise ValueError(
                f"sample {sample_id!r}: generation_model={actual_model!r}, "
                f"kỳ vọng {expected_generation_model!r}"
            )


def _review_examples(
    rows: Sequence[Mapping[str, Any]],
    sample_lengths: Mapping[str, int],
    *,
    count: int,
    seed: int,
) -> list[Dict[str, Any]]:
    selected = set(
        select_stratified_sample_ids(rows, sample_lengths, sample_count=count, seed=seed)
    )
    examples = []
    for row in rows:
        sample_id = str(row["id"])
        if sample_id not in selected:
            continue
        conversations = row["conversations"]
        user_text = next(
            (
                str(message.get("content", ""))
                for message in reversed(conversations[:-1])
                if isinstance(message, dict)
                and str(message.get("role", "")).lower() == "user"
            ),
            "",
        )
        metadata = row.get("metadata") or {}
        examples.append(
            {
                "id": sample_id,
                "source": str(row.get("source", "unknown")),
                "token_length": int(sample_lengths[sample_id]),
                "user_preview": user_text[:500],
                "response_preview": _last_assistant_text(row)[:1000],
                "reference_preview": str(metadata.get("reference_summary", ""))[:1000],
            }
        )
    return examples


def _iter_selected_cache_samples(
    cache_root: Path,
    manifest: Mapping[str, Any],
    selected_ids: set[str],
) -> Iterable[Dict[str, Any]]:
    """Yield selected samples while retaining at most one shard in memory."""
    remaining = set(selected_ids)
    for descriptor in manifest.get("shards", []):
        descriptor_ids = {str(value) for value in descriptor.get("ids", [])}
        if descriptor_ids and not (descriptor_ids & remaining):
            continue
        shard_path = cache_root / str(descriptor["path"])
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        samples = payload.get("samples") if isinstance(payload, dict) else payload
        for sample in samples:
            sample_id = str(sample.get("id", ""))
            if sample_id in remaining:
                remaining.remove(sample_id)
                yield sample
        del payload
        del samples
        if not remaining:
            break
    if remaining:
        raise ValueError(f"không tìm thấy selected sample trong cache: {sorted(remaining)[:5]}")


def _make_capture(manifest: Mapping[str, Any], device: str, local_files_only: bool):
    from MR_DFlash.capture import HFTargetCapture

    return HFTargetCapture(
        str(manifest["target_model_path"]),
        [int(value) for value in manifest["feature_layer_ids"]],
        torch_dtype=str(manifest["requested_torch_dtype"]),
        device=device,
        local_files_only=local_files_only,
        target_revision=manifest.get("target_revision"),
        attention_backend=str(manifest.get("attention_backend") or "auto"),
    )


def _recompute_hidden_for_split(
    *,
    cache_root: Path,
    manifest: Mapping[str, Any],
    selected_ids: Sequence[str],
    capture: Any,
    atol: float,
    rtol: float,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    selected_set = set(selected_ids)
    sample_reports = []
    failed = 0
    for cached_sample in _iter_selected_cache_samples(cache_root, manifest, selected_set):
        sample_id = str(cached_sample["id"])
        input_ids = torch.as_tensor(cached_sample["input_ids"], dtype=torch.long).flatten()
        recomputed = capture.capture_one(input_ids.tolist())
        if recomputed.ndim == 3 and recomputed.shape[0] == 1:
            recomputed = recomputed[0]
        comparison = compare_hidden_values(
            cached_sample["hidden_states"],
            recomputed,
            atol=atol,
            rtol=rtol,
        )
        if not comparison["allclose"]:
            failed += 1
        sample_reports.append({"id": sample_id, **comparison})
        if progress_callback is not None:
            progress_callback(len(sample_reports), len(selected_ids))
        del cached_sample
        del recomputed
    selection_order = {sample_id: index for index, sample_id in enumerate(selected_ids)}
    sample_reports.sort(key=lambda item: selection_order[item["id"]])
    return {
        "requested_samples": len(selected_ids),
        "checked_samples": len(sample_reports),
        "failed_samples": failed,
        "atol": float(atol),
        "rtol": float(rtol),
        "samples": sample_reports,
        "status": "pass" if failed == 0 and sample_reports else "fail",
    }


def audit_run_root(
    run_root: str | Path,
    *,
    splits: Sequence[str] = ("train", "val"),
    regenerated_dir: str = "regenerated_full",
    feature_dir: str = "target_features_qwen3_4b_full",
    hidden_samples_per_split: int = 32,
    review_examples_per_split: int = 8,
    seed: int = 42,
    device: str = "cuda:0",
    atol: float = 0.05,
    rtol: float = 0.05,
    min_response_tokens: int = 4,
    repetition_ngram_size: int = 4,
    repetition_warning_ratio: float = 0.30,
    expected_generation_model: Optional[str] = None,
    local_files_only: bool = True,
    tokenizer: Any = None,
    capture_factory: Optional[Callable[[Mapping[str, Any], str], Any]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Audit toàn bộ regenerated/cache và đối chiếu hidden trên mẫu phân tầng."""
    root = Path(run_root)
    normalized_splits = tuple(str(split) for split in splits)
    if not normalized_splits or any(split not in {"train", "val", "test"} for split in normalized_splits):
        raise ValueError("splits phải là một tập con không rỗng của train/val/test")
    if len(set(normalized_splits)) != len(normalized_splits):
        raise ValueError("split bị lặp")
    if hidden_samples_per_split < 1:
        raise ValueError("hidden_samples_per_split phải >= 1")
    if review_examples_per_split < 0:
        raise ValueError("review_examples_per_split phải >= 0")
    if atol < 0 or rtol < 0:
        raise ValueError("atol/rtol phải >= 0")

    cache_locations: Dict[str, tuple[Path, Path, Dict[str, Any]]] = {}
    manifests: Dict[str, Dict[str, Any]] = {}
    for split in normalized_splits:
        data_path = root / regenerated_dir / f"{split}.jsonl"
        cache_path = root / feature_dir / split
        if not data_path.is_file():
            raise FileNotFoundError(f"thiếu regenerated JSONL: {data_path}")
        cache_root, manifest = _load_manifest(cache_path)
        manifests[split] = manifest
        cache_locations[split] = (data_path, cache_root, manifest)
    baseline_manifest = _validate_manifest_compatibility(manifests)

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(baseline_manifest["target_model_path"]),
            local_files_only=local_files_only,
        )

    result: Dict[str, Any] = {
        "schema_version": "mr_dflash_regenerated_cache_quality_v1",
        "status": "pass",
        "run_root": str(root),
        "splits": {},
        "settings": {
            "splits": list(normalized_splits),
            "hidden_samples_per_split": int(hidden_samples_per_split),
            "review_examples_per_split": int(review_examples_per_split),
            "seed": int(seed),
            "device": str(device),
            "atol": float(atol),
            "rtol": float(rtol),
            "target_model_path": str(baseline_manifest["target_model_path"]),
            "feature_layer_ids": [int(value) for value in baseline_manifest["feature_layer_ids"]],
            "max_length": int(baseline_manifest["max_length"]),
            "torch_dtype": str(baseline_manifest["requested_torch_dtype"]),
            "attention_backend": baseline_manifest.get("attention_backend"),
            "expected_generation_model": expected_generation_model or str(baseline_manifest["target_model_path"]),
        },
    }
    all_selected: Dict[str, list[str]] = {}
    effective_expected_model = expected_generation_model or str(baseline_manifest["target_model_path"])

    for split in normalized_splits:
        data_path, cache_root, manifest = cache_locations[split]
        cache_path = root / feature_dir / split
        if progress_callback is not None:
            progress_callback(f"cache {split}: bắt đầu quét JSONL, shard, tensor và token parity")

        def report_cache_progress(done: int, total: int, *, current_split: str = split) -> None:
            if progress_callback is not None:
                progress_callback(f"cache {current_split}: {done}/{total} samples")

        cache_report = validate_feature_cache(
            data_path=data_path,
            cache_path=cache_path,
            tokenizer=tokenizer,
            max_length=int(manifest["max_length"]),
            supervision_mode="last_assistant",
            expected_target_model=str(manifest["target_model_path"]),
            expected_feature_layer_ids=[int(value) for value in manifest["feature_layer_ids"]],
            progress_callback=report_cache_progress,
        )
        rows = list(read_jsonl(data_path))
        _validate_regenerated_rows(
            rows,
            expected_generation_model=effective_expected_model,
        )
        lengths = _cache_lengths(manifest, cache_root)
        if set(lengths) != {str(row["id"]) for row in rows}:
            raise ValueError(f"{split}: cache length coverage không khớp regenerated")
        quality = analyze_regenerated_quality(
            rows,
            min_response_tokens=min_response_tokens,
            repetition_ngram_size=repetition_ngram_size,
            repetition_warning_ratio=repetition_warning_ratio,
        )
        quality["review_examples"] = _review_examples(
            rows,
            lengths,
            count=min(review_examples_per_split, len(rows)),
            seed=seed + normalized_splits.index(split),
        )
        selected_ids = select_stratified_sample_ids(
            rows,
            lengths,
            sample_count=min(hidden_samples_per_split, len(rows)),
            seed=seed + normalized_splits.index(split),
        )
        all_selected[split] = selected_ids
        if progress_callback is not None:
            progress_callback(
                f"cache {split}: hoàn tất; bắt đầu so hidden cho {len(selected_ids)} mẫu"
            )
        result["splits"][split] = {
            "regenerated": {
                "path": str(data_path),
                "num_samples": len(rows),
                "generation_model_counts": quality["generation_model_counts"],
            },
            "cache": cache_report,
            "quality": quality,
            "hidden_recompute": None,
        }

    capture = None
    if capture_factory is None:
        capture = _make_capture(baseline_manifest, device, local_files_only)
    else:
        capture = capture_factory(baseline_manifest, device)
    try:
        for split in normalized_splits:
            _data_path, cache_root, manifest = cache_locations[split]
            hidden_report = _recompute_hidden_for_split(
                cache_root=cache_root,
                manifest=manifest,
                selected_ids=all_selected[split],
                capture=capture,
                atol=atol,
                rtol=rtol,
                progress_callback=(
                    (lambda done, total, current_split=split: progress_callback(
                        f"hidden {current_split}: {done}/{total} samples"
                    ))
                    if progress_callback is not None
                    else None
                ),
            )
            result["splits"][split]["hidden_recompute"] = hidden_report
            if hidden_report["status"] != "pass":
                result["status"] = "fail"
    finally:
        close = getattr(capture, "close", None)
        if callable(close):
            close()

    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit regenerated JSONL và target hidden cache MR-DFlash"
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val"))
    parser.add_argument("--regenerated-dir", default="regenerated_full")
    parser.add_argument("--feature-dir", default="target_features_qwen3_4b_full")
    parser.add_argument("--hidden-samples-per-split", type=int, default=32)
    parser.add_argument("--review-examples-per-split", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0", help="mặc định chỉ dùng một GPU, ví dụ cuda:0")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--min-response-tokens", type=int, default=4)
    parser.add_argument("--repetition-ngram-size", type=int, default=4)
    parser.add_argument("--repetition-warning-ratio", type=float, default=0.30)
    parser.add_argument("--expected-generation-model", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.run_root)
    report_path = Path(args.report) if args.report else root / "manifests" / "quality_audit.json"
    try:
        report = audit_run_root(
            root,
            splits=args.splits,
            regenerated_dir=args.regenerated_dir,
            feature_dir=args.feature_dir,
            hidden_samples_per_split=args.hidden_samples_per_split,
            review_examples_per_split=args.review_examples_per_split,
            seed=args.seed,
            device=args.device,
            atol=args.atol,
            rtol=args.rtol,
            min_response_tokens=args.min_response_tokens,
            repetition_ngram_size=args.repetition_ngram_size,
            repetition_warning_ratio=args.repetition_warning_ratio,
            expected_generation_model=args.expected_generation_model,
            local_files_only=bool(args.local_files_only),
            progress_callback=lambda message: print(
                f"[audit_regenerated_cache_quality] {message}", flush=True
            ),
        )
    except Exception as exc:
        report = {
            "schema_version": "mr_dflash_regenerated_cache_quality_v1",
            "status": "fail",
            "run_root": str(root),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json(report_path, report)
        print(f"[audit_regenerated_cache_quality] FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"[audit_regenerated_cache_quality] report={report_path}", file=sys.stderr)
        return 1

    write_json(report_path, report)
    total_samples = sum(split["regenerated"]["num_samples"] for split in report["splits"].values())
    checked_hidden = sum(split["hidden_recompute"]["checked_samples"] for split in report["splits"].values())
    warnings = sum(
        sum(split["quality"]["warning_counts"].values())
        for split in report["splits"].values()
    )
    print(
        f"[audit_regenerated_cache_quality] status={report['status']} "
        f"samples={total_samples} hidden_recomputed={checked_hidden} warnings={warnings}"
    )
    print(f"[audit_regenerated_cache_quality] report={report_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
