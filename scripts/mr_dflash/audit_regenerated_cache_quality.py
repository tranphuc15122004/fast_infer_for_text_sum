"""Audit chất lượng regenerated samples và MR-DFlash hidden cache.

Script kiểm tra metadata trước khi mở shard, sau đó quét cấu trúc, coverage,
token/mask parity và dấu hiệu response đáng ngờ. Cuối cùng nó chạy lại target
backbone cho một mẫu phân tầng nhỏ trên một GPU và so sánh hidden với cache.
Checkpoint theo shard/sample giúp lần chạy sau resume phần đã hoàn tất. Alias
vLLM được lấy từ pipeline_plan.json khi có.

Ví dụ trên B200::

    python3 scripts/mr_dflash/audit_regenerated_cache_quality.py \
      --run-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/run \
      --device cuda:0 --hidden-samples-per-split 32 --resume

Có thể tách phần quét cache khỏi hidden recompute: chạy lần đầu với
``--skip-hidden-recompute``; sau đó chạy lại cùng lệnh bỏ cờ này để resume
cache và chỉ thực hiện hidden recompute còn thiếu.

Các cảnh báo nội dung chỉ nhằm hỗ trợ review; lỗi cấu trúc hoặc hidden
recompute vượt tolerance làm lệnh trả mã khác 0. Script không sửa dữ liệu
regenerate/cache; nó chỉ ghi report và checkpoint trong run root.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
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


def _file_signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "inode": int(stat.st_ino),
    }


def _target_asset_signature(target_model_path: str) -> list[Dict[str, Any]]:
    """Stat tokenizer/config/weight files without reading large model weights."""
    root = Path(target_model_path)
    if not root.is_dir():
        return []
    candidates: set[Path] = set()
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        path = root / name
        if path.is_file():
            candidates.add(path)
    candidates.update(path for path in root.glob("model-*.safetensors") if path.is_file())
    candidates.update(path for path in root.glob("pytorch_model-*.bin") if path.is_file())
    return [_file_signature(path) for path in sorted(candidates)]


def _split_cache_identity(
    *,
    data_path: Path,
    cache_root: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Identity for receipts; shard contents use per-file stat fingerprints."""
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    target_model_path = str(manifest["target_model_path"])
    return {
        "data": _file_signature(data_path),
        "manifest": _file_signature(manifest_path),
        "manifest_sha256": manifest_digest,
        "cache_root": str(cache_root.resolve()),
        "target_model_path": target_model_path,
        "target_revision": manifest.get("target_revision"),
        "target_assets": _target_asset_signature(target_model_path),
        "feature_layer_ids": [int(value) for value in manifest.get("feature_layer_ids", [])],
        "max_length": int(manifest.get("max_length", 0)),
        "supervision_mode": "last_assistant",
    }


def _load_audit_checkpoint(path: Optional[Path], run_root: Path, *, resume: bool) -> Dict[str, Any]:
    if not resume or path is None or not path.is_file():
        return {
            "schema_version": "mr_dflash_quality_checkpoint_v1",
            "run_root": str(run_root.resolve()),
            "splits": {},
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": "mr_dflash_quality_checkpoint_v1",
            "run_root": str(run_root.resolve()),
            "splits": {},
            "resume_note": "checkpoint không đọc được; bắt đầu audit từ đầu",
        }
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "mr_dflash_quality_checkpoint_v1"
        or payload.get("run_root") != str(run_root.resolve())
    ):
        return {
            "schema_version": "mr_dflash_quality_checkpoint_v1",
            "run_root": str(run_root.resolve()),
            "splits": {},
            "resume_note": "checkpoint không khớp schema/run-root; bắt đầu audit từ đầu",
        }
    if not isinstance(payload.get("splits"), dict):
        payload["splits"] = {}
    return payload


def _resolve_expected_generation_model(
    run_root: Path,
    target_model_path: str,
    explicit_value: Optional[str],
) -> tuple[str, str]:
    """Resolve the generation label separately from the local target path."""
    plan_path = run_root / "pipeline_plan.json"
    plan: Dict[str, Any] = {}
    if plan_path.is_file():
        try:
            parsed = json.loads(plan_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                plan = parsed
        except (OSError, json.JSONDecodeError):
            plan = {}
    options = plan.get("options") if isinstance(plan.get("options"), dict) else {}
    planned_target = str(options.get("target_model_path", ""))
    if planned_target and planned_target != target_model_path:
        raise ValueError(
            "pipeline_plan target_model_path không khớp feature cache: "
            f"{planned_target!r} != {target_model_path!r}"
        )
    if explicit_value:
        return str(explicit_value), "--expected-generation-model"
    if str(options.get("regenerate_backend", "")) == "vllm":
        served_model = options.get("vllm_model")
        if served_model:
            return str(served_model), "pipeline_plan.options.vllm_model"
    return target_model_path, "feature_cache_manifest.target_model_path"


def _validate_regenerated_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_generation_model: Optional[str],
) -> Dict[str, Any]:
    seen: set[str] = set()
    errors: Counter[str] = Counter()
    examples: list[Dict[str, str]] = []
    generation_models: Counter[str] = Counter()

    def record_error(kind: str, sample_id: str, detail: str) -> None:
        errors[kind] += 1
        if len(examples) < 50:
            examples.append({"id": sample_id, "kind": kind, "detail": detail})

    for row in rows:
        sample_id = str(row.get("id", ""))
        if not sample_id:
            record_error("empty_id", sample_id, "id rỗng")
        elif sample_id in seen:
            record_error("duplicate_id", sample_id, "id bị trùng")
        else:
            seen.add(sample_id)
        try:
            _last_assistant_text(row)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            record_error("assistant_message", sample_id, str(exc))
        conversations = row.get("conversations")
        if not isinstance(conversations, list):
            conversations = []
        if not any(
            isinstance(message, dict)
            and str(message.get("role", "")).strip().lower() == "user"
            and str(message.get("content", "")).strip()
            for message in conversations[:-1]
        ):
            record_error("user_prompt", sample_id, "thiếu user prompt không rỗng")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or not metadata.get("generation_model"):
            record_error("generation_model_missing", sample_id, "thiếu metadata.generation_model")
            continue
        actual_model = str(metadata["generation_model"])
        generation_models[actual_model] += 1
        if expected_generation_model and actual_model != expected_generation_model:
            record_error(
                "generation_model_mismatch",
                sample_id,
                f"generation_model={actual_model!r}, kỳ vọng {expected_generation_model!r}",
            )
    return {
        "valid": not errors,
        "num_samples": len(rows),
        "generation_model_counts": dict(sorted(generation_models.items())),
        "error_counts": dict(sorted(errors.items())),
        "errors": examples,
        "errors_truncated": sum(errors.values()) > len(examples),
    }


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
    if not selected_ids:
        return
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
    existing_reports: Optional[Mapping[str, Mapping[str, Any]]] = None,
    sample_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    selected_set = set(selected_ids)
    reports_by_id: Dict[str, Dict[str, Any]] = {
        str(sample_id): dict(report)
        for sample_id, report in (existing_reports or {}).items()
        if str(sample_id) in selected_set and isinstance(report, Mapping)
    }
    pending_ids = [sample_id for sample_id in selected_ids if sample_id not in reports_by_id]
    pending_set = set(pending_ids)
    for cached_sample in _iter_selected_cache_samples(cache_root, manifest, pending_set):
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
        sample_report = {"id": sample_id, **comparison}
        reports_by_id[sample_id] = sample_report
        if sample_callback is not None:
            sample_callback(sample_report)
        if progress_callback is not None:
            progress_callback(len(reports_by_id), len(selected_ids))
        del cached_sample
        del recomputed
    missing = selected_set - set(reports_by_id)
    if missing:
        raise ValueError(f"chưa đối chiếu hidden cho sample: {sorted(missing)[:5]}")
    sample_reports = [reports_by_id[sample_id] for sample_id in selected_ids]
    failed = sum(not bool(report.get("allclose")) for report in sample_reports)
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
    report_path: Optional[str | Path] = None,
    checkpoint_path: Optional[str | Path] = None,
    resume: bool = True,
    skip_hidden_recompute: bool = False,
) -> Dict[str, Any]:
    """Audit regenerated/cache theo phase, có checkpoint để resume an toàn."""
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

    cache_locations: Dict[str, tuple[Path, Path, Dict[str, Any], Path]] = {}
    manifests: Dict[str, Dict[str, Any]] = {}
    for split in normalized_splits:
        data_path = root / regenerated_dir / f"{split}.jsonl"
        cache_path = root / feature_dir / split
        if not data_path.is_file():
            raise FileNotFoundError(f"thiếu regenerated JSONL: {data_path}")
        cache_root, manifest = _load_manifest(cache_path)
        manifest_path = cache_path if cache_path.is_file() else cache_path / "manifest.json"
        manifests[split] = manifest
        cache_locations[split] = (data_path, cache_root, manifest, manifest_path)
    baseline_manifest = _validate_manifest_compatibility(manifests)
    effective_expected_model, expected_model_source = _resolve_expected_generation_model(
        root,
        str(baseline_manifest["target_model_path"]),
        expected_generation_model,
    )

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(baseline_manifest["target_model_path"]),
            local_files_only=local_files_only,
        )

    report_file = Path(report_path) if report_path is not None else None
    checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint = _load_audit_checkpoint(checkpoint_file, root, resume=resume)
    checkpoint.pop("last_error", None)
    result: Dict[str, Any] = {
        "schema_version": "mr_dflash_regenerated_cache_quality_v1",
        "status": "running",
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
            "expected_generation_model": effective_expected_model,
            "expected_generation_model_source": expected_model_source,
            "skip_hidden_recompute": bool(skip_hidden_recompute),
            "resume": bool(resume),
        },
    }

    def save_checkpoint() -> None:
        if checkpoint_file is not None:
            write_json(checkpoint_file, checkpoint)

    def save_report() -> None:
        if report_file is not None:
            result["checkpoint_path"] = str(checkpoint_file) if checkpoint_file is not None else None
            write_json(report_file, result)

    if checkpoint.get("resume_note"):
        result["resume_note"] = checkpoint["resume_note"]
    save_report()

    # Preflight every split before opening a feature shard. This catches cheap,
    # deterministic metadata/schema errors before the expensive tensor scan.
    preflight_errors: list[str] = []
    for split in normalized_splits:
        data_path, cache_root, manifest, manifest_path = cache_locations[split]
        split_identity = _split_cache_identity(
            data_path=data_path,
            cache_root=cache_root,
            manifest_path=manifest_path,
            manifest=manifest,
        )
        split_state = checkpoint["splits"].get(split, {})
        if not isinstance(split_state, dict) or split_state.get("identity") != split_identity:
            split_state = {
                "identity": split_identity,
                "cache_shards": {},
                "cache_complete": False,
                "hidden_identity": None,
                "hidden_samples": {},
            }
            checkpoint["splits"][split] = split_state
        for state_key in ("cache_shards", "hidden_samples"):
            if not isinstance(split_state.get(state_key), dict):
                split_state[state_key] = {}

        preflight_identity = {
            "split_identity": split_identity,
            "expected_generation_model": effective_expected_model,
            "min_response_tokens": int(min_response_tokens),
            "repetition_ngram_size": int(repetition_ngram_size),
            "repetition_warning_ratio": float(repetition_warning_ratio),
            "review_examples_per_split": int(review_examples_per_split),
            "seed": int(seed + normalized_splits.index(split)),
            "hidden_samples_per_split": int(hidden_samples_per_split),
        }
        preflight_reused = bool(
            resume
            and split_state.get("preflight_identity") == preflight_identity
            and isinstance(split_state.get("preflight"), dict)
            and isinstance(split_state.get("quality"), dict)
            and isinstance(split_state.get("selected_ids"), list)
        )
        if preflight_reused:
            preflight = dict(split_state["preflight"])
            quality = dict(split_state["quality"])
            selected_ids = [str(value) for value in split_state["selected_ids"]]
            if progress_callback is not None:
                progress_callback(f"preflight {split}: resume từ checkpoint")
        else:
            rows = list(read_jsonl(data_path))
            preflight = _validate_regenerated_rows(
                rows,
                expected_generation_model=effective_expected_model,
            )
            quality: Dict[str, Any] = {}
            selected_ids: list[str] = []
            lengths: Dict[str, int] = {}
            if preflight["valid"]:
                lengths = _cache_lengths(manifest, cache_root)
                if set(lengths) != {str(row["id"]) for row in rows}:
                    preflight["valid"] = False
                    preflight["error_counts"] = {"cache_length_coverage": 1}
                    preflight["errors"] = [
                        {
                            "id": "",
                            "kind": "cache_length_coverage",
                            "detail": "cache length coverage không khớp regenerated",
                        }
                    ]
                else:
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
            split_state["preflight_identity"] = preflight_identity
            split_state["preflight"] = preflight
            split_state["quality"] = quality
            split_state["selected_ids"] = selected_ids

        result["splits"][split] = {
            "preflight": preflight,
            "regenerated": {
                "path": str(data_path),
                "num_samples": int(preflight.get("num_samples", 0)),
                "generation_model_counts": preflight.get("generation_model_counts", {}),
            },
            "quality": quality if quality else None,
            "cache": None,
            "hidden_recompute": None,
        }
        save_checkpoint()
        save_report()
        if not preflight.get("valid"):
            preflight_errors.append(
                f"{split}: {preflight.get('error_counts', {})}; "
                f"ví dụ={preflight.get('errors', [])[:3]}"
            )
        elif progress_callback is not None:
            progress_callback(
                f"preflight {split}: pass, {preflight.get('num_samples', 0)} samples; "
                f"generation_model={preflight.get('generation_model_counts', {})}"
            )

    if preflight_errors:
        checkpoint["last_error"] = "regenerated preflight failed"
        save_checkpoint()
        raise ValueError(
            "regenerated preflight thất bại trước khi quét cache: "
            + " | ".join(preflight_errors)
        )

    for split in normalized_splits:
        data_path, cache_root, manifest, manifest_path = cache_locations[split]
        cache_path = root / feature_dir / split
        split_state = checkpoint["splits"][split]
        current_identity = _split_cache_identity(
            data_path=data_path,
            cache_root=cache_root,
            manifest_path=manifest_path,
            manifest=manifest,
        )
        if split_state.get("identity") != current_identity:
            raise RuntimeError(f"{split}: input/manifest/model thay đổi giữa preflight và cache audit")
        rows = list(read_jsonl(data_path))
        if progress_callback is not None:
            progress_callback(f"cache {split}: kiểm tra shard/tensor/token parity")

        def report_cache_progress(done: int, total: int, *, current_split: str = split) -> None:
            if progress_callback is not None:
                progress_callback(f"cache {current_split}: {done}/{total} samples")

        def save_shard_receipt(
            shard_key: str,
            receipt: Dict[str, Any],
            *,
            current_state: Dict[str, Any] = split_state,
        ) -> None:
            current_state.setdefault("cache_shards", {})[shard_key] = receipt
            current_state["cache_complete"] = False
            save_checkpoint()

        cache_report = validate_feature_cache(
            data_path=data_path,
            cache_path=cache_path,
            tokenizer=tokenizer,
            max_length=int(manifest["max_length"]),
            supervision_mode="last_assistant",
            expected_target_model=str(manifest["target_model_path"]),
            expected_feature_layer_ids=[int(value) for value in manifest["feature_layer_ids"]],
            progress_callback=report_cache_progress,
            input_rows=rows,
            resume_shards=split_state.get("cache_shards", {}) if resume else None,
            shard_callback=save_shard_receipt,
        )
        split_state["cache_report"] = cache_report
        split_state["cache_complete"] = True
        result["splits"][split]["cache"] = cache_report
        save_checkpoint()
        save_report()

    if skip_hidden_recompute:
        result["status"] = "partial"
        result["hidden_recompute_skipped"] = True
        save_checkpoint()
        save_report()
        return result

    capture = None
    try:
        for split in normalized_splits:
            data_path, cache_root, manifest, _manifest_path = cache_locations[split]
            split_state = checkpoint["splits"][split]
            selected_ids = [str(value) for value in split_state.get("selected_ids", [])]
            shard_fingerprints = {
                str(descriptor["path"]): _file_signature(cache_root / str(descriptor["path"]))
                for descriptor in manifest.get("shards", [])
            }
            hidden_identity = {
                "split_identity": split_state["identity"],
                "shards": shard_fingerprints,
                "selected_ids": selected_ids,
                "atol": float(atol),
                "rtol": float(rtol),
                "device": str(device),
            }
            if split_state.get("hidden_identity") != hidden_identity:
                split_state["hidden_identity"] = hidden_identity
                split_state["hidden_samples"] = {}
            existing_reports = split_state.setdefault("hidden_samples", {}) if resume else {}
            completed_hidden_ids = set(map(str, existing_reports)) & set(selected_ids)
            if len(completed_hidden_ids) < len(selected_ids) and capture is None:
                if capture_factory is None:
                    capture = _make_capture(baseline_manifest, device, local_files_only)
                else:
                    capture = capture_factory(baseline_manifest, device)

            if progress_callback is not None:
                progress_callback(
                    f"hidden {split}: {len(existing_reports)}/{len(selected_ids)} đã có checkpoint"
                )

            def save_hidden_sample(
                sample_report: Dict[str, Any],
                *,
                current_state: Dict[str, Any] = split_state,
            ) -> None:
                current_state.setdefault("hidden_samples", {})[str(sample_report["id"])] = sample_report
                save_checkpoint()

            hidden_report = _recompute_hidden_for_split(
                cache_root=cache_root,
                manifest=manifest,
                selected_ids=selected_ids,
                capture=capture,
                atol=atol,
                rtol=rtol,
                existing_reports=existing_reports,
                sample_callback=save_hidden_sample,
                progress_callback=(
                    (lambda done, total, current_split=split: progress_callback(
                        f"hidden {current_split}: {done}/{total} samples"
                    ))
                    if progress_callback is not None
                    else None
                ),
            )
            split_state["hidden_report"] = hidden_report
            result["splits"][split]["hidden_recompute"] = hidden_report
            if hidden_report["status"] != "pass":
                result["status"] = "fail"
            save_checkpoint()
            save_report()
    finally:
        close = getattr(capture, "close", None)
        if callable(close):
            close()

    if result["status"] != "fail":
        result["status"] = "pass"
    checkpoint.pop("last_error", None)
    save_checkpoint()
    save_report()
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
    parser.add_argument("--checkpoint", default=None, help="checkpoint resume; mặc định cạnh report")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume các shard/sample đã kiểm tra nếu artifact không đổi (mặc định bật)",
    )
    parser.add_argument(
        "--skip-hidden-recompute",
        action="store_true",
        help="chỉ chạy preflight + integrity; chạy lại cùng lệnh bỏ flag này để so hidden",
    )
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _partial_report(
    report_path: Path,
    checkpoint_path: Path,
    *,
    include_checkpoint: bool,
) -> Dict[str, Any]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            report = {}
    except (OSError, json.JSONDecodeError):
        report = {}
    try:
        checkpoint = (
            json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if include_checkpoint
            else {}
        )
    except (OSError, json.JSONDecodeError):
        checkpoint = {}
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    if (
        checkpoint.get("schema_version") != "mr_dflash_quality_checkpoint_v1"
        or checkpoint.get("run_root") != str(Path(report.get("run_root", "")).resolve())
    ):
        checkpoint = {}
    split_progress = {}
    for split, state in (checkpoint.get("splits", {}) or {}).items():
        if not isinstance(state, dict):
            continue
        split_progress[str(split)] = {
            "cache_complete": bool(state.get("cache_complete", False)),
            "cache_shards_completed": len(state.get("cache_shards", {}) or {}),
            "hidden_samples_completed": len(state.get("hidden_samples", {}) or {}),
        }
    report["checkpoint_path"] = str(checkpoint_path)
    report["checkpoint_progress"] = split_progress
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.run_root)
    report_path = Path(args.report) if args.report else root / "manifests" / "quality_audit.json"
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else report_path.with_name(f"{report_path.stem}.checkpoint.json")
    )
    if report_path.resolve() == checkpoint_path.resolve():
        print("--report và --checkpoint phải trỏ tới hai file khác nhau", file=sys.stderr)
        return 2
    write_json(
        report_path,
        {
            "schema_version": "mr_dflash_regenerated_cache_quality_v1",
            "status": "running",
            "run_root": str(root),
            "splits": {},
            "checkpoint_path": str(checkpoint_path),
        },
    )
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
            report_path=report_path,
            checkpoint_path=checkpoint_path,
            resume=bool(args.resume),
            skip_hidden_recompute=bool(args.skip_hidden_recompute),
            progress_callback=lambda message: print(
                f"[audit_regenerated_cache_quality] {message}", flush=True
            ),
        )
    except KeyboardInterrupt:
        report = _partial_report(
            report_path,
            checkpoint_path,
            include_checkpoint=bool(args.resume),
        )
        report.update(
            {
                "schema_version": "mr_dflash_regenerated_cache_quality_v1",
                "status": "interrupted",
                "run_root": str(root),
                "error_type": "KeyboardInterrupt",
                "error": "audit bị dừng; dùng lại cùng lệnh để resume checkpoint",
                "checkpoint_path": str(checkpoint_path),
            }
        )
        write_json(report_path, report)
        print(
            f"[audit_regenerated_cache_quality] INTERRUPTED; checkpoint={checkpoint_path}; report={report_path}",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        report = _partial_report(
            report_path,
            checkpoint_path,
            include_checkpoint=bool(args.resume),
        )
        report.update(
            {
                "schema_version": "mr_dflash_regenerated_cache_quality_v1",
                "status": "fail",
                "run_root": str(root),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "checkpoint_path": str(checkpoint_path),
            }
        )
        write_json(report_path, report)
        print(f"[audit_regenerated_cache_quality] FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"[audit_regenerated_cache_quality] report={report_path}", file=sys.stderr)
        print(f"[audit_regenerated_cache_quality] checkpoint={checkpoint_path}", file=sys.stderr)
        return 1

    total_samples = sum(
        int(split.get("regenerated", {}).get("num_samples", 0))
        for split in report["splits"].values()
    )
    checked_hidden = sum(
        int((split.get("hidden_recompute") or {}).get("checked_samples", 0))
        for split in report["splits"].values()
    )
    warnings = sum(
        sum((split.get("quality") or {}).get("warning_counts", {}).values())
        for split in report["splits"].values()
    )
    print(
        f"[audit_regenerated_cache_quality] status={report['status']} "
        f"samples={total_samples} hidden_recomputed={checked_hidden} warnings={warnings}"
    )
    print(f"[audit_regenerated_cache_quality] report={report_path}")
    print(f"[audit_regenerated_cache_quality] checkpoint={checkpoint_path}")
    if report["status"] == "pass":
        return 0
    if report["status"] == "partial" and report.get("hidden_recompute_skipped"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
