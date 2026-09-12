"""Regenerate assistant trajectories bằng frozen HF target.

Có hai input mode:
* ``--target-model-path``: chạy Qwen3/Llama trực tiếp offline;
* ``--responses-jsonl``: gắn response đã sinh bởi server bên ngoài, key theo id
  (hữu ích trên server không có internet nhưng có model service).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from _common import read_jsonl, write_json, write_jsonl
from auto_batch import AdaptiveBatchController
from MR_DFlash.generation_batching import left_pad_prompt_ids, select_generation_group
from progress import ProgressReporter, install_exception_hook


def _is_cuda_oom(exc: BaseException) -> bool:
    """OOM không được skip: CUDA context có thể đã ở trạng thái không an toàn."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _reset_peak_vram(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _peak_vram_gb(device: torch.device) -> Optional[float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize(device)
    reserved = torch.cuda.max_memory_reserved(device)
    allocated = torch.cuda.max_memory_allocated(device)
    return float(max(reserved, allocated)) / 2**30


def _generation_batch_key(item: Dict[str, Any]) -> str:
    """Bucket theo budget và prompt length để không trộn batch quá khác nhau."""
    prompt_tokens = max(1, int(item.get("prompt_tokens", 1)))
    prompt_bucket = ((prompt_tokens + 4095) // 4096) * 4096
    return f"budget={int(item['generation_budget'])}:prompt_bucket={prompt_bucket}"


def _count_shard_rows(
    input_path: str | Path,
    *,
    shard_index: int,
    num_shards: int,
    limit: int | None = None,
) -> int:
    """Đếm trước workload của worker để progress không hiển thị ``0/0``."""
    if num_shards < 1 or shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index phải nằm trong [0, num_shards)")
    total = 0
    for index, _row in enumerate(read_jsonl(input_path)):
        if index % num_shards != shard_index:
            continue
        total += 1
        if limit is not None and total >= int(limit):
            break
    return total


def resolve_generation_budget(
    prompt_tokens: int,
    max_length: int,
    requested_max_new_tokens: int,
) -> tuple[int, bool]:
    """Tính budget response mà không cắt prompt.

    ``max_length`` là giới hạn tổng prompt + response của target. Khi prompt
    dài, response được giới hạn ở phần context còn lại. Nếu prompt tự nó đã
    chiếm hết context thì không thể generate chính xác và caller phải skip
    hoặc fail theo policy.
    """
    prompt_tokens = int(prompt_tokens)
    max_length = int(max_length)
    requested_max_new_tokens = int(requested_max_new_tokens)
    if prompt_tokens < 0 or max_length < 1 or requested_max_new_tokens < 1:
        raise ValueError("prompt/max_length/max_new_tokens phải là số dương hợp lệ")
    available = max_length - prompt_tokens
    if available < 1:
        raise ValueError(
            f"prompt đã chiếm {prompt_tokens} tokens, không còn chỗ cho response "
            f"trong max_length={max_length}"
        )
    actual = min(requested_max_new_tokens, available)
    return actual, actual < requested_max_new_tokens


def _load_status_ids(path: Path) -> set[str]:
    """Đọc id đã hoàn thành và phục hồi một dòng cuối bị kill giữa lúc ghi.

    Chỉ tự sửa một JSON line cuối *không có newline kết thúc*. JSON lỗi ở giữa
    file hoặc một dòng lỗi đã kết thúc newline là corruption thật và phải làm
    pipeline dừng để không silently mất dữ liệu.
    """
    if not path.exists():
        return set()
    raw = path.read_text(encoding="utf-8")
    if not raw:
        return set()
    lines = raw.splitlines()
    has_terminal_newline = raw.endswith(("\n", "\r"))
    ids: set[str] = set()
    valid_lines: list[str] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            valid_lines.append(line)
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            is_trailing_partial = line_number == len(lines) and not has_terminal_newline
            if not is_trailing_partial:
                raise ValueError(f"JSONL hỏng tại {path}:{line_number}") from exc
            temporary = path.with_name(f".{path.name}.repair.tmp")
            repaired = "\n".join(valid_lines)
            if repaired:
                repaired += "\n"
            temporary.write_text(repaired, encoding="utf-8")
            os.replace(temporary, path)
            break
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} phải là JSON object")
        sample_id = str(row.get("id", ""))
        if sample_id:
            ids.add(sample_id)
        valid_lines.append(line)
    return ids


def _append_jsonl_durable(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Append batch JSONL rồi flush/fsync để resume không bỏ mất batch cuối."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() > 0:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
        for row in rows:
            handle.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def _ensure_durable_empty_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.flush()
        os.fsync(handle.fileno())


class _ProgressLogitsProcessor:
    """Heartbeat mỗi vài token, không thay đổi logits hay output target."""

    def __init__(
        self,
        reporter: ProgressReporter,
        *,
        prompt_tokens: int,
        sample_id: str,
        budget: int,
        batch_size: int = 1,
    ) -> None:
        self.reporter = reporter
        self.prompt_tokens = int(prompt_tokens)
        self.sample_id = str(sample_id)
        self.budget = int(budget)
        self.batch_size = int(batch_size)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        generated_tokens = max(0, int(input_ids.shape[-1]) - self.prompt_tokens)
        self.reporter.maybe_tokens(
            generated_tokens,
            sample_id=self.sample_id,
            prompt_tokens=self.prompt_tokens,
            generation_budget=self.budget,
            batch_size=self.batch_size,
        )
        return scores


def _apply_chat(tokenizer: Any, messages: List[Dict[str, str]], *, generation: bool, enable_thinking: bool = False):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": generation,
        "return_tensors": "pt",
        "return_dict": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _as_ids(value: Any) -> torch.Tensor:
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "input_ids"):
        value = value.input_ids
    return torch.as_tensor(value, dtype=torch.long)


def _truncate_prompt(tokenizer: Any, messages: List[Dict[str, str]], budget: int) -> List[Dict[str, str]]:
    """Giữ phần đầu document khi prompt vượt budget dành cho target response."""
    if int(_as_ids(_apply_chat(tokenizer, messages, generation=True)).shape[-1]) <= budget:
        return messages
    index = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"), None)
    if index is None:
        return messages
    content = str(messages[index].get("content", ""))
    content_ids = tokenizer(content, add_special_tokens=False)["input_ids"]
    lo, hi = 0, len(content_ids)
    best = content_ids[:0]
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = list(messages)
        candidate[index] = {**messages[index], "content": tokenizer.decode(content_ids[:mid], skip_special_tokens=True)}
        length = int(_as_ids(_apply_chat(tokenizer, candidate, generation=True)).shape[-1])
        if length <= budget:
            best = content_ids[:mid]
            lo = mid + 1
        else:
            hi = mid - 1
    candidate = list(messages)
    candidate[index] = {**messages[index], "content": tokenizer.decode(best, skip_special_tokens=True)}
    return candidate


def _load_responses(path: str) -> Dict[str, str]:
    responses: Dict[str, str] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("id", ""))
        text = row.get("assistant", row.get("response", row.get("text", "")))
        if sample_id and text:
            responses[sample_id] = str(text)
    return responses


def _generate_prepared_batch(
    model: Any,
    tokenizer: Any,
    items: List[Dict[str, Any]],
    *,
    reporter: ProgressReporter,
    device: torch.device,
    args: argparse.Namespace,
) -> List[str]:
    """Generate một nhóm prompt đã validate/tokenize.

    Tất cả item trong nhóm có cùng ``generation_budget``. Left padding giúp
    decoder-only model xử lý batch có prompt dài ngắn khác nhau mà vẫn tách
    đúng phần token mới sinh khỏi prefix.
    """
    if not items:
        return []
    budget = int(items[0]["generation_budget"])
    if any(int(item["generation_budget"]) != budget for item in items):
        raise ValueError("generation batch chứa các sample khác generation budget")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer cần pad_token_id hoặc eos_token_id để batch generation")
    prompt_ids, attention_mask = left_pad_prompt_ids(
        [item["prompt_ids"] for item in items],
        pad_token_id=int(pad_token_id),
    )
    prompt_ids = prompt_ids.to(device)
    attention_mask = attention_mask.to(device)
    padded_prompt_length = int(prompt_ids.shape[-1])
    sample_id = str(items[0]["sample_id"])
    reporter.reset_tokens()
    reporter.update(
        "generating_batch",
        sample_id=sample_id,
        batch_size=len(items),
        batch_sample_ids=[str(item["sample_id"]) for item in items],
        prompt_tokens=max(int(item["prompt_tokens"]) for item in items),
        padded_prompt_tokens=padded_prompt_length,
        generation_budget=budget,
        generated_tokens=0,
        completed_samples=int(items[0].get("completed_samples", 0)),
    )
    kwargs: Dict[str, Any] = {
        "max_new_tokens": budget,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": int(pad_token_id),
    }
    if args.temperature > 0:
        kwargs.update({"do_sample": True, "temperature": args.temperature})
    if args.progress_path:
        from transformers import LogitsProcessorList

        kwargs["logits_processor"] = LogitsProcessorList(
            [
                _ProgressLogitsProcessor(
                    reporter,
                    prompt_tokens=padded_prompt_length,
                    sample_id=sample_id,
                    budget=budget,
                    batch_size=len(items),
                )
            ]
        )
    _reset_peak_vram(device)
    with torch.inference_mode():
        generated_ids = model.generate(
            prompt_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
    if generated_ids.ndim != 2 or int(generated_ids.shape[0]) != len(items):
        raise RuntimeError(
            "target generate trả shape không khớp batch: "
            f"shape={tuple(generated_ids.shape)} batch={len(items)}"
        )
    responses: List[str] = []
    for row, item in enumerate(items):
        if int(generated_ids.shape[1]) >= padded_prompt_length:
            new_ids = generated_ids[row, padded_prompt_length:]
        else:  # pragma: no cover - guard cho model backend bất thường
            new_ids = generated_ids[row]
        response = tokenizer.decode(new_ids.tolist(), skip_special_tokens=True).strip()
        responses.append(response)
    reporter.update(
        "generation_batch_done",
        sample_id=sample_id,
        batch_size=len(items),
        batch_sample_ids=[str(item["sample_id"]) for item in items],
        prompt_tokens=max(int(item["prompt_tokens"]) for item in items),
        padded_prompt_tokens=padded_prompt_length,
        generation_budget=budget,
        generated_tokens=max(0, int(generated_ids.shape[1]) - padded_prompt_length),
    )
    # Namespace is used instead of changing the public return type; external
    # callers/tests historically expect this helper to return ``List[str]``.
    args._last_peak_vram_gb = _peak_vram_gb(device)
    return responses


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate target trajectories")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--target-model-path", default=None)
    parser.add_argument("--target-revision", default=None)
    parser.add_argument("--responses-jsonl", default=None)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--preserve-full-input",
        action="store_true",
        help=(
            "không truncate prompt; tự giảm response budget còn phần context "
            "còn lại, hoặc ghi sample vào skipped report nếu prompt tự nó vượt giới hạn"
        ),
    )
    parser.add_argument(
        "--overflow-policy",
        choices=["error", "skip"],
        default="error",
        help="khi prompt tự nó vượt context: dừng (error) hoặc ghi report và tiếp tục (skip)",
    )
    parser.add_argument(
        "--sample-error-policy",
        choices=["error", "skip"],
        default="error",
        help="lỗi dữ liệu/generate từng sample: dừng (error) hoặc ghi report và tiếp tục (skip)",
    )
    parser.add_argument(
        "--skipped-report",
        default=None,
        help="JSONL ghi id/reason của sample không thể xử lý; mặc định cạnh output",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="giữ output hiện có và bỏ qua sample id đã regenerate",
    )
    parser.add_argument(
        "--progress-path",
        default=None,
        help="JSON heartbeat per-worker; bỏ trống nếu chạy standalone",
    )
    parser.add_argument(
        "--progress-interval-tokens",
        type=int,
        default=256,
        help="số token giữa hai heartbeat trong model.generate",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=1,
        help="batch inference thật trong model.generate; 1 giữ behavior legacy",
    )
    parser.add_argument(
        "--auto-batch",
        action="store_true",
        help="tự tăng generation batch theo peak VRAM và backoff khi OOM",
    )
    parser.add_argument(
        "--auto-batch-target-vram-gb",
        type=float,
        default=None,
        help="mục tiêu peak VRAM mỗi GPU; ví dụ 170 trên B200 180GB",
    )
    parser.add_argument(
        "--auto-batch-max-size",
        type=int,
        default=128,
        help="batch tối đa cho auto-batch trên mỗi bucket",
    )
    parser.add_argument(
        "--auto-batch-growth-factor",
        type=float,
        default=2.0,
        help="hệ số tăng batch sau mỗi lần chạy thành công",
    )
    parser.add_argument(
        "--output-batch-size",
        type=int,
        default=1,
        help="số sample gom trước khi flush output; 1 giúp quan sát/resume an toàn",
    )
    args = parser.parse_args(argv)
    if bool(args.target_model_path) == bool(args.responses_jsonl):
        raise ValueError("chọn đúng một trong --target-model-path hoặc --responses-jsonl")
    if args.max_length < 1 or args.max_new_tokens < 1:
        raise ValueError("max-length và max-new-tokens phải >= 1")
    if args.max_new_tokens > args.max_length:
        raise ValueError("max-new-tokens không được lớn hơn max-length")
    if args.num_shards < 1 or args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("shard-index phải nằm trong [0, num-shards)")
    if args.generation_batch_size < 1:
        raise ValueError("generation-batch-size phải >= 1")
    if args.auto_batch_max_size < args.generation_batch_size:
        raise ValueError("auto-batch-max-size phải >= generation-batch-size")
    if args.auto_batch_growth_factor <= 1.0:
        raise ValueError("auto-batch-growth-factor phải > 1")
    if args.auto_batch_target_vram_gb is not None and args.auto_batch_target_vram_gb <= 0:
        raise ValueError("auto-batch-target-vram-gb phải > 0")
    if args.progress_interval_tokens < 1 or args.output_batch_size < 1:
        raise ValueError("progress-interval-tokens và output-batch-size phải >= 1")
    reporter = ProgressReporter(args.progress_path, interval_tokens=args.progress_interval_tokens)
    previous_hook = install_exception_hook(reporter)
    shard_total_samples = _count_shard_rows(
        args.input,
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
        limit=args.limit,
    )
    reporter.set_context(total_samples=shard_total_samples)
    reporter.update(
        "starting",
        input=str(args.input),
        output=str(args.output),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
        completed_samples=0,
    )
    torch.manual_seed(args.seed)
    responses = _load_responses(args.responses_jsonl) if args.responses_jsonl else {}
    tokenizer = model = None
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.target_model_path:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs = {
            "cache_dir": args.cache_dir,
            "local_files_only": args.local_files_only,
        }
        if args.target_revision:
            load_kwargs["revision"] = args.target_revision
        reporter.update("loading_model", target_model_path=str(args.target_model_path))
        tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, **load_kwargs)
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.torch_dtype]
        model = AutoModelForCausalLM.from_pretrained(
            args.target_model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            **load_kwargs,
        ).to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        reporter.update(
            "model_ready",
            target_model_path=str(args.target_model_path),
            device=str(device),
            dtype=args.torch_dtype,
        )
    else:
        reporter.update("responses_ready", response_count=len(responses))

    output = Path(args.output)
    skipped_report = Path(args.skipped_report) if args.skipped_report else output.with_name(output.stem + ".skipped.jsonl")
    if skipped_report == output:
        raise ValueError("skipped-report phải khác output")
    if (output.exists() or skipped_report.exists()) and not args.resume:
        raise FileExistsError(
            f"artifact regenerate đã tồn tại ({output} hoặc {skipped_report}); "
            "dùng --resume hoặc thư mục mới"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_status_ids(output) if args.resume else set()
    existing.update(_load_status_ids(skipped_report) if args.resume else set())
    generated: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "input_rows": 0,
        "written": 0,
        "skipped_existing": 0,
        "skipped_invalid": 0,
        "skipped_overflow": 0,
        "skipped_errors": 0,
        "clipped_outputs": 0,
    }

    def record_skip(
        sample_id: str,
        *,
        kind: str,
        error: str,
        prompt_tokens: Optional[int] = None,
    ) -> None:
        allowed = (
            args.overflow_policy == "skip"
            if kind == "overflow"
            else args.sample_error_policy == "skip"
        )
        if not allowed:
            raise ValueError(error)
        row = {
            "id": sample_id,
            "kind": kind,
            "error": error,
            "prompt_tokens": prompt_tokens,
            "max_length": int(args.max_length),
            "requested_max_new_tokens": int(args.max_new_tokens),
        }
        _append_jsonl_durable(skipped_report, [row])
        existing.add(sample_id)
        if kind == "overflow":
            stats["skipped_overflow"] += 1
        elif kind == "invalid":
            stats["skipped_invalid"] += 1
        else:
            stats["skipped_errors"] += 1

    generation_pending: List[Dict[str, Any]] = []
    generation_controller = (
        AdaptiveBatchController(
            initial_batch_size=(1 if args.temperature > 0 else int(args.generation_batch_size)),
            max_batch_size=(1 if args.temperature > 0 else int(args.auto_batch_max_size)),
            growth_factor=float(args.auto_batch_growth_factor),
            target_vram_gb=args.auto_batch_target_vram_gb,
        )
        if args.auto_batch
        else None
    )

    def emit_result(
        item: Dict[str, Any],
        assistant: str,
        generated_token_count: Optional[int],
    ) -> None:
        """Đóng gói một response và flush theo output-batch-size."""
        sample_id = str(item["sample_id"])
        prompt_messages = item["prompt_messages"]
        generation_budget = int(item["generation_budget"])
        prompt_ids_len = item.get("prompt_tokens")
        budget_clipped = bool(item.get("budget_clipped", False))
        row = item["row"]
        if not assistant:
            record_skip(
                sample_id,
                kind="invalid",
                error="response rỗng",
                prompt_tokens=prompt_ids_len,
            )
            return
        final_row = {
            **row,
            "conversations": prompt_messages + [{"role": "assistant", "content": assistant}],
            "metadata": {
                **(row.get("metadata") or {}),
                "generation_model": args.target_model_path or "external_response_server",
                "generation_temperature": args.temperature,
                "enable_thinking": bool(args.enable_thinking),
                "max_new_tokens": args.max_new_tokens,
                "actual_max_new_tokens": generation_budget,
                "generation_budget_clipped": budget_clipped,
                "prompt_tokens": prompt_ids_len,
            },
        }
        generated.append(final_row)
        existing.add(sample_id)
        stats["written"] += 1
        if budget_clipped:
            stats["clipped_outputs"] += 1
        reporter.update(
            "writing_output",
            row_index=int(item["row_index"]),
            sample_id=sample_id,
            generated_tokens=generated_token_count,
            pending_output_rows=len(generated),
            completed_samples=len(existing),
        )
        if len(generated) >= args.output_batch_size:
            _append_jsonl_durable(output, generated)
            written_now = len(generated)
            generated.clear()
            reporter.update(
                "sample_done",
                row_index=int(item["row_index"]),
                sample_id=sample_id,
                completed_samples=len(existing),
                written=int(stats["written"]),
                flushed_rows=written_now,
            )
            print(
                f"[regenerate_pilot] written={stats['written']} last_id={sample_id}",
                flush=True,
            )

    def flush_generation_pending(*, force: bool = False) -> None:
        """Generate các item đang chờ, cùng budget và có padding chung."""
        while generation_pending:
            # Chọn bucket đầu tiên đủ một batch hiện tại. Nếu chưa đủ và đây
            # không phải flush cuối, giữ lại để look-ahead có cơ hội gom thêm.
            selected_key: Optional[str] = None
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for pending_item in generation_pending:
                grouped.setdefault(_generation_batch_key(pending_item), []).append(pending_item)
            for candidate_key, candidate_items in grouped.items():
                candidate_limit = (
                    1
                    if args.temperature > 0
                    else int(args.generation_batch_size)
                )
                if generation_controller is not None:
                    candidate_limit = generation_controller.batch_size(
                        candidate_key,
                        default=int(args.generation_batch_size),
                    )
                if force or len(candidate_items) >= candidate_limit:
                    selected_key = candidate_key
                    break
            if selected_key is None:
                return
            effective_batch_size = (
                1
                if args.temperature > 0
                else int(args.generation_batch_size)
            )
            if generation_controller is not None:
                effective_batch_size = generation_controller.batch_size(
                    selected_key,
                    default=int(args.generation_batch_size),
                )
            candidates = grouped[selected_key]
            group = list(
                select_generation_group(candidates, max_batch_size=effective_batch_size)
            )
            if not group:
                raise RuntimeError("không chọn được generation batch từ pending items")
            # Chọn candidate theo độ dài để giảm padding, nhưng trả kết quả
            # theo thứ tự input để standalone regenerate vẫn giữ JSONL order.
            group.sort(key=lambda item: int(item["row_index"]))
            attempted_batch_size = len(group)
            try:
                responses_batch = _generate_prepared_batch(
                    model,
                    tokenizer,
                    group,
                    reporter=reporter,
                    device=device,
                    args=args,
                )
            except Exception as exc:
                if _is_cuda_oom(exc):
                    if generation_controller is None or attempted_batch_size <= 1:
                        raise
                    next_batch_size = generation_controller.record_oom(
                        selected_key,
                        attempted_batch_size=attempted_batch_size,
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    reporter.update(
                        "auto_batch_oom",
                        batch_size=attempted_batch_size,
                        next_batch_size=next_batch_size,
                        batch_key=selected_key,
                    )
                    # Không xóa group: vòng lặp sau retry chính các sample đó.
                    continue
                for item in group:
                    record_skip(
                        str(item["sample_id"]),
                        kind="error",
                        error=f"target generation lỗi: {exc!r}",
                        prompt_tokens=int(item["prompt_tokens"]),
                    )
                responses_batch = [""] * len(group)
            if generation_controller is not None and attempted_batch_size >= effective_batch_size:
                generation_controller.record_success(
                    selected_key,
                    peak_vram_gb=getattr(args, "_last_peak_vram_gb", None),
                )
            group_ids = {id(item) for item in group}
            generation_pending[:] = [
                item for item in generation_pending if id(item) not in group_ids
            ]
            for item, assistant in zip(group, responses_batch):
                if assistant:
                    generated_token_count = None
                    emit_result(item, assistant, generated_token_count)

    for index, row in enumerate(read_jsonl(args.input)):
        if index % args.num_shards != args.shard_index:
            continue
        stats["input_rows"] += 1
        if args.limit is not None and stats["written"] >= args.limit:
            break
        sample_id = str(row.get("id", ""))
        reporter.update(
            "reading_sample",
            row_index=int(index),
            sample_id=sample_id or f"row_{index}",
            completed_samples=len(existing),
            written=int(stats["written"]),
        )
        if not sample_id:
            record_skip(f"row_{index}", kind="invalid", error="sample thiếu id")
            continue
        if sample_id in existing:
            stats["skipped_existing"] += 1
            continue
        messages = list(row.get("conversations") or [])
        if not messages:
            record_skip(
                sample_id,
                kind="invalid",
                error="sample không có conversations",
            )
            continue
        if any(
            str(message.get("role", "")).lower() == "assistant"
            for message in messages
            if isinstance(message, dict)
        ):
            record_skip(
                sample_id,
                kind="invalid",
                error=(
                    f"sample {sample_id!r} đã chứa assistant response; "
                    "regenerate chỉ nhận prompt-only input"
                ),
            )
            continue
        prompt_messages = messages
        assistant = responses.get(sample_id)
        prompt_ids_len: Optional[int] = None
        generation_budget = int(args.max_new_tokens)
        budget_clipped = False
        if tokenizer is not None:
            try:
                if not args.preserve_full_input:
                    budget = max(1, args.max_length - args.max_new_tokens)
                    prompt_messages = _truncate_prompt(tokenizer, messages, budget)
                prompt_ids = _as_ids(
                    _apply_chat(tokenizer, prompt_messages, generation=True)
                ).to(device)
                if prompt_ids.ndim == 1:
                    prompt_ids = prompt_ids.unsqueeze(0)
                prompt_ids_len = int(prompt_ids.shape[-1])
            except Exception as exc:
                if _is_cuda_oom(exc):
                    raise
                record_skip(sample_id, kind="error", error=f"tokenize prompt lỗi: {exc!r}")
                continue
            try:
                generation_budget, budget_clipped = resolve_generation_budget(
                    prompt_ids_len,
                    args.max_length,
                    args.max_new_tokens,
                )
            except ValueError as exc:
                record_skip(
                    sample_id,
                    kind="overflow",
                    error=f"sample {sample_id!r}: {exc}",
                    prompt_tokens=prompt_ids_len,
                )
                continue
            item = {
                "row": row,
                "row_index": int(index),
                "sample_id": sample_id,
                "prompt_messages": prompt_messages,
                "prompt_ids": prompt_ids.flatten(),
                "prompt_tokens": int(prompt_ids_len),
                "generation_budget": int(generation_budget),
                "budget_clipped": bool(budget_clipped),
                "completed_samples": len(existing),
            }
            if assistant is None:
                generation_pending.append(item)
                # Look-ahead buffer amortizes model.generate while remaining
                # bounded in CPU/GPU memory. The group itself is still capped
                # by generation_batch_size.
                pending_limit = max(
                    1,
                    (
                        int(args.auto_batch_max_size) * 4
                        if generation_controller is not None
                        else int(args.generation_batch_size) * 4
                    ),
                )
                if len(generation_pending) >= pending_limit:
                    flush_generation_pending()
                continue
            emit_result(item, assistant, None)
            continue
        if not assistant:
            record_skip(sample_id, kind="invalid", error="response rỗng")
            continue
        emit_result(
            {
                "row": row,
                "row_index": int(index),
                "sample_id": sample_id,
                "prompt_messages": prompt_messages,
                "generation_budget": generation_budget,
                "budget_clipped": budget_clipped,
                "prompt_tokens": prompt_ids_len,
            },
            assistant,
            None,
        )

    flush_generation_pending(force=True)
    if generated:
        pending = len(generated)
        _append_jsonl_durable(output, generated)
        reporter.update(
            "sample_done",
            completed_samples=len(existing),
            written=int(stats["written"]),
            flushed_rows=pending,
        )
    _ensure_durable_empty_file(output)
    _ensure_durable_empty_file(skipped_report)
    stats["covered_rows"] = len(existing)
    stats["uncovered_rows"] = max(0, int(stats["input_rows"]) - int(stats["covered_rows"]))
    manifest_path = Path(args.manifest) if args.manifest else output.with_name(output.stem + "_manifest.json")
    write_json(
        manifest_path,
        {
            "schema_version": "mr_dflash_regeneration_v1",
            "target_model": args.target_model_path or "external_response_server",
            "target_revision": args.target_revision,
            "temperature": args.temperature,
            "enable_thinking": bool(args.enable_thinking),
            "max_new_tokens": args.max_new_tokens,
            "max_length": args.max_length,
            "seed": args.seed,
            "preserve_full_input": bool(args.preserve_full_input),
            "overflow_policy": args.overflow_policy,
            "sample_error_policy": args.sample_error_policy,
            "generation_batch_size": int(args.generation_batch_size),
            "auto_batch": bool(args.auto_batch),
            "auto_batch_target_vram_gb": args.auto_batch_target_vram_gb,
            "auto_batch_max_size": int(args.auto_batch_max_size),
            "auto_batch_growth_factor": float(args.auto_batch_growth_factor),
            "auto_batch_profile": (
                generation_controller.snapshot()
                if generation_controller is not None
                else {}
            ),
            "output_batch_size": int(args.output_batch_size),
            "skipped_report": str(skipped_report),
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
            "stats": stats,
        },
    )
    print(f"[regenerate_pilot] {stats}")
    reporter.update(
        "done",
        completed_samples=len(existing),
        written=int(stats["written"]),
        skipped=int(stats["skipped_invalid"] + stats["skipped_overflow"] + stats["skipped_errors"]),
        input_rows=int(stats["input_rows"]),
    )
    sys.excepthook = previous_hook


if __name__ == "__main__":
    main()
