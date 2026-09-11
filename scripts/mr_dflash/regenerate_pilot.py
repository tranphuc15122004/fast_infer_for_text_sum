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
from progress import ProgressReporter, install_exception_hook


def _is_cuda_oom(exc: BaseException) -> bool:
    """OOM không được skip: CUDA context có thể đã ở trạng thái không an toàn."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


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

    def __init__(self, reporter: ProgressReporter, *, prompt_tokens: int, sample_id: str, budget: int) -> None:
        self.reporter = reporter
        self.prompt_tokens = int(prompt_tokens)
        self.sample_id = str(sample_id)
        self.budget = int(budget)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        generated_tokens = max(0, int(input_ids.shape[-1]) - self.prompt_tokens)
        self.reporter.maybe_tokens(
            generated_tokens,
            sample_id=self.sample_id,
            prompt_tokens=self.prompt_tokens,
            generation_budget=self.budget,
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
    if args.progress_interval_tokens < 1 or args.output_batch_size < 1:
        raise ValueError("progress-interval-tokens và output-batch-size phải >= 1")
    reporter = ProgressReporter(args.progress_path, interval_tokens=args.progress_interval_tokens)
    previous_hook = install_exception_hook(reporter)
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
            record_skip(sample_id or f"row_{index}", kind="invalid", error="sample không có conversations")
            continue
        if any(str(message.get("role", "")).lower() == "assistant" for message in messages if isinstance(message, dict)):
            record_skip(
                sample_id or f"row_{index}",
                kind="invalid",
                error=(
                    f"sample {sample_id!r} đã chứa assistant response; "
                    "regenerate chỉ nhận prompt-only input"
                ),
            )
            continue
        prompt_messages = messages
        assistant = responses.get(sample_id)
        prompt_ids_len = None
        generation_budget = int(args.max_new_tokens)
        budget_clipped = False
        generated_token_count: Optional[int] = None
        if tokenizer is not None:
            try:
                if not args.preserve_full_input:
                    budget = max(1, args.max_length - args.max_new_tokens)
                    prompt_messages = _truncate_prompt(tokenizer, messages, budget)
                prompt_ids = _as_ids(_apply_chat(tokenizer, prompt_messages, generation=True)).to(device)
                if prompt_ids.ndim == 1:
                    prompt_ids = prompt_ids.unsqueeze(0)
                prompt_ids_len = int(prompt_ids.shape[-1])
            except Exception as exc:
                if _is_cuda_oom(exc):
                    raise
                record_skip(
                    sample_id,
                    kind="error",
                    error=f"tokenize prompt lỗi: {exc!r}",
                )
                continue
            try:
                generation_budget, budget_clipped = resolve_generation_budget(
                    prompt_ids_len,
                    args.max_length,
                    args.max_new_tokens,
                )
            except ValueError as exc:
                record_skip(sample_id or f"row_{index}", kind="overflow", error=f"sample {sample_id!r}: {exc}", prompt_tokens=prompt_ids_len)
                continue
            if assistant is None:
                try:
                    reporter.reset_tokens()
                    reporter.update(
                        "generating",
                        row_index=int(index),
                        sample_id=sample_id,
                        prompt_tokens=int(prompt_ids_len),
                        generation_budget=int(generation_budget),
                        generated_tokens=0,
                        completed_samples=len(existing),
                    )
                    with torch.inference_mode():
                        kwargs = {"max_new_tokens": generation_budget, "do_sample": False}
                        if args.temperature > 0:
                            kwargs.update({"do_sample": True, "temperature": args.temperature})
                        # LogitsProcessor chỉ đọc input_ids và trả scores
                        # nguyên vẹn; vì vậy heartbeat không làm đổi output.
                        if args.progress_path:
                            from transformers import LogitsProcessorList

                            kwargs["logits_processor"] = LogitsProcessorList(
                                [
                                    _ProgressLogitsProcessor(
                                        reporter,
                                        prompt_tokens=prompt_ids_len,
                                        sample_id=sample_id,
                                        budget=generation_budget,
                                    )
                                ]
                            )
                        generated_ids = model.generate(
                            prompt_ids,
                            attention_mask=torch.ones_like(prompt_ids),
                            **kwargs,
                        )
                except Exception as exc:
                    if _is_cuda_oom(exc):
                        raise
                    record_skip(
                        sample_id or f"row_{index}",
                        kind="error",
                        error=f"target generation lỗi: {exc!r}",
                        prompt_tokens=prompt_ids_len,
                    )
                    continue
                generated_token_count = max(0, int(generated_ids.shape[-1]) - int(prompt_ids_len))
                reporter.update(
                    "generation_done",
                    row_index=int(index),
                    sample_id=sample_id,
                    prompt_tokens=int(prompt_ids_len),
                    generation_budget=int(generation_budget),
                    generated_tokens=generated_token_count,
                )
                assistant = tokenizer.decode(generated_ids[0, prompt_ids_len:], skip_special_tokens=True).strip()
        if not assistant:
            record_skip(sample_id or f"row_{index}", kind="invalid", error="response rỗng", prompt_tokens=prompt_ids_len)
            continue
        final_messages = prompt_messages + [{"role": "assistant", "content": assistant}]
        final_row = {
            **row,
            "conversations": final_messages,
            "metadata": {
                **(row.get("metadata") or {}),
                "generation_model": args.target_model_path or "external_response_server",
                "generation_temperature": args.temperature,
                "enable_thinking": bool(args.enable_thinking),
                "max_new_tokens": args.max_new_tokens,
                "actual_max_new_tokens": generation_budget,
                "generation_budget_clipped": bool(budget_clipped),
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
            row_index=int(index),
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
                row_index=int(index),
                sample_id=sample_id,
                completed_samples=len(existing),
                written=int(stats["written"]),
                flushed_rows=written_now,
            )
            print(f"[regenerate_pilot] written={stats['written']} last_id={sample_id}", flush=True)
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
