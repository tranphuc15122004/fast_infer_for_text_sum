"""Regenerate assistant trajectories through a vLLM OpenAI-compatible server.

The tokenizer remains local so the prompt/chat-template contract is identical
to the HF path.  Generation is delegated to vLLM using token-id prompts,
which lets one server use continuous batching across requests without making
the pipeline load a second copy of the target model.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from _common import read_jsonl, write_json
from progress import ProgressReporter, install_exception_hook
from regenerate_pilot import (
    _append_jsonl_durable,
    _as_ids,
    _apply_chat,
    _count_shard_rows,
    _ensure_durable_empty_file,
    _fit_assistant_response,
    _is_cuda_oom,
    _load_sample_ids,
    _load_status_ids,
    _truncate_prompt,
    resolve_generation_budget,
)


class VLLMRequestError(RuntimeError):
    """HTTP/protocol error returned by a vLLM OpenAI-compatible endpoint."""

    def __init__(self, status: int | None, url: str, body: str) -> None:
        self.status = status
        self.url = url
        self.body = body
        status_text = "network" if status is None else str(status)
        super().__init__(f"vLLM request failed ({status_text}) at {url}: {body[:500]}")

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status == 408 or self.status == 429 or self.status >= 500


class VLLMCompletionClient:
    """Small dependency-free client compatible with vLLM 0.24.0."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        timeout_seconds: float = 300.0,
    ) -> None:
        base_url = str(base_url).rstrip("/")
        if not base_url:
            raise ValueError("server-address không được rỗng")
        if timeout_seconds <= 0:
            raise ValueError("request-timeout-seconds phải > 0")
        if base_url.endswith("/v1/completions"):
            self.url = base_url
        elif base_url.endswith("/v1"):
            self.url = f"{base_url}/completions"
        else:
            self.url = f"{base_url}/v1/completions"
        self.model = str(model)
        self.timeout_seconds = float(timeout_seconds)

    def complete(
        self,
        prompt_ids: Iterable[int],
        *,
        max_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": [int(value) for value in prompt_ids],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = int(seed)
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise VLLMRequestError(int(exc.code), self.url, body) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise VLLMRequestError(None, self.url, str(exc)) from exc
        if status < 200 or status >= 300:
            raise VLLMRequestError(status, self.url, body)
        try:
            payload_out = json.loads(body)
            text = payload_out["choices"][0]["text"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise VLLMRequestError(status, self.url, f"invalid completion payload: {body[:500]}") from exc
        return str(text).strip()


class TokenAdmissionController:
    """Adaptive client-side admission for vLLM continuous batching.

    vLLM performs the actual scheduling.  This controller prevents the client
    from flooding it with too many long requests and grows concurrency only
    after successful groups.  ``max_batched_tokens`` is a conservative
    prompt-plus-generation budget, not a replacement for server-side limits.
    """

    def __init__(
        self,
        *,
        initial_size: int,
        max_size: int,
        max_batched_tokens: int,
        growth_factor: float = 2.0,
    ) -> None:
        if initial_size < 1 or max_size < initial_size:
            raise ValueError("request concurrency không hợp lệ")
        if max_batched_tokens < 1:
            raise ValueError("max-batched-tokens phải >= 1")
        if growth_factor <= 1.0:
            raise ValueError("growth-factor phải > 1")
        self.current_size = int(initial_size)
        self.max_size = int(max_size)
        self.max_batched_tokens = int(max_batched_tokens)
        self.growth_factor = float(growth_factor)

    @staticmethod
    def _cost(item: Dict[str, Any]) -> int:
        return max(1, int(item.get("prompt_tokens", 0))) + max(
            1, int(item.get("generation_budget", 0))
        )

    def select(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        total_tokens = 0
        # Admit the shortest requests first so one long tail request does not
        # strand the rest of a token budget. Results are emitted in input order
        # after the futures complete.
        for item in sorted(items, key=self._cost):
            if len(selected) >= self.current_size:
                break
            cost = self._cost(item)
            if selected and total_tokens + cost > self.max_batched_tokens:
                break
            selected.append(item)
            total_tokens += cost
        # One request is always admitted, even when it is longer than the
        # client budget; the server/model context is the final authority.
        return selected or items[:1]

    def record_success(self, admitted: int) -> None:
        if int(admitted) > 0:
            grown = max(self.current_size + 1, int(self.current_size * self.growth_factor))
            self.current_size = min(self.max_size, grown)

    def record_failure(self) -> None:
        self.current_size = max(1, self.current_size // 2)


def _retryable_complete(
    client: VLLMCompletionClient,
    item: Dict[str, Any],
    *,
    seed: int,
    max_retries: int,
) -> str:
    attempts = 0
    while True:
        try:
            return client.complete(
                item["prompt_ids"],
                max_tokens=int(item["generation_budget"]),
                temperature=float(item["temperature"]),
                seed=seed,
            )
        except VLLMRequestError as exc:
            if not exc.retryable or attempts >= max_retries:
                raise
            time.sleep(min(8.0, 0.5 * (2**attempts)))
            attempts += 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate trajectories through vLLM")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--target-model-path", required=True, help="local tokenizer snapshot")
    parser.add_argument("--vllm-model", default=None, help="served model id; defaults to target-model-path")
    parser.add_argument("--server-address", required=True, help="vLLM base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--target-revision", default=None)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--preserve-full-input", action="store_true")
    parser.add_argument("--overflow-policy", choices=["error", "skip"], default="error")
    parser.add_argument("--sample-error-policy", choices=["error", "skip"], default="error")
    parser.add_argument("--skipped-report", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample-ids-file", default=None)
    parser.add_argument("--progress-path", default=None)
    parser.add_argument("--progress-interval-tokens", type=int, default=256)
    parser.add_argument("--request-concurrency", type=int, default=64, help="hard cap concurrent requests per server")
    parser.add_argument("--request-concurrency-start", type=int, default=8, help="initial concurrent requests")
    parser.add_argument("--max-batched-tokens", type=int, default=262144)
    parser.add_argument("--request-growth-factor", type=float, default=2.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--output-batch-size", type=int, default=16)
    return parser


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    if args.max_length < 1 or args.max_new_tokens < 1 or args.max_new_tokens > args.max_length:
        raise ValueError("max-length/max-new-tokens không hợp lệ")
    if args.num_shards < 1 or args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("shard-index phải nằm trong [0, num-shards)")
    if args.request_concurrency < 1 or args.request_concurrency_start < 1:
        raise ValueError("request concurrency phải >= 1")
    if args.request_concurrency_start > args.request_concurrency:
        raise ValueError("request-concurrency-start phải <= request-concurrency")
    if args.max_batched_tokens < 1 or args.request_retries < 0 or args.output_batch_size < 1:
        raise ValueError("request/batch options không hợp lệ")
    if args.temperature < 0:
        raise ValueError("temperature không được âm")

    reporter = ProgressReporter(args.progress_path, interval_tokens=args.progress_interval_tokens)
    previous_hook = install_exception_hook(reporter)
    total_samples = _count_shard_rows(
        args.input,
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
        limit=args.limit,
    )
    reporter.set_context(total_samples=total_samples)
    reporter.update("starting", input=str(args.input), output=str(args.output), server_address=args.server_address)

    from transformers import AutoTokenizer

    load_kwargs = {"cache_dir": args.cache_dir, "local_files_only": args.local_files_only}
    if args.target_revision:
        load_kwargs["revision"] = args.target_revision
    reporter.update("loading_tokenizer", target_model_path=str(args.target_model_path))
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, **load_kwargs)
    reporter.update("vllm_ready", model=args.vllm_model or args.target_model_path, server_address=args.server_address)
    client = VLLMCompletionClient(
        args.server_address,
        model=args.vllm_model or args.target_model_path,
        timeout_seconds=args.request_timeout_seconds,
    )
    effective_request_concurrency = 1 if args.temperature > 0 else args.request_concurrency
    effective_request_start = 1 if args.temperature > 0 else args.request_concurrency_start
    controller = TokenAdmissionController(
        initial_size=effective_request_start,
        max_size=effective_request_concurrency,
        max_batched_tokens=args.max_batched_tokens,
        growth_factor=args.request_growth_factor,
    )

    output = Path(args.output)
    skipped_report = Path(args.skipped_report) if args.skipped_report else output.with_name(output.stem + ".skipped.jsonl")
    if skipped_report == output:
        raise ValueError("skipped-report phải khác output")
    if (output.exists() or skipped_report.exists()) and not args.resume:
        raise FileExistsError(f"artifact regenerate đã tồn tại ({output} hoặc {skipped_report}); dùng --resume")
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_status_ids(output) if args.resume else set()
    if args.resume:
        existing.update(_load_status_ids(skipped_report))
    generated: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    request_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.request_concurrency)
    )
    stats: Dict[str, Any] = {
        "input_rows": 0,
        "written": 0,
        "skipped_existing": 0,
        "skipped_invalid": 0,
        "skipped_overflow": 0,
        "skipped_errors": 0,
        "clipped_outputs": 0,
        "request_errors": 0,
    }

    def record_skip(sample_id: str, *, kind: str, error: str, prompt_tokens: Optional[int] = None) -> None:
        allowed = args.overflow_policy == "skip" if kind == "overflow" else args.sample_error_policy == "skip"
        if not allowed:
            raise ValueError(error)
        _append_jsonl_durable(
            skipped_report,
            [{
                "id": sample_id,
                "kind": kind,
                "error": error,
                "prompt_tokens": prompt_tokens,
                "max_length": int(args.max_length),
                "requested_max_new_tokens": int(args.max_new_tokens),
            }],
        )
        existing.add(sample_id)
        stats[{"overflow": "skipped_overflow", "invalid": "skipped_invalid"}.get(kind, "skipped_errors")] += 1

    def emit_result(item: Dict[str, Any], assistant: str) -> None:
        assistant, response_clipped = _fit_assistant_response(
            tokenizer,
            item["prompt_messages"],
            assistant,
            args.max_length,
        )
        if not assistant:
            record_skip(str(item["sample_id"]), kind="invalid", error="response rỗng", prompt_tokens=item.get("prompt_tokens"))
            return
        row = item["row"]
        final_row = {
            **row,
            "conversations": item["prompt_messages"] + [{"role": "assistant", "content": assistant}],
            "metadata": {
                **(row.get("metadata") or {}),
                "generation_model": args.vllm_model or args.target_model_path,
                "generation_backend": "vllm",
                "generation_server": args.server_address,
                "generation_temperature": args.temperature,
                "enable_thinking": bool(args.enable_thinking),
                "max_new_tokens": args.max_new_tokens,
                "actual_max_new_tokens": item["generation_budget"],
                "generation_budget_clipped": bool(item["budget_clipped"] or response_clipped),
                "response_template_clipped": bool(response_clipped),
                "prompt_tokens": item.get("prompt_tokens"),
            },
        }
        generated.append(final_row)
        existing.add(str(item["sample_id"]))
        stats["written"] += 1
        stats["clipped_outputs"] += int(item["budget_clipped"] or response_clipped)
        if len(generated) >= args.output_batch_size:
            flushed = len(generated)
            _append_jsonl_durable(output, generated)
            generated.clear()
            reporter.update("sample_done", sample_id=str(item["sample_id"]), completed_samples=len(existing), written=stats["written"], flushed_rows=flushed)

    def flush_pending(*, force: bool = False) -> None:
        while pending:
            group = controller.select(pending)
            if not force and len(group) < controller.current_size and len(pending) < controller.current_size:
                return
            group_ids = {id(item) for item in group}
            reporter.update(
                "generating_batch",
                batch_size=len(group),
                batch_sample_ids=[str(item["sample_id"]) for item in group],
                prompt_tokens=max(int(item["prompt_tokens"]) for item in group),
                generation_budget=max(int(item["generation_budget"]) for item in group),
            )
            responses: Dict[int, str] = {}
            future_map = {
                request_executor.submit(
                    _retryable_complete,
                    client,
                    item,
                    seed=int(args.seed) + int(item["row_index"]),
                    max_retries=int(args.request_retries),
                ): item
                for item in group
            }
            request_error: Exception | None = None
            for future in concurrent.futures.as_completed(future_map):
                item = future_map[future]
                try:
                    responses[id(item)] = future.result()
                except Exception as exc:
                    request_error = request_error or exc
            if request_error is not None:
                exc = request_error
                stats["request_errors"] += len(group)
                if isinstance(exc, VLLMRequestError) and exc.retryable and len(group) > 1:
                    controller.record_failure()
                    reporter.update("vllm_backoff", error=repr(exc), next_concurrency=controller.current_size)
                    continue
                for item in group:
                    record_skip(str(item["sample_id"]), kind="error", error=f"vLLM generation lỗi: {exc!r}", prompt_tokens=item["prompt_tokens"])
                    pending.remove(item)
                continue
            controller.record_success(len(group))
            pending[:] = [item for item in pending if id(item) not in group_ids]
            for item in sorted(group, key=lambda value: int(value["row_index"])):
                emit_result(item, responses[id(item)])
            reporter.update("generation_batch_done", batch_size=len(group), generated_tokens=None)

    selected_sample_ids = _load_sample_ids(args.sample_ids_file)
    for index, row in enumerate(read_jsonl(args.input)):
        if index % args.num_shards != args.shard_index:
            continue
        stats["input_rows"] += 1
        if args.limit is not None and stats["written"] >= args.limit:
            break
        sample_id = str(row.get("id", ""))
        if selected_sample_ids is not None and sample_id not in selected_sample_ids:
            continue
        if not sample_id:
            record_skip(f"row_{index}", kind="invalid", error="sample thiếu id")
            continue
        if sample_id in existing:
            stats["skipped_existing"] += 1
            continue
        messages = list(row.get("conversations") or [])
        if not messages:
            record_skip(sample_id, kind="invalid", error="sample không có conversations")
            continue
        if any(str(message.get("role", "")).lower() == "assistant" for message in messages if isinstance(message, dict)):
            record_skip(sample_id, kind="invalid", error=f"sample {sample_id!r} đã chứa assistant response; regenerate chỉ nhận prompt-only input")
            continue
        prompt_messages = messages
        prompt_tokens: Optional[int] = None
        try:
            if not args.preserve_full_input:
                prompt_messages = _truncate_prompt(tokenizer, messages, max(1, args.max_length - args.max_new_tokens))
            prompt_ids = _as_ids(_apply_chat(tokenizer, prompt_messages, generation=True)).flatten()
            prompt_tokens = int(prompt_ids.numel())
            generation_budget, budget_clipped = resolve_generation_budget(prompt_tokens, args.max_length, args.max_new_tokens)
        except ValueError as exc:
            record_skip(sample_id, kind="overflow", error=f"sample {sample_id!r}: {exc}", prompt_tokens=prompt_tokens)
            continue
        except Exception as exc:
            if _is_cuda_oom(exc):
                raise
            record_skip(sample_id, kind="error", error=f"tokenize prompt lỗi: {exc!r}")
            continue
        pending.append({
            "row": row,
            "row_index": int(index),
            "sample_id": sample_id,
            "prompt_messages": prompt_messages,
            "prompt_ids": prompt_ids.tolist(),
            "prompt_tokens": prompt_tokens,
            "generation_budget": int(generation_budget),
            "budget_clipped": bool(budget_clipped),
            "temperature": float(args.temperature),
        })
        if len(pending) >= controller.current_size * 2:
            flush_pending()
    flush_pending(force=True)
    request_executor.shutdown(wait=True)
    if generated:
        pending_rows = len(generated)
        _append_jsonl_durable(output, generated)
        generated.clear()
        reporter.update("sample_done", completed_samples=len(existing), written=stats["written"], flushed_rows=pending_rows)
    _ensure_durable_empty_file(output)
    _ensure_durable_empty_file(skipped_report)
    stats["covered_rows"] = len(existing)
    stats["uncovered_rows"] = max(0, int(stats["input_rows"]) - int(stats["covered_rows"]))
    manifest_path = Path(args.manifest) if args.manifest else output.with_name(output.stem + "_manifest.json")
    write_json(
        manifest_path,
        {
            "schema_version": "mr_dflash_regeneration_v1",
            "target_model": args.vllm_model or args.target_model_path,
            "target_tokenizer": args.target_model_path,
            "target_revision": args.target_revision,
            "generation_backend": "vllm",
            "server_address": args.server_address,
            "temperature": args.temperature,
            "request_concurrency_requested": args.request_concurrency,
            "enable_thinking": bool(args.enable_thinking),
            "max_new_tokens": args.max_new_tokens,
            "max_length": args.max_length,
            "seed": args.seed,
            "request_concurrency_max": effective_request_concurrency,
            "request_concurrency_final": controller.current_size,
            "max_batched_tokens": args.max_batched_tokens,
            "request_growth_factor": args.request_growth_factor,
            "request_retries": args.request_retries,
            "output_batch_size": args.output_batch_size,
            "skipped_report": str(skipped_report),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "sample_ids_file": args.sample_ids_file,
            "stats": stats,
        },
    )
    print(f"[vllm_regenerate] {stats}")
    reporter.update("done", completed_samples=len(existing), written=stats["written"], skipped=stats["skipped_invalid"] + stats["skipped_overflow"] + stats["skipped_errors"], input_rows=stats["input_rows"])
    sys.excepthook = previous_hook


if __name__ == "__main__":
    main()
