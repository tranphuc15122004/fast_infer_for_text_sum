"""Teacher-forced source-only K/V fake-quantization experiments (E44)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fidelity import install_source_cache_quantizer


_CONFIG = re.compile(r"K(4|8|16)V(4|8|16)\Z")


class AttentionOutputCapture:
    """Capture projected attention outputs for decode steps only."""

    def __init__(self, model: Any) -> None:
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            raise ValueError("model must expose model.layers")
        self.latest: dict[int, Any] = {}
        self._handles = []
        for layer_id, layer in enumerate(layers):
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                raise ValueError(f"layer {layer_id} has no self_attn module")
            self._handles.append(
                attention.register_forward_hook(self._make_hook(layer_id))
            )

    def _make_hook(self, layer_id: int):
        def hook(_module: Any, _args: Any, output: Any) -> None:
            import torch

            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(tensor) or tensor.ndim < 2:
                raise ValueError("self-attention output has an unsupported shape")
            if tensor.ndim == 3:
                tensor = tensor[0, -1]
            elif tensor.ndim == 2:
                tensor = tensor[-1]
            else:
                raise ValueError("self-attention output rank must be 2 or 3")
            self.latest[layer_id] = tensor.detach().float().cpu()

        return hook

    def clear(self) -> None:
        self.latest.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "AttentionOutputCapture":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def parse_configs(value: str) -> list[tuple[str, int, int]]:
    """Parse K/V bit-width labels such as ``K16V16,K8V4``."""

    result: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for item in str(value).split(","):
        label = item.strip().upper()
        if not label:
            continue
        match = _CONFIG.fullmatch(label)
        if match is None:
            raise ValueError(f"invalid K/V precision config: {item!r}")
        if label not in seen:
            result.append((label, int(match.group(1)), int(match.group(2))))
            seen.add(label)
    if not result:
        raise ValueError("at least one K/V precision config is required")
    return result


def _read_reference_tokens(path: Path) -> dict[str, list[int]]:
    tokens: dict[str, list[int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok" or not row.get("sample_id"):
                continue
            values = row.get("generated_token_ids")
            if not isinstance(values, list):
                raise ValueError(f"reference row {line_number} has no generated_token_ids")
            tokens[str(row["sample_id"])] = [int(value) for value in values]
    if not tokens:
        raise ValueError("reference trace contains no usable generated-token rows")
    return tokens


def _load_records(paths: Sequence[str], *, max_samples: int) -> list[tuple[dict[str, Any], str]]:
    records: list[tuple[dict[str, Any], str]] = []
    for raw_path in paths:
        path = Path(raw_path)
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected = rows if max_samples <= 0 else rows[:max_samples]
        records.extend((row, path.stem) for row in selected)
    if not records:
        raise ValueError("input files contain no records")
    return records


def filter_records_by_id(
    records: Sequence[tuple[dict[str, Any], str]], sample_ids: Sequence[str] | str
) -> list[tuple[dict[str, Any], str]]:
    if isinstance(sample_ids, str):
        sample_ids = tuple(part.strip() for part in sample_ids.split(","))
    requested = {str(sample_id) for sample_id in sample_ids if str(sample_id)}
    if not requested:
        return list(records)
    selected = [item for item in records if str(item[0].get("id", "")) in requested]
    found = {str(item[0].get("id", "")) for item in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"sample IDs not found in selected inputs: {missing}")
    return selected


def _step_input(input_ids: Any, reference_tokens: Sequence[int], step: int, device: Any) -> Any:
    import torch

    if step == 0:
        return input_ids[:, -1:]
    return torch.tensor([[int(reference_tokens[step - 1])]], dtype=torch.long, device=device)


def _nll(log_probs: Any, token_id: int) -> float:
    value = float(log_probs[int(token_id)].item())
    if not math.isfinite(value):
        raise ValueError("reference-token log probability is not finite")
    return -value


def _run_dense_reference(
    model: Any,
    rendered: Any,
    reference_tokens: Sequence[int],
    *,
    device: Any,
    prefill_chunk_size: int,
) -> tuple[list[Any], list[int], list[dict[int, Any]], float, int]:
    import torch
    from src.TrainingFree.collector import _prefill_with_hidden
    from src.analyze.groundsync.trace_target import _model_call

    input_ids = rendered.input_ids.to(device)
    if input_ids.shape[1] < 2:
        raise ValueError("rendered prompt must have at least two tokens")
    with torch.inference_mode():
        past, _ = _prefill_with_hidden(
            model, input_ids[:, :-1], chunk_size=prefill_chunk_size
        )
        log_probs: list[Any] = []
        predictions: list[int] = []
        output_steps: list[dict[int, Any]] = []
        reference_nll = 0.0
        reference_agreement = 0
        with AttentionOutputCapture(model) as capture:
            for step, expected_token in enumerate(reference_tokens):
                capture.clear()
                outputs = _model_call(
                    model,
                    input_ids=_step_input(input_ids, reference_tokens, step, device),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                current_log_probs = torch.log_softmax(outputs.logits[0, -1].float(), dim=-1)
                prediction = int(current_log_probs.argmax().item())
                log_probs.append(current_log_probs.detach())
                predictions.append(prediction)
                output_steps.append(dict(capture.latest))
                reference_nll += _nll(current_log_probs, int(expected_token))
                reference_agreement += int(prediction == int(expected_token))
                past = outputs.past_key_values
                del outputs
        del past
    return log_probs, predictions, output_steps, reference_nll, reference_agreement


def _run_quantized_candidate(
    model: Any,
    rendered: Any,
    reference_tokens: Sequence[int],
    dense_log_probs: Sequence[Any],
    dense_predictions: Sequence[int],
    dense_attention_outputs: Sequence[Mapping[int, Any]],
    *,
    source_start: int,
    source_end: int,
    k_bits: int,
    v_bits: int,
    device: Any,
    prefill_chunk_size: int,
) -> dict[str, Any]:
    import torch
    from src.TrainingFree.collector import _prefill_with_hidden
    from src.analyze.groundsync.trace_target import _model_call

    input_ids = rendered.input_ids.to(device)
    with torch.inference_mode():
        past, _ = _prefill_with_hidden(
            model, input_ids[:, :-1], chunk_size=prefill_chunk_size
        )
        install_source_cache_quantizer(
            past,
            source_start=source_start,
            source_end=source_end,
            k_bits=k_bits,
            v_bits=v_bits,
        )
        candidate_nll = 0.0
        dense_nll = 0.0
        top1_matches = 0
        kl_values: list[float] = []
        attention_errors: list[float] = []
        with AttentionOutputCapture(model) as capture:
            for step, expected_token in enumerate(reference_tokens):
                capture.clear()
                outputs = _model_call(
                    model,
                    input_ids=_step_input(input_ids, reference_tokens, step, device),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                candidate_log_probs = torch.log_softmax(outputs.logits[0, -1].float(), dim=-1)
                reference_log_probs = dense_log_probs[step]
                reference_probabilities = reference_log_probs.exp()
                kl = torch.sum(
                    reference_probabilities * (reference_log_probs - candidate_log_probs)
                )
                kl_values.append(float(kl.item()))
                top1_matches += int(
                    int(candidate_log_probs.argmax().item()) == dense_predictions[step]
                )
                candidate_nll += _nll(candidate_log_probs, int(expected_token))
                dense_nll += _nll(reference_log_probs, int(expected_token))

                dense_outputs = dense_attention_outputs[step]
                if set(capture.latest) != set(dense_outputs):
                    raise ValueError("attention-output hooks did not capture every layer")
                for layer_id, quantized_output in capture.latest.items():
                    reference_output = dense_outputs[layer_id]
                    denominator = float(torch.linalg.vector_norm(reference_output).item())
                    numerator = float(
                        torch.linalg.vector_norm(quantized_output - reference_output).item()
                    )
                    attention_errors.append(numerator / max(denominator, 1e-12))
                past = outputs.past_key_values
                del outputs
        del past

    token_count = len(reference_tokens)
    return {
        "token_count": token_count,
        "top1_agreement": top1_matches / token_count,
        "delta_nll_percent": 100.0 * (candidate_nll / max(dense_nll, 1e-12) - 1.0),
        "candidate_nll_sum": candidate_nll,
        "dense_nll_sum": dense_nll,
        "top1_matches": top1_matches,
        "kl_mean": sum(kl_values) / len(kl_values),
        "kl_p99": float(torch.quantile(torch.tensor(kl_values), 0.99).item()),
        "attention_output_error_mean": sum(attention_errors) / len(attention_errors),
        "attention_output_error_p99": float(
            torch.quantile(torch.tensor(attention_errors), 0.99).item()
        ),
        "attention_output_error_values": attention_errors,
        "source_kv_bits_per_component": (int(k_bits) + int(v_bits)) / 2.0,
        "source_kv_total_bits_per_element": int(k_bits) + int(v_bits),
        "ideal_source_kv_compression_ratio": 32.0 / (int(k_bits) + int(v_bits)),
    }


def _metric_pass(metrics: Mapping[str, Any]) -> bool:
    return bool(
        metrics.get("top1_agreement", -1.0) >= 0.99
        and metrics.get("delta_nll_percent", math.inf) <= 1.0
        and metrics.get("attention_output_error_mean", math.inf) <= 0.01
        and metrics.get("attention_output_error_p99", math.inf) <= 0.05
    )


def _pool_global_metrics(target: dict[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    import numpy as np

    errors = [error for row in rows for error in row["attention_output_error_values"]]
    target["attention_output_error_mean"] = float(np.mean(errors))
    target["attention_output_error_p99"] = float(np.quantile(errors, 0.99))
    candidate_nll = sum(float(row["candidate_nll_sum"]) for row in rows)
    dense_nll = sum(float(row["dense_nll_sum"]) for row in rows)
    target["delta_nll_percent"] = 100.0 * (candidate_nll / max(dense_nll, 1e-12) - 1.0)


def _summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    configs: Sequence[tuple[str, int, int]],
) -> dict[str, Any]:
    by_config: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        for config, metrics in row.get("candidates", {}).items():
            by_config[config].append((str(row.get("dataset", "unknown")), metrics))

    result: dict[str, Any] = {}
    for name, k_bits, v_bits in configs:
        values = by_config.get(name, [])
        token_count = sum(int(metrics["token_count"]) for _, metrics in values)
        if not token_count:
            result[name] = {"n": 0, "status": "NO_VALID_ROWS", "gate_pass": False}
            continue
        weights = [int(metrics["token_count"]) for _, metrics in values]
        aggregate = {
            metric: sum(float(row[metric]) * weight for (_, row), weight in zip(values, weights)) / token_count
            for metric in (
                "top1_agreement",
                "kl_mean",
                "delta_nll_percent",
                "attention_output_error_mean",
                "attention_output_error_p99",
            )
        }
        aggregate.update(
            {
                "n": len(values),
                "tokens": token_count,
                "source_kv_bits_per_component": (k_bits + v_bits) / 2.0,
                "source_kv_total_bits_per_element": k_bits + v_bits,
                "ideal_source_kv_compression_ratio": 32.0 / (k_bits + v_bits),
                "gate_pass": _metric_pass(aggregate),
            }
        )
        _pool_global_metrics(aggregate, [metrics for _, metrics in values])
        dataset_summaries: dict[str, Any] = {}
        for dataset in sorted({label for label, _ in values}):
            dataset_values = [metrics for label, metrics in values if label == dataset]
            dataset_tokens = sum(int(item["token_count"]) for item in dataset_values)
            dataset_summary = {
                metric: sum(float(item[metric]) * int(item["token_count"]) for item in dataset_values) / dataset_tokens
                for metric in (
                    "top1_agreement",
                    "kl_mean",
                    "delta_nll_percent",
                    "attention_output_error_mean",
                    "attention_output_error_p99",
                )
            }
            _pool_global_metrics(dataset_summary, dataset_values)
            dataset_summary.update(
                {"n": len(dataset_values), "tokens": dataset_tokens, "gate_pass": _metric_pass(dataset_summary)}
            )
            dataset_summaries[dataset] = dataset_summary
        aggregate["datasets"] = dataset_summaries
        result[name] = aggregate
    return result


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def render_report(manifest: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    lines = [
        "# FidelityKV E44 fake-quantization scan",
        "",
        "## Run status",
        "",
        f"- Model: `{manifest.get('model')}`",
        f"- GPU: `{manifest.get('gpu_name')}`",
        f"- Cohort: {manifest.get('ok_samples')}/{manifest.get('requested_samples')} valid samples",
        f"- Dense-reference token agreement with E43: `{manifest.get('e43_reference_token_agreement')}`",
        "- KV mutation in this scan is fake quantization only; no physical memory or latency gain is claimed.",
        "",
        "## Candidate metrics",
        "",
        "| Config | n | tokens | Top-1 | ΔNLL % | KL mean | Attn out err mean | Attn out err P99 | ideal source-KV ratio | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for config, values in summary.items():
        if not values.get("tokens"):
            lines.append(f"| {config} | 0 | 0 | — | — | — | — | — | — | no data |")
            continue
        lines.append(
            f"| {config} | {values['n']} | {values['tokens']} | {values['top1_agreement']:.6f} | "
            f"{values['delta_nll_percent']:.4f} | {values['kl_mean']:.6g} | "
            f"{values['attention_output_error_mean']:.6g} | "
            f"{values['attention_output_error_p99']:.6g} | "
            f"{values['ideal_source_kv_compression_ratio']:.3f}× | "
            f"{'PASS' if values['gate_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "Search gate: Top-1 ≥99%, relative ΔNLL ≤1%, attention-output relative-L2 mean ≤1% and P99 ≤5%.",
            "A pilot is only an early screen; it does not establish a full-cohort result.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    import transformers
    from src.TrainingFree.run import _record_dataset
    from src.analyze.groundsync.trace_target import (
        DEFAULT_INSTRUCTION,
        _record_document,
        load_local_model,
        render_document_prompt,
    )

    started = time.perf_counter()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    records = _load_records(args.input, max_samples=args.max_samples)
    records = filter_records_by_id(records, args.sample_ids)
    reference_tokens = _read_reference_tokens(Path(args.reference_trace))
    configs = parse_configs(args.configs)
    model, tokenizer, device = load_local_model(
        args.model, device=args.device, dtype=args.dtype
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = "cpu"
    rows: list[dict[str, Any]] = []
    e43_token_total = 0
    e43_token_matches = 0
    for index, (record, source_name) in enumerate(records):
        sample_id = str(record.get("id", index))
        dataset = _record_dataset(record, source_name)
        try:
            expected_tokens = reference_tokens.get(sample_id)
            if expected_tokens is None:
                raise ValueError(f"sample {sample_id} is absent from the E43 reference trace")
            expected_tokens = expected_tokens[: args.max_new_tokens]
            if not expected_tokens:
                raise ValueError(f"sample {sample_id} has an empty E43 token sequence")
            rendered = render_document_prompt(tokenizer, _record_document(record))
            baseline_log_probs, baseline_predictions, baseline_attention_outputs, _, ref_matches = _run_dense_reference(
                model,
                rendered,
                expected_tokens,
                device=device,
                prefill_chunk_size=args.prefill_chunk_size,
            )
            e43_token_total += len(expected_tokens)
            e43_token_matches += ref_matches
            candidate_metrics: dict[str, Any] = {}
            for name, k_bits, v_bits in configs:
                candidate_metrics[name] = _run_quantized_candidate(
                    model,
                    rendered,
                    expected_tokens,
                    baseline_log_probs,
                    baseline_predictions,
                    baseline_attention_outputs,
                    source_start=rendered.source_start,
                    source_end=rendered.source_end,
                    k_bits=k_bits,
                    v_bits=v_bits,
                    device=device,
                    prefill_chunk_size=args.prefill_chunk_size,
                )
            rows.append(
                {
                    "status": "ok",
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "input_tokens": int(rendered.input_ids.shape[1]),
                    "source_tokens": int(rendered.source_end - rendered.source_start),
                    "reference_tokens": len(expected_tokens),
                    "e43_top1_agreement": ref_matches / len(expected_tokens),
                    "candidates": candidate_metrics,
                }
            )
            print(
                f"[fidelity] {index + 1}/{len(records)} {dataset}/{sample_id} "
                f"tokens={len(expected_tokens)} E43_top1={ref_matches / len(expected_tokens):.4f}",
                flush=True,
            )
            del baseline_log_probs, baseline_attention_outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            rows.append(
                {
                    "status": "error",
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[fidelity] ERROR {dataset}/{sample_id}: {type(exc).__name__}: {exc}", flush=True)

    candidate_summary = _summarize_rows(rows, configs)
    errors = sum(row.get("status") != "ok" for row in rows)
    manifest = {
        "schema_version": "fidelitykv.e44.v1",
        "status": "ok" if errors == 0 else "partial",
        "run_id": args.run_id,
        "experiment": "E44_source_only_fake_quantization",
        "model": args.model,
        "device": str(device),
        "gpu_name": gpu_name,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "input_files": list(args.input),
        "input_sha256": {
            str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in args.input
        },
        "reference_trace": args.reference_trace,
        "reference_trace_sha256": hashlib.sha256(Path(args.reference_trace).read_bytes()).hexdigest(),
        "reference_instruction_sha256": hashlib.sha256(DEFAULT_INSTRUCTION.encode()).hexdigest(),
        "requested_samples": len(records),
        "ok_samples": len(rows) - errors,
        "error_samples": errors,
        "max_new_tokens": args.max_new_tokens,
        "prefill_chunk_size": args.prefill_chunk_size,
        "seed": args.seed,
        "configs": [
            {"name": name, "k_bits": k_bits, "v_bits": v_bits}
            for name, k_bits, v_bits in configs
        ],
        "fake_quantization": {
            "scheme": "symmetric per-layer, per-KV-head, per-head-dimension min/max over source sequence",
            "storage": "BF16 fake-quantized/dequantized; no packed physical cache",
            "generated_kv": "unquantized model dtype",
            "source_only": True,
        },
        "search_gate": {
            "top1_agreement_min": 0.99,
            "delta_nll_percent_max": 1.0,
            "attention_output_error_mean_max": 0.01,
            "attention_output_error_p99_max": 0.05,
        },
        "e43_reference_token_agreement": (
            e43_token_matches / e43_token_total if e43_token_total else None
        ),
        "candidate_metrics": candidate_summary,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    _write_jsonl(output_dir / "fidelity_rows.jsonl", rows)
    (output_dir / "fidelity_summary.json").write_text(
        json.dumps(candidate_summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_report(manifest, candidate_summary), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0 if errors == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--reference-trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-samples", type=int, default=20, help="maximum records per input file")
    parser.add_argument("--sample-ids", default="", help="comma-separated exact sample IDs to run")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--configs", default="K16V16,K16V8,K16V4,K8V16,K4V16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="bfloat16")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_samples < 0 or args.max_new_tokens <= 0 or args.prefill_chunk_size <= 0:
        raise SystemExit("sample limit must be non-negative and token/chunk sizes positive")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
