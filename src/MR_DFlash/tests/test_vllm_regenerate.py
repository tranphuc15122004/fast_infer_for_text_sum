"""Protocol/controller tests for the vLLM regenerate worker."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "mr_dflash"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_vllm_completion_client_sends_token_ids_and_parses_text(monkeypatch) -> None:
    import vllm_regenerate

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"text": " answer "}]})

    monkeypatch.setattr(vllm_regenerate.urllib.request, "urlopen", fake_urlopen)
    client = vllm_regenerate.VLLMCompletionClient(
        "http://127.0.0.1:8000/v1",
        model="served-model",
        timeout_seconds=17,
    )

    assert client.complete([11, 12, 13], max_tokens=27, temperature=0.0, seed=42) == "answer"
    assert captured == {
        "url": "http://127.0.0.1:8000/v1/completions",
        "timeout": 17.0,
        "payload": {
            "model": "served-model",
            "prompt": [11, 12, 13],
            "max_tokens": 27,
            "temperature": 0.0,
            "seed": 42,
            "stream": False,
        },
    }


def test_token_admission_controller_is_token_aware_and_recovers_after_oom() -> None:
    from vllm_regenerate import TokenAdmissionController

    controller = TokenAdmissionController(
        initial_size=4,
        max_size=16,
        max_batched_tokens=100,
        growth_factor=2.0,
    )
    items = [
        {"prompt_tokens": 20, "generation_budget": 30},
        {"prompt_tokens": 10, "generation_budget": 20},
        {"prompt_tokens": 40, "generation_budget": 40},
    ]

    selected = controller.select(items)
    assert len(selected) == 2
    assert controller.current_size == 4

    controller.record_success(len(selected))
    assert controller.current_size == 8
    controller.record_failure()
    assert controller.current_size == 4


def test_token_admission_controller_never_returns_empty_for_one_oversized_prompt() -> None:
    from vllm_regenerate import TokenAdmissionController

    controller = TokenAdmissionController(
        initial_size=8,
        max_size=8,
        max_batched_tokens=10,
    )
    item = {"prompt_tokens": 100, "generation_budget": 100}
    assert controller.select([item]) == [item]


def test_vllm_metrics_parser_reads_peak_gpu_cache_usage() -> None:
    from vllm_regenerate import parse_vllm_metrics

    payload = """
# HELP vllm:gpu_cache_usage_perc GPU KV cache usage
vllm:gpu_cache_usage_perc{gpu="0"} 0.72
vllm:gpu_cache_usage_perc{gpu="1"} 0.91
vllm:num_requests_running 8
"""

    assert parse_vllm_metrics(payload)["gpu_cache_usage"] == 0.91


def test_token_admission_controller_grows_tokens_until_cache_target() -> None:
    from vllm_regenerate import TokenAdmissionController

    controller = TokenAdmissionController(
        initial_size=4,
        max_size=32,
        max_batched_tokens=256,
        initial_batched_tokens=64,
        growth_factor=2.0,
        gpu_cache_target=0.90,
        gpu_cache_hard=0.98,
    )

    assert controller.current_token_budget == 64
    controller.record_success(4, cache_usage=0.40)
    assert controller.current_size == 8
    assert controller.current_token_budget == 128
    controller.record_success(8, cache_usage=0.92)
    assert controller.current_size == 8
    assert controller.current_token_budget == 128


def test_token_admission_controller_backs_off_tokens_on_cache_pressure() -> None:
    from vllm_regenerate import TokenAdmissionController

    controller = TokenAdmissionController(
        initial_size=16,
        max_size=64,
        max_batched_tokens=512,
        initial_batched_tokens=256,
    )

    controller.record_success(16, cache_usage=0.99)
    assert controller.current_size == 8
    assert controller.current_token_budget == 128


def test_retryable_vllm_error_is_retried_then_returns(monkeypatch) -> None:
    from vllm_regenerate import VLLMRequestError, _retryable_complete

    class FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise VLLMRequestError(503, "http://server/v1/completions", "busy")
            return "ok"

    monkeypatch.setattr("vllm_regenerate.time.sleep", lambda _seconds: None)
    client = FlakyClient()
    assert _retryable_complete(
        client,
        {"prompt_ids": [1], "generation_budget": 2, "temperature": 0.0},
        seed=42,
        max_retries=1,
    ) == "ok"
    assert client.calls == 2
