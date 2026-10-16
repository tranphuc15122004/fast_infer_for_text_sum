"""CPU contracts for collection, resume, and scientific diagnostics."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def test_collection_continues_after_unsupported_and_failed_job(tmp_path):
    from Benchmark.vllm_research.execution import collect_jobs
    from Benchmark.vllm_research.plan import ResearchJob

    jobs = [
        ResearchJob("bad", "A1", "delete", "replay", {}, (), "same_prefix_replay"),
        ResearchJob("err", "E2E", "ar", "generation", {}, (), "native_generation"),
        ResearchJob("ok", "E2E", "dflash_full", "generation", {}, (), "native_generation"),
    ]
    visited = []
    def execute(job, job_dir):
        visited.append(job.id)
        if job.id == "err":
            raise RuntimeError("request failed")
        return {"status": "succeeded", "records": [{"sample_id": "s", "quality_valid": False}]}

    results = collect_jobs(jobs, tmp_path, {"native_generation": {"available": True}}, execute)
    assert [row["status"] for row in results] == ["unsupported", "failed", "succeeded"]
    assert visited == ["err", "ok"]
    assert results[-1]["records"][0]["quality_valid"] is False


def test_resume_does_not_rerun_success_but_retries_failed(tmp_path):
    from Benchmark.vllm_research.execution import collect_jobs
    from Benchmark.vllm_research.plan import ResearchJob

    jobs = [ResearchJob("a", "E2E", "ar", "generation", {}, (), "native_generation")]
    calls = []
    def execute(job, directory):
        calls.append(job.id)
        return {"status": "succeeded", "records": []}
    caps = {"native_generation": {"available": True}}
    collect_jobs(jobs, tmp_path, caps, execute)
    collect_jobs(jobs, tmp_path, caps, execute, resume=True)
    assert calls == ["a"]


def test_missing_dependency_is_terminal_and_does_not_prevent_fixed_arm(tmp_path):
    from Benchmark.vllm_research.execution import collect_jobs
    from Benchmark.vllm_research.plan import ResearchJob
    jobs = [
        ResearchJob("learned", "B3", "adaptive", "generation", {}, ("fit",), "adaptive_verification"),
        ResearchJob("fixed", "B3", "fixed_k", "generation", {"k": 7}, (), "verification_prefix"),
    ]
    result = collect_jobs(jobs, tmp_path,
        {"verification_prefix": {"available": True}, "adaptive_verification": {"available": True}},
        lambda job, directory: {"status": "succeeded", "records": []})
    assert [row["status"] for row in result] == ["dependency_missing", "succeeded"]


def test_campaign_finished_is_distinct_from_execution_complete():
    from Benchmark.vllm_research.execution import summarize
    report = summarize([{"status": "succeeded"}, {"status": "unsupported"}])
    assert report["campaign_status"] == "finished"
    assert report["execution_status"] == "partial"
    assert report["unsupported"] == 1


def test_batch_timing_uses_request_metrics_and_makespan_separately():
    from Benchmark.vllm_research.execution import batch_totals
    report = batch_totals([{"output_tokens": 10, "e2e_ms": 100}, {"output_tokens": 20, "e2e_ms": 300}], 400)
    assert report["throughput_tok_s"] == pytest.approx(75)
    assert report["qps"] == pytest.approx(5)
    assert report["latency_p50_ms"] == pytest.approx(200)


def test_resume_refuses_changed_job(tmp_path):
    from Benchmark.vllm_research.execution import collect_jobs
    from Benchmark.vllm_research.plan import ResearchJob
    job = ResearchJob("a", "E2E", "ar", "generation", {"max_tokens": 10}, (), "native_generation")
    caps = {"native_generation": {"available": True}}
    collect_jobs([job], tmp_path, caps, lambda j, p: {"status": "succeeded"})
    changed = ResearchJob("a", "E2E", "ar", "generation", {"max_tokens": 20}, (), "native_generation")
    with pytest.raises(ValueError, match="identity"):
        collect_jobs([changed], tmp_path, caps, lambda j, p: {}, resume=True)
