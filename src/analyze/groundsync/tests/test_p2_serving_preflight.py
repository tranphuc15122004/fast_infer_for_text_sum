from src.analyze.groundsync.p2_serving_preflight import check_serving_environment


def test_serving_preflight_has_explicit_status_and_checks():
    result = check_serving_environment()
    assert result["status"] in {"READY", "UNAVAILABLE"}
    assert result["decision"] in {"READY_FOR_SERVING_BENCHMARK", "UNAVAILABLE"}
    assert "vllm_importable" in result["checks"]
    assert "canonical_server_repo_mounted" in result["checks"]
