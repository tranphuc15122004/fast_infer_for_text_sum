from pathlib import Path

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def test_server_requirements_are_portable_and_include_b200_cache_stack() -> None:
    names: set[str] = set()
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        assert not line.startswith(("-e ", "git+"))
        assert "@ file://" not in line
        assert "@ http://" not in line
        assert "@ https://" not in line
        assert "+ubuntu" not in line
        requirement = Requirement(line)
        normalized = requirement.name.lower().replace("_", "-")
        assert normalized not in names
        names.add(normalized)

    assert {
        "torch",
        "torchvision",
        "flashinfer-python",
        "flashinfer-cubin",
        "flashinfer-jit-cache",
        "sglang",
        "sglang-kernel",
        "yunchang",
        "cuda-tile",
        "nvidia-cuda-tileiras",
        "llmlingua",
    } <= names
