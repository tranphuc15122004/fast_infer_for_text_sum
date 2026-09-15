from pathlib import Path

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def test_server_manifest_does_not_embed_public_package_indexes() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "--index-url" not in requirements
    assert "--extra-index-url" not in requirements
    assert "download.pytorch.org" not in requirements
    assert "flashinfer.ai" not in requirements


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
        "sglang",
        "sglang-kernel",
        "yunchang",
        "cuda-tile",
        "nvidia-cuda-tileiras",
        "llmlingua",
    } <= names


def test_server_manifest_matches_the_mirrored_torch_214_stack() -> None:
    requirements = {
        Requirement(line).name.lower().replace("_", "-"): Requirement(line)
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    expected = {
        "torch": "==2.14.0+cu130",
        "torchvision": "==0.29.0+cu130",
        "flashinfer-python": "==0.6.13",
        "flashinfer-cubin": "==0.6.13",
        "triton": "==3.8.0",
        "sglang": "==0.5.19",
        "sglang-kernel": "==0.4.7",
        "vllm": "==0.29.0",
    }
    for name, specifier in expected.items():
        assert str(requirements[name].specifier) == specifier

    assert "torchaudio" not in requirements
    assert "flashinfer-jit-cache" not in requirements
    assert "flash-attn" not in requirements
