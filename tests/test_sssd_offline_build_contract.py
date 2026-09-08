from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SSSD_BUILD = ROOT / "externals/SSSD/sssd_speculator/sssd_speculator/CMakeLists.txt"
SSSD_COMPAT = (
    ROOT
    / "externals/SSSD/sssd_speculator/third_party/spdlog_compat/include"
    / "spdlog/spdlog.h"
)


def test_sssd_build_has_no_remote_fetch_and_uses_local_dependency_contract():
    """The native SSSD build must not require git access on the benchmark server."""

    cmake = SSSD_BUILD.read_text(encoding="utf-8")

    assert "GIT_REPOSITORY" not in cmake
    assert "SSSD_LIBSAIS_SOURCE_DIR" in cmake
    assert "SSSD_SPDLOG_SOURCE_DIR" in cmake
    assert "spdlog_compat" in cmake
    assert (ROOT / "externals/SSSD/sssd_speculator/evaluation/REST/DraftRetriever/src/libsais/libsais.c").is_file()
    assert SSSD_COMPAT.is_file()
