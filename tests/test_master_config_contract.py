from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "scripts" / "common" / "config.sh"


def _write_pointer(tmp_path: Path, master: Path, *, pointer_name: str = "master.path") -> Path:
    pointer = tmp_path / "config" / pointer_name
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(master) + "\n", encoding="utf-8")
    return pointer


def run_loader(
    tmp_path: Path,
    *,
    master_text: str,
    baseline: str | None = None,
    caller_env: dict[str, str] | None = None,
    use_pointer: bool = True,
) -> subprocess.CompletedProcess[str]:
    master = tmp_path / "master.env"
    master.write_text(master_text, encoding="utf-8")
    env = dict(os.environ)
    env["ROOT"] = str(tmp_path)
    if use_pointer:
        _write_pointer(tmp_path, master)
    if caller_env:
        env.update(caller_env)

    requested = [
        "FAST_INFER_MASTER_CONFIG",
        "MODEL_TARGET",
        "TARGET_MODEL",
        "DRAFT_MODEL",
        "DATA_FILE",
        "MAX_NEW_TOKENS",
        "MAX_GEN_LEN",
        "SMOKE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ]
    body = [
        f'source "{LOADER}"',
        "fast_infer_load_master" if baseline is None else f'fast_infer_load_config "{baseline}"',
        'printf \'MASTER=%s\\n\' "$(fast_infer_master_path)"',
    ]
    body.extend(f'printf \'{name}=%s\\n\' "${{{name}-}}"' for name in requested)
    return subprocess.run(
        ["bash", "-c", "set -e\n" + "\n".join(body)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def test_master_path_reads_plain_text_pointer(tmp_path):
    master = tmp_path / "master.env"
    master.write_text("MODEL_TARGET=/models/target\n", encoding="utf-8")
    pointer = _write_pointer(tmp_path, master)
    result = run_loader(tmp_path, master_text=master.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stderr
    assert f"MASTER={master}" in result.stdout
    assert pointer.read_text(encoding="utf-8").strip() == str(master)


def test_repository_keeps_only_the_master_pointer_in_config_directory():
    assert sorted(path.name for path in (ROOT / "config").iterdir()) == ["master.path"]
    assert (ROOT / "docs/fast_infer_master.example.env").is_file()


def test_all_baseline_launchers_use_the_shared_config_loader():
    wrappers = sorted((ROOT / "scripts").glob("run_*.sh"))
    for wrapper in wrappers:
        if wrapper.name in {"run.sh", "run_b200_smoke.sh", "run_representative_100.sh"}:
            continue
        text = wrapper.read_text(encoding="utf-8")
        assert 'source "$ROOT/scripts/common/config.sh"' in text, wrapper
        assert "config/" not in text, wrapper


def test_active_docs_describe_master_config_instead_of_deleted_per_baseline_files():
    for path in (ROOT / "README.md", ROOT / "docs/README.md", ROOT / "AGENTS.md"):
        text = path.read_text(encoding="utf-8")
        assert "master.path" in text
        assert "config/<baseline>.env" not in text


def test_environment_master_path_overrides_pointer(tmp_path):
    pointer_master = tmp_path / "pointer-master.env"
    env_master = tmp_path / "env-master.env"
    pointer_master.write_text("MODEL_TARGET=/models/pointer\n", encoding="utf-8")
    env_master.write_text("MODEL_TARGET=/models/environment\n", encoding="utf-8")
    _write_pointer(tmp_path, pointer_master)
    result = run_loader(
        tmp_path,
        master_text=pointer_master.read_text(encoding="utf-8"),
        caller_env={"FAST_INFER_MASTER_CONFIG": str(env_master)},
    )
    assert result.returncode == 0, result.stderr
    assert f"MASTER={env_master}" in result.stdout
    assert "MODEL_TARGET=/models/environment" in result.stdout


def test_missing_master_pointer_fails_with_actionable_error(tmp_path):
    result = run_loader(
        tmp_path,
        master_text="MODEL_TARGET=/models/target\n",
        use_pointer=False,
    )
    assert result.returncode != 0
    assert "master.path" in result.stderr


def test_empty_master_pointer_fails_without_sourcing_old_configs(tmp_path):
    _write_pointer(tmp_path, tmp_path / "missing-master.env")
    (tmp_path / "config" / "master.path").write_text("\n# no path\n", encoding="utf-8")
    result = run_loader(
        tmp_path,
        master_text="MODEL_TARGET=/models/target\n",
        use_pointer=False,
    )
    assert result.returncode != 0
    assert "empty" in result.stderr.lower()


def test_dflash_mapping_preserves_caller_override(tmp_path):
    result = run_loader(
        tmp_path,
        baseline="dflash",
        master_text=(
            "MODEL_TARGET=/models/target\n"
            "MODEL_DFLASH_DRAFT=/models/draft\n"
            "DATA_INPUT=/data/input.jsonl\n"
            "RUN_MODE=smoke\n"
            "RUN_SAMPLES=1\n"
            "RUN_MAX_NEW_TOKENS=8\n"
            "RUN_TEMPERATURE=0\n"
            "DFLASH_MODE=representative\n"
            "DFLASH_BLOCK_SIZE=4\n"
        ),
        caller_env={"TARGET_MODEL": "/models/override"},
    )
    assert result.returncode == 0, result.stderr
    assert "TARGET_MODEL=/models/override" in result.stdout
    assert "DRAFT_MODEL=/models/draft" in result.stdout
    assert "DATA_FILE=/data/input.jsonl" in result.stdout
    assert "SMOKE=1" in result.stdout
    assert "HF_HUB_OFFLINE=1" in result.stdout
    assert "TRANSFORMERS_OFFLINE=1" in result.stdout


def test_baseline_specific_values_override_shared_run_defaults(tmp_path):
    result = run_loader(
        tmp_path,
        baseline="gemfilter",
        master_text=(
            "MODEL_TARGET=/models/target\n"
            "OUTPUT_ROOT=outputs\n"
            "RUN_MODE=full\n"
            "RUN_MAX_NEW_TOKENS=99\n"
            "GEMFILTER_MAX_NEW_TOKENS=17\n"
        ),
    )
    assert result.returncode == 0, result.stderr
    assert "MAX_GEN_LEN=17" in result.stdout
    assert "SMOKE=0" in result.stdout
