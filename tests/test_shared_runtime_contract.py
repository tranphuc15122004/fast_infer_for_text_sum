from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = sorted((ROOT / "scripts").glob("run_*.sh"))


def test_shared_runtime_helper_and_offline_setup_exist():
    helper = ROOT / "scripts/common/runtime.sh"
    setup = ROOT / "scripts/setup_venv.sh"
    assert helper.is_file()
    assert setup.is_file()
    assert "--offline" in setup.read_text()


def test_project_manifest_targets_python312():
    assert (ROOT / ".python-version").read_text().strip() == "3.12"
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'requires-python = "==3.12.*"' in pyproject


def test_main_runners_source_shared_runtime():
    for runner in RUNNERS:
        if runner.name == "run.sh":
            continue
        text = runner.read_text()
        assert "runtime.sh" in text, runner
        assert "uv run" not in text, runner
        assert "--project" not in text, runner


def test_main_runners_do_not_bypass_shared_interpreter():
    forbidden = ("uv run", "--project", " exec python ", " python3 ")
    for runner in RUNNERS:
        if runner.name == "run.sh":
            continue
        text = runner.read_text()
        assert 'source "$ROOT/scripts/common/runtime.sh" || exit 1' in text, runner
        for token in forbidden:
            assert token not in text, (runner, token)


def test_python_subprocesses_do_not_use_ambient_python():
    for name in ("infer_magicdec.py", "magicdec_prepare_checkpoint.py",
                 "infer_specextend.py", "infer_longspec.py"):
        text = (ROOT / "scripts" / name).read_text()
        assert '"python"' not in text, name
        assert '"python3"' not in text, name
        assert "sys.executable" in text, name


def test_legacy_uv_venvs_are_removed():
    assert not any((ROOT / "envs").glob("*/.venv"))
    assert not (ROOT / "scripts/setup_envs.sh").exists()


def test_user_facing_docs_use_shared_setup():
    for name in ("README.md", "docs/README.md", "AGENTS.md"):
        text = (ROOT / name).read_text()
        assert "setup_venv.sh" in text, name


def test_preflight_is_offline_and_non_model_loading():
    text = (ROOT / "scripts/check_shared_env.py").read_text()
    assert "offline" in text
    assert "from_pretrained" not in text


def test_preflight_checks_local_dflash_and_cuda():
    text = (ROOT / "scripts/check_shared_env.py").read_text()
    assert '"dflash"' in text
    assert "cuda.is_available" in text


def test_requirements_cover_runtime_preflight_modules():
    requirements = (ROOT / "requirements.txt").read_text()
    assert "sentence-transformers" in requirements
    preflight = (ROOT / "scripts/check_shared_env.py").read_text()
    assert "externals" in preflight


def test_setup_reports_missing_local_requirement_sources():
    text = (ROOT / "scripts/setup_venv.sh").read_text()
    assert "file://" in text
    assert "local requirement source missing" in text
    assert "https://" in text
    assert "uv cache" in text
    assert "uv add" in text
    assert "--requirements \"$ROOT/requirements.txt\"" in text


def test_dispatcher_accepts_smoke_flag_without_treating_it_as_config():
    env = dict(os.environ)
    env["FAST_INFER_PYTHON"] = "/tmp/fast-infer-python-that-does-not-exist"
    proc = subprocess.run(
        ["bash", "scripts/run.sh", "fastkv", "--smoke"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "Configuration file not found: --smoke" not in output
    assert "Shared Python interpreter not found" in output


def test_dflash_gsm8k_wrapper_consumes_smoke_flag(tmp_path):
    fake_python = tmp_path / "python"
    argument_log = tmp_path / "args.txt"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" == \"-c\" ]]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$FAKE_ARGUMENT_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = dict(os.environ)
    env["FAST_INFER_PYTHON"] = str(fake_python)
    env["FAKE_ARGUMENT_LOG"] = str(argument_log)
    proc = subprocess.run(
        [
            "bash",
            "scripts/run.sh",
            "dflash",
            "config/dflash_gsm8k.env",
            "--smoke",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    args = argument_log.read_text(encoding="utf-8").splitlines()
    assert "--smoke" not in args
    assert args[args.index("--max-samples") + 1] == "1"
    assert args[args.index("--max-new-tokens") + 1] == "128"


def test_semantic_selection_wrapper_has_a_direct_smoke_input(tmp_path):
    fake_python = tmp_path / "python"
    argument_log = tmp_path / "args.txt"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" == \"-c\" ]]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$FAKE_ARGUMENT_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = dict(os.environ)
    env["FAST_INFER_PYTHON"] = str(fake_python)
    env["FAKE_ARGUMENT_LOG"] = str(argument_log)
    proc = subprocess.run(
        ["bash", "scripts/run.sh", "semantic_selection", "--smoke"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    args = argument_log.read_text(encoding="utf-8").splitlines()
    assert args[args.index("--input") + 1] == "data/smoke_long_docs.jsonl"
    assert args[args.index("--limit") + 1] == "1"
    assert "--smoke" not in args
