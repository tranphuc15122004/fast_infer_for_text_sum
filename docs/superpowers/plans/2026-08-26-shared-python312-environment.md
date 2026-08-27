# Shared Python 3.12 Environment Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chuyển toàn bộ launcher và subprocess của repo sang một venv Python 3.12 dùng `requirements.txt`, xóa các uv environment cũ và xác minh offline.

**Architecture:** Một runtime helper shell resolve interpreter theo thứ tự `FAST_INFER_PYTHON`, `FAST_INFER_VENV`, venv đang activate và `.venv`; mọi wrapper gọi trực tiếp interpreter đã validate. Setup dùng `uv venv`/`uv pip` với `--offline`, không còn uv project hoặc lock riêng.

**Tech Stack:** Bash, Python 3.12, `uv pip`, pytest contract tests, local model/package cache.

**Spec:** `docs/superpowers/specs/2026-08-26-shared-python312-environment-design.md`

## Global Constraints

- Server đích dùng Python 3.12 và không có internet trực tiếp.
- `requirements.txt` là dependency manifest duy nhất; giữ nguyên các path/package server-specific và chỉ bổ sung dependency project còn thiếu khi codebase cần.
- Execution dùng executable của venv chung; không dùng `uv run --project`, root uv project hoặc env riêng.
- Cài đặt dependency phải có `--offline`; thiếu Python/wheel/path phải báo lỗi cụ thể.
- Giữ nguyên `PYTHONPATH`, cwd, config arguments, output schema và semantics của baseline.
- Không tải model/dataset trong static checks; runtime smoke chỉ pass khi cache/artefact sẵn có.

---

### Task 1: Add the shared runtime contract and offline setup

**Files:**
- Create: `scripts/common/runtime.sh`
- Create: `scripts/setup_venv.sh`
- Create: `tests/test_shared_runtime_contract.py`
- Modify: `.gitignore`

**Interfaces:**
- `scripts/common/runtime.sh` exports `FAST_INFER_PYTHON` and provides `fast_infer_resolve_python` plus `fast_infer_require_python312`.
- `scripts/setup_venv.sh [--recreate|--check]` creates/checks `$FAST_INFER_VENV` or `$ROOT/.venv`, then installs `requirements.txt` with `uv pip --offline`.
- The test reads all main shell entrypoints without importing heavyweight ML packages.

- [x] **Step 1: Write the failing contract test**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = sorted((ROOT / "scripts").glob("run_*.sh"))


def test_shared_runtime_helper_and_offline_setup_exist():
    helper = ROOT / "scripts/common/runtime.sh"
    setup = ROOT / "scripts/setup_venv.sh"
    assert helper.is_file()
    assert setup.is_file()
    assert "--offline" in setup.read_text()


def test_main_runners_source_shared_runtime():
    for runner in RUNNERS:
        if runner.name == "run.sh":
            continue
        text = runner.read_text()
        assert "runtime.sh" in text, runner
        assert "uv run" not in text, runner
        assert "--project" not in text, runner
```

- [x] **Step 2: Run the contract test and verify it fails for the missing helper**

Run: `pytest -q tests/test_shared_runtime_contract.py`

Expected: FAIL because `scripts/common/runtime.sh` and `scripts/setup_venv.sh` do not exist yet.

- [x] **Step 3: Implement the runtime helper**

`runtime.sh` resolves the executable in this order: explicit `FAST_INFER_PYTHON`,
`FAST_INFER_VENV/bin/python`, active `$VIRTUAL_ENV/bin/python`, then
`$ROOT/.venv/bin/python`. It must reject a missing executable and run:

```bash
"$FAST_INFER_PYTHON" -c 'import sys; raise SystemExit("Python 3.12 required") if sys.version_info[:2] != (3, 12) else None'
```

The helper must be sourceable from every wrapper regardless of current cwd.

- [x] **Step 4: Implement offline venv setup**

`setup_venv.sh` must:

1. resolve a Python 3.12 executable from `PYTHON312_BIN`, `python3.12`, or `uv python find 3.12` with a writable task-local `UV_CACHE_DIR` only when explicitly configured;
2. use `uv venv --python <resolved-3.12> <venv>` (or recreate only when `--recreate` is supplied);
3. validate `<venv>/bin/python` as Python 3.12;
4. run `uv pip install --offline --python <venv>/bin/python -r requirements.txt`;
5. never call `curl`, `uv sync`, `uv run`, or an online installer.

- [x] **Step 5: Run the contract test again**

Run: `pytest -q tests/test_shared_runtime_contract.py`

Expected: the helper/setup assertions pass, while the complete launcher assertion still fails on the old wrappers; those failures are the red checkpoint for Task 2.

### Task 2: Migrate every main shell launcher

**Files:**
- Modify: `scripts/run_dflash.sh`
- Modify: `scripts/run_dflash_gsm8k.sh`
- Modify: `scripts/run_eagle3_qwen3.sh`
- Modify: `scripts/run_fastkv.sh`
- Modify: `scripts/run_flexprefill.sh`
- Modify: `scripts/run_gemfilter.sh`
- Modify: `scripts/run_higoe.sh`
- Modify: `scripts/run_llmlingua.sh`
- Modify: `scripts/run_longspec.sh`
- Modify: `scripts/run_magicdec.sh`
- Modify: `scripts/run_minference.sh`
- Modify: `scripts/run_rocketkv.sh`
- Modify: `scripts/run_semantic_selection.sh`
- Modify: `scripts/run_specextend.sh`
- Modify: `scripts/run_specprefill.sh`
- Modify: `scripts/run_qwen3_long_profile.sh`
- Modify: `scripts/run_representative_100.sh`
- Modify: `scripts/run.sh`

**Interfaces:**
- Every wrapper sources `scripts/common/runtime.sh` after computing `ROOT`.
- Every Python launch is `"$FAST_INFER_PYTHON" ...`.
- The dispatcher remains a thin wrapper and does not resolve a different environment.

- [x] **Step 1: Extend the contract test with the forbidden invocation checks**

```python
def test_main_runners_do_not_bypass_shared_interpreter():
    forbidden = ("uv run", "--project", " exec python ", " python3 ")
    for runner in RUNNERS:
        text = runner.read_text()
        if runner.name == "run.sh":
            continue
        assert 'source "$ROOT/scripts/common/runtime.sh"' in text, runner
        for token in forbidden:
            assert token not in text, (runner, token)
```

- [x] **Step 2: Run the extended test and observe failures**

Run: `pytest -q tests/test_shared_runtime_contract.py`

Expected: FAIL listing runners that still contain `uv run`, `python3`, or do not source the helper.

- [x] **Step 3: Replace launcher execution commands**

For each wrapper, source the helper and replace `uv run --project ... --locked python` with
`"$FAST_INFER_PYTHON"`. Preserve all arguments and environment exports. Replace the
representative runner's direct `python3` helper calls and collector invocation with
`"$FAST_INFER_PYTHON"`.

- [x] **Step 4: Run shell syntax and contract checks**

Run: `for f in scripts/*.sh scripts/common/*.sh; do bash -n "$f"; done`

Run: `pytest -q tests/test_shared_runtime_contract.py`

Expected: exit 0 and all launcher contract tests pass.

### Task 3: Force Python subprocesses to inherit the shared interpreter

**Files:**
- Modify: `scripts/infer_magicdec.py`
- Modify: `scripts/magicdec_prepare_checkpoint.py`
- Modify: `scripts/infer_specextend.py`
- Modify: `scripts/infer_longspec.py`
- Modify: `tests/test_shared_runtime_contract.py`

**Interfaces:**
- Child benchmark commands use `sys.executable`.
- Torch distributed launch uses `sys.executable -m torch.distributed.run` when requested.
- Existing child environment variables and cwd remain unchanged.

- [x] **Step 1: Add a source-level subprocess contract test**

```python
def test_python_subprocesses_do_not_use_ambient_python():
    for name in ("infer_magicdec.py", "magicdec_prepare_checkpoint.py",
                 "infer_specextend.py", "infer_longspec.py"):
        text = (ROOT / "scripts" / name).read_text()
        assert '"python"' not in text, name
        assert '"python3"' not in text, name
        assert "sys.executable" in text, name
```

- [x] **Step 2: Run the test and verify the existing ambient commands fail it**

Run: `pytest -q tests/test_shared_runtime_contract.py::test_python_subprocesses_do_not_use_ambient_python`

Expected: FAIL for at least SpecExtend/LongSpec/prepare scripts that currently build commands with `python`.

- [x] **Step 3: Replace child command prefixes**

Use `[sys.executable, ...]` for all Python child commands. In MagicDec, use
`[sys.executable, "-m", "torch.distributed.run", ...]` instead of a bare `torchrun`
when `--use-torchrun` is selected.

- [x] **Step 4: Run subprocess contract and compile checks**

Run: `pytest -q tests/test_shared_runtime_contract.py::test_python_subprocesses_do_not_use_ambient_python`

Run: `python3 -m compileall -q scripts tests/test_shared_runtime_contract.py`

Expected: both commands exit 0 without importing model packages.

### Task 4: Remove old uv venvs and update repository documentation

**Files:**
- Delete: `.venv/`
- Delete: `envs/*/.venv/`
- Delete: `scripts/setup_envs.sh`
- Modify: `scripts/bootstrap.sh`
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `AGENTS.md`
- Modify: baseline docs under `docs/baselines/`
- Modify: `docs/representative_100_benchmark.md`
- Modify: `scripts/common/metrics.py`
- Modify: `scripts/common/rouge.py`

**Interfaces:**
- The documented setup command is `bash scripts/setup_venv.sh`.
- Bootstrap is offline-only and calls setup once.
- Docs describe `.venv`/`requirements.txt`, not env groups or uv locks.

- [x] **Step 1: Add documentation contract assertions**

```python
def test_legacy_uv_venvs_are_removed():
    assert not (ROOT / ".venv").exists()
    assert not any((ROOT / "envs").glob("*/.venv"))
    assert not (ROOT / "scripts/setup_envs.sh").exists()


def test_user_facing_docs_use_shared_setup():
    for name in ("README.md", "docs/README.md", "AGENTS.md"):
        text = (ROOT / name).read_text()
        assert "setup_venv.sh" in text, name
```

- [x] **Step 2: Run the new assertions and verify they fail before deletion**

Run: `pytest -q tests/test_shared_runtime_contract.py::test_legacy_uv_venvs_are_removed tests/test_shared_runtime_contract.py::test_user_facing_docs_use_shared_setup`

Expected: FAIL because old project metadata/envs exist and docs still describe them.

- [x] **Step 3: Delete old venv directories and update bootstrap/ignore rules**

Remove the root and group `.venv` directories and the old setup script. Keep
env-group manifests/locks as archival references, but mark them as unused.
Update `.gitignore` to ignore `.venv/` and task-local uv caches.
Update bootstrap to require preinstalled `uv` and call `scripts/setup_venv.sh --offline`.

- [x] **Step 4: Rewrite user-facing docs and helper module comments**

Replace Python 3.11/env-group/lock instructions with Python 3.12, `.venv`,
`requirements.txt`, `FAST_INFER_PYTHON`, `FAST_INFER_VENV`, and offline cache/path
requirements. Preserve baseline mappings and known model/cache limitations.

- [x] **Step 5: Run documentation contract and forbidden-reference scans**

Run: `pytest -q tests/test_shared_runtime_contract.py::test_legacy_uv_venvs_are_removed tests/test_shared_runtime_contract.py::test_user_facing_docs_use_shared_setup`

Run: `rg -n --hidden -g '!requirements.txt' -g '!*.pyc' -g '!*.lock' 'uv run|--project|envs/|Python 3\.11' scripts README.md docs AGENTS.md pyproject.toml 2>/dev/null`

Expected: contract tests pass; remaining matches are only historical/external references explicitly outside the active runtime scope, or the command exits 1 with no matches.

### Task 5: Validate the shared environment and debug runtime paths

**Files:**
- Create: `scripts/check_shared_env.py`
- Modify: `scripts/common/runtime.sh`
- Modify: `tests/test_shared_runtime_contract.py`
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

**Interfaces:**
- `scripts/check_shared_env.py` performs no model download and checks Python 3.12 plus import/version metadata for the requirements-driven runtime.
- Exit 0 means interpreter and requested imports are available; exit 1 reports each failed check without hiding failures.

- [x] **Step 1: Add the preflight contract test**

```python
def test_preflight_is_offline_and_non_model_loading():
    text = (ROOT / "scripts/check_shared_env.py").read_text()
    assert "local_files_only" in text or "offline" in text
    assert "from_pretrained" not in text
```

- [x] **Step 2: Run the test and verify it fails because preflight is absent**

Run: `pytest -q tests/test_shared_runtime_contract.py::test_preflight_is_offline_and_non_model_loading`

Expected: FAIL because `scripts/check_shared_env.py` does not exist.

- [x] **Step 3: Implement import/version preflight**

The preflight invokes the resolved interpreter, prints `sys.version`, CUDA availability,
and import results for `torch`, `transformers`, `vllm`, `triton`, `flashinfer`, `llmlingua`,
`dflash`, and `sentence_transformers`. It must not call Hugging Face loading APIs.

- [x] **Step 4: Run all static verification**

Run: `for f in scripts/*.sh scripts/common/*.sh; do bash -n "$f"; done`

Run: `pytest -q tests`

Run: `FAST_INFER_VENV="$PWD/.venv" bash scripts/setup_venv.sh --check`

Expected: static tests pass. The setup/preflight command either passes on a prepared Python 3.12 offline cache or exits with an explicit missing-Python/dependency/path message.

- [x] **Step 5: Run the server simulation and classify every baseline**

On an available Python 3.12 interpreter with the server cache, run:

```bash
FAST_INFER_VENV="$PWD/.venv" python3 scripts/check_shared_env.py
for baseline in eagle3 dflash llmlingua fastkv rocketkv gemfilter specprefill minference magicdec longspec specextend higoe semantic_selection flexprefill; do
  bash scripts/run.sh "$baseline" --smoke >"/tmp/fast_infer_${baseline}.log" 2>&1 || true
done
```

Record each result as `PASS`, launcher/dependency failure, CUDA/binary failure, or missing model/data cache. Do not convert unavailable Python 3.12 or server-only local paths into false passes.
