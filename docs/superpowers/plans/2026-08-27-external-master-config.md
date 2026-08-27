# External Master Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Replace all per-baseline configuration files with one repository pointer and one external master shell-env config consumed by every inference launcher.

**Architecture:** Add scripts/common/config.sh as the only configuration loader. It reads FAST_INFER_MASTER_CONFIG or the path in config/master.path, sources canonical namespaced values, maps them to the legacy variables expected by each adapter, and preserves caller overrides. All launchers use the loader; B200 and representative runners pass per-run overrides through the child environment instead of creating config files.

**Tech Stack:** Bash 4+, Python 3.12 runtime, pytest contract tests, existing shell launchers and vendored baseline adapters.

**Spec:** docs/superpowers/specs/2026-08-27-external-master-config-design.md

## Global Constraints

- The server has no direct internet connection; the master config sets HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1.
- Python runtime remains the shared Python 3.12 executable resolved by scripts/common/runtime.sh.
- No YAML/TOML parser or new runtime dependency is added.
- Model and dataset values remain external absolute paths on B200.
- User-facing documentation remains Vietnamese.
- Output JSONL schema, baseline algorithm behavior, and existing CLI inference arguments remain unchanged.
- The final repository has one config pointer at config/master.path; all old config/*.env files are deleted.

---

### Task 1: Add failing tests for pointer and master loader

**Files:**
- Create: tests/test_master_config_contract.py
- Modify: tests/test_shared_runtime_contract.py only where it asserts old config files or old config arguments

**Interfaces:**
- The tests define fast_infer_master_path, fast_infer_load_master, and fast_infer_load_config.
- A temporary master uses FI_PYTHON, FI_HF_HOME, MODEL_TARGET, MODEL_DFLASH_DRAFT, DATA_INPUT, RUN_MODE, RUN_SAMPLES, RUN_MAX_NEW_TOKENS, RUN_TEMPERATURE, DFLASH_MODE, and DFLASH_BLOCK_SIZE.

- [x] Step 1: Test pointer resolution and environment override

~~~python
def test_master_path_reads_plain_text_pointer(tmp_path):
    master = tmp_path / "master.env"
    master.write_text("MODEL_TARGET=/models/target\n", encoding="utf-8")
    pointer = tmp_path / "master.path"
    pointer.write_text(str(master) + "\n", encoding="utf-8")
    result = run_loader(pointer=pointer)
    assert result.returncode == 0
    assert result.stdout.splitlines()[-1] == str(master)


def test_environment_master_path_overrides_pointer(tmp_path):
    pointer_master = tmp_path / "pointer-master.env"
    env_master = tmp_path / "env-master.env"
    pointer_master.write_text("MODEL_TARGET=/models/pointer\n", encoding="utf-8")
    env_master.write_text("MODEL_TARGET=/models/environment\n", encoding="utf-8")
    pointer = tmp_path / "master.path"
    pointer.write_text(str(pointer_master) + "\n", encoding="utf-8")
    result = run_loader(pointer=pointer, env_master=env_master)
    assert result.returncode == 0
    assert "MODEL_TARGET=/models/environment" in result.stdout
~~~

- [x] Step 2: Test actionable errors for missing and empty pointer

~~~python
def test_missing_master_pointer_fails_with_actionable_error(tmp_path):
    result = run_loader(pointer=tmp_path / "missing.path")
    assert result.returncode != 0
    assert "master.path" in result.stderr


def test_empty_master_pointer_fails_without_sourcing_old_configs(tmp_path):
    pointer = tmp_path / "master.path"
    pointer.write_text("\n# no path\n", encoding="utf-8")
    result = run_loader(pointer=pointer)
    assert result.returncode != 0
    assert "empty" in result.stderr.lower()
~~~

- [x] Step 3: Test DFlash mapping, offline flags, and caller precedence

~~~python
def test_dflash_mapping_preserves_caller_override(tmp_path):
    master = tmp_path / "master.env"
    master.write_text(
        "MODEL_TARGET=/models/target\n"
        "MODEL_DFLASH_DRAFT=/models/draft\n"
        "DATA_INPUT=/data/input.jsonl\n"
        "RUN_MODE=smoke\n"
        "RUN_SAMPLES=1\n"
        "RUN_MAX_NEW_TOKENS=8\n"
        "RUN_TEMPERATURE=0\n"
        "DFLASH_MODE=representative\n"
        "DFLASH_BLOCK_SIZE=4\n",
        encoding="utf-8",
    )
    result = run_loader(
        master=master,
        baseline="dflash",
        env={"TARGET_MODEL": "/models/override"},
    )
    assert result.returncode == 0
    assert "TARGET_MODEL=/models/override" in result.stdout
    assert "DRAFT_MODEL=/models/draft" in result.stdout
    assert "DATA_FILE=/data/input.jsonl" in result.stdout
    assert "SMOKE=1" in result.stdout
    assert "HF_HUB_OFFLINE=1" in result.stdout
    assert "TRANSFORMERS_OFFLINE=1" in result.stdout
~~~

- [x] Step 4: Run the new tests and verify the expected RED state

Run: pytest -q tests/test_master_config_contract.py

Expected: FAIL because config/master.path and scripts/common/config.sh do not exist.

### Task 2: Implement the common loader and production pointer

**Files:**
- Create: config/master.path
- Create: scripts/common/config.sh
- Test: tests/test_master_config_contract.py

**Interfaces:**
- fast_infer_master_path prints the resolved master path and returns nonzero for missing, empty, or non-file pointers.
- fast_infer_load_master sources the external master once, exports runtime aliases, and sets offline flags.
- fast_infer_load_config baseline maps model, data, run, and baseline-specific values to current launcher variables.
- Variables already set by the caller are never overwritten by master defaults.

- [x] Step 1: Add the pointer

Create config/master.path containing exactly:

~~~text
/workspace/shared_storage/config/fast_infer_master.env
~~~

- [x] Step 2: Implement safe pointer parsing and one-time source

Implement scripts/common/config.sh with these functions:

~~~bash
fast_infer_master_path
fast_infer_load_master
fast_infer_default_from
fast_infer_load_config
~~~

The loader must resolve FAST_INFER_MASTER_CONFIG first, then config/master.path; resolve relative paths against ROOT; reject empty/missing paths; source the master with set -a; and set HF_HUB_OFFLINE=1 plus TRANSFORMERS_OFFLINE=1 when FI_OFFLINE is not 0.

- [x] Step 3: Implement canonical runtime/model/data aliases

Map values without overwriting caller values:

~~~text
FI_PYTHON                 -> FAST_INFER_PYTHON
FI_DEVICE                 -> B200_DEVICE and DEVICE
FI_GPU_IDS                -> CUDA_VISIBLE_DEVICES
FI_HF_HOME                -> HF_HOME
FI_TRANSFORMERS_CACHE     -> TRANSFORMERS_CACHE
FI_TRITON_CACHE           -> TRITON_CACHE_DIR
FI_FLASHINFER_CACHE       -> FLASHINFER_WORKSPACE_BASE
FI_TORCH_EXTENSIONS_CACHE -> TORCH_EXTENSIONS_DIR
FI_TARGET_GPU             -> B200_TARGET_GPU
MODEL_TARGET              -> B200_TARGET_MODEL
MODEL_DFLASH_DRAFT        -> B200_DFLASH_MODEL
DATA_INPUT                -> B200_DATA_FILE
~~~

- [x] Step 4: Implement all baseline mappings

fast_infer_load_config maps every current launcher input:

~~~text
dflash, dflash_gsm8k: TARGET_MODEL, DRAFT_MODEL, DATA_FILE, MAX_SAMPLES,
MAX_NEW_TOKENS, TEMPERATURE, BLOCK_SIZE, PROMPT, OUTPUT_FILE, BACKEND, DATASET

eagle3: BASE_MODEL, EAGLE_MODEL, DATA_FILE, BENCH_NAME, QUESTION_BEGIN,
QUESTION_END, NUM_CHOICES, TOTAL_TOKEN, DEPTH, TOP_K, OUTPUT_FILE

fastkv: MODEL, METHOD, ATTN_IMPL, DATA_FILE, MAX_SAMPLES, OUTPUT_FILE,
WINDOW_SIZE, MAX_CAPACITY_PROMPTS, RETAIN_RATE, EVICTION_MODE, NUM_RUNS

gemfilter: MODEL, TOPK, SELECT_LAYER_IDX, PROMPT, DATA_FILE, MAX_GEN_LEN,
MAX_SAMPLES, NUM_RUNS, OUTPUT_FILE

llmlingua: COMPRESSOR_MODEL, TARGET_MODEL, DOC_FILE, COMPRESSION_RATE,
MAX_SAMPLES, MAX_NEW_TOKENS, DEVICE, OUTPUT_FILE

minference: MODEL, ATTN_TYPE, MAX_NEW_TOKENS, MAX_MODEL_LEN, DEVICE,
ATTN_IMPLEMENTATION, DATA_FILE, MAX_SAMPLES, OUTPUT_FILE

specprefill: TARGET_MODEL, SPEC_MODEL, SPEC_CONFIG, MAX_TOKENS,
GPU_MEMORY_UTILIZATION, DATA_FILE, MAX_SAMPLES, OUTPUT_FILE

semantic_selection: MODEL, SELECTORS, TOKEN_BUDGETS, SMOKE_TOKEN_BUDGETS,
MAX_SAMPLES, MAX_NEW_TOKENS, DEVICE, DTYPE, ATTN_IMPLEMENTATION,
WARMUP_ROUNDS, EMBEDDING_MODEL, EMBEDDING_DEVICE, RANDOM_SEED, MMR_LAMBDA,
INPUT_FILE, OUTPUT_FILE

specextend: BASE_MODEL, DRAFT_MODEL, INPUT_FILE, SCRIPT, MODEL_NAME,
MAX_SAMPLES, MAX_GEN_LEN, MAX_INPUT_TOKENS, WARMUP_RUNS, USE_SPECEXTEND,
OUTPUT_FILE

longspec: TARGET_MODEL, DRAFT_MODEL, DATA_FILE, MODEL_NAME, METHOD, TASK,
MAX_GEN_LEN, TREE_SHAPE, MAX_SAMPLES, DATA_PATH_PREFIX, OUTPUT_FILE

magicdec: MODEL_PTH, MODEL_NAME, MAGICDEC_DATA_ROOT, BATCH_SIZE, PREFIX_LEN,
MAX_LEN, NUM_RUNS, WINDOW_SIZE, SELF_SPEC, GAMMA, DRAFT_BUDGET,
PREPARE_CHECKPOINT, REPO_ID, MODEL_KEY, USE_TORCHRUN, OUTPUT_FILE

rocketkv: TOKEN_BUDGET, SEQ_LEN, MAX_NEW_TOKENS, HEAD_DIM, NUM_RUNS, FULL,
OUTPUT_FILE

higoe: RETRIEVER_MODEL, NUM_DOCS, OUTPUT_FILE

flexprefill: MODEL, PATTERN, DATA_FILE, MAX_SAMPLES, MAX_NEW_TOKENS,
MAX_INPUT_TOKENS, SKIP_NAIVE, OUTPUT_FILE

qwen3_long_profile: MODEL, INPUT_FILE, WORD_MARKS, MAX_NEW_TOKENS, REPEATS,
WARMUP_RUNS, DEVICE, ATTN_IMPLEMENTATION, OUTPUT_DIR, LOCAL_FILES_ONLY
~~~

- [x] Step 5: Run loader tests until GREEN

Run: pytest -q tests/test_master_config_contract.py

Expected: all pointer, mapping, precedence, and offline tests pass.

### Task 3: Migrate direct launchers

**Files:**
- Modify: scripts/run_dflash.sh
- Modify: scripts/run_dflash_gsm8k.sh
- Modify: scripts/run_eagle3_qwen3.sh
- Modify: scripts/run_fastkv.sh
- Modify: scripts/run_flexprefill.sh
- Modify: scripts/run_gemfilter.sh
- Modify: scripts/run_higoe.sh
- Modify: scripts/run_llmlingua.sh
- Modify: scripts/run_longspec.sh
- Modify: scripts/run_magicdec.sh
- Modify: scripts/run_minference.sh
- Modify: scripts/run_qwen3_long_profile.sh
- Modify: scripts/run_rocketkv.sh
- Modify: scripts/run_semantic_selection.sh
- Modify: scripts/run_specextend.sh
- Modify: scripts/run_specprefill.sh
- Modify: tests/test_master_config_contract.py
- Modify: tests/test_shared_runtime_contract.py
- Modify: tests/test_dflash_wrapper_contract.py

**Interfaces:**
- Every direct launcher calls fast_infer_load_config <baseline> before runtime.sh.
- No direct launcher reads config/<baseline>.env.
- bash scripts/run.sh <baseline> [extra args] remains the normal entrypoint.
- DFlash mode comes from DFLASH_MODE=representative|gsm8k.

- [x] Step 1: Add the all-launcher contract test

~~~python
def test_all_inference_launchers_use_master_loader():
    for runner in sorted((ROOT / "scripts").glob("run_*.sh")):
        if runner.name in {"run.sh", "run_b200_smoke.sh"}:
            continue
        text = runner.read_text(encoding="utf-8")
        assert "scripts/common/config.sh" in text, runner
        assert "fast_infer_load_config" in text, runner
~~~

- [x] Step 2: Replace config-file preambles

Use this pattern after ROOT is computed:

~~~bash
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config "dflash"
source "$ROOT/scripts/common/runtime.sh" || exit 1
~~~

Use the actual baseline name for each launcher. Remove positional per-baseline CONFIG_FILE parsing and missing-config checks.

- [x] Step 3: Migrate DFlash legacy mode

run_dflash.sh uses DFLASH_MODE=representative for infer_dflash.py and execs
run_dflash_gsm8k.sh when DFLASH_MODE=gsm8k. Reject another value with exit
code 2. The GSM8K child loads the same master and does not receive a config
file argument.

- [x] Step 4: Run migration tests and shell syntax

Run: pytest -q tests/test_master_config_contract.py tests/test_shared_runtime_contract.py tests/test_dflash_wrapper_contract.py

Run: for f in scripts/*.sh scripts/common/*.sh; do bash -n "$f"; done

Expected: all selected tests pass and every shell file parses.

### Task 4: Remove B200 config overlays

**Files:**
- Modify: scripts/run_b200_smoke.sh
- Modify: scripts/b200_smoke.py
- Modify: scripts/check_b200_env.py only if canonical aliases need support
- Modify: tests/test_b200_launcher_contract.py
- Modify: tests/test_b200_smoke_runner_contract.py

**Interfaces:**
- run_b200_smoke.sh loads the external master and keeps CLI options for
  baselines, output directory, timeout, and preflight-only.
- b200_smoke.py passes child overrides through child_env.update(...).
- It does not write generated/*.env or set FAST_INFER_CONFIG_OVERLAY.

- [x] Step 1: Add a failing no-overlay test

Use a temporary master and fake ready preflight/runtime. Assert that the runner
can pass a DFlash child environment and that no .env file is created under the
output directory.

- [x] Step 2: Pass B200 overrides directly

Replace the overlay assignment with:

~~~python
child_env.update(_run_env_values(baseline, output, generated_inputs))
~~~

Remove _write_overlay, overlay path creation, and FAST_INFER_CONFIG_OVERLAY.
Keep generated input JSONL files because they are data adapters.

- [x] Step 3: Load master in the B200 shell runner

Source config.sh, call fast_infer_load_master, then source runtime.sh. Use
master values for the B200 baseline list, output directory, timeout, sample
limit, and token limit while preserving explicit CLI precedence.

- [x] Step 4: Run B200 tests

Run: pytest -q tests/test_b200_launcher_contract.py tests/test_b200_smoke_runner_contract.py tests/test_master_config_contract.py

Expected: pass without generated config overlays.

### Task 5: Refactor representative runner

**Files:**
- Modify: scripts/run_representative_100.sh
- Modify: tests/test_representative_runner_contract.py

**Interfaces:**
- The runner reads BENCH_MODE, BENCH_BASELINES, BENCH_DATASETS,
  BENCH_SAMPLES, BENCH_NEW_TOKENS, and BENCH_OUTPUT_DIR from master.
- Per (baseline, dataset) overrides are passed through env KEY=value bash
  scripts/run.sh baseline.
- It no longer reads per-baseline configs or writes outputs/.../configs/*.env.
- Dry-run remains usable without Python 3.12 or model cache.

- [x] Step 1: Add failing master-only tests

~~~python
def test_representative_runner_does_not_reference_per_baseline_configs():
    text = (ROOT / "scripts/run_representative_100.sh").read_text(encoding="utf-8")
    assert "config/$(config_for" not in text
    assert 'cat "$base_cfg"' not in text
    assert "fast_infer_load_master" in text


def test_representative_dry_run_does_not_create_config_directory(tmp_path):
    result = run_representative(
        tmp_path,
        ["--mode", "full", "--baselines", "dflash",
         "--datasets", "xsum", "--dry-run"],
    )
    assert result.returncode == 0
    assert not (tmp_path / "configs").exists()
~~~

- [x] Step 2: Load representative defaults from master before CLI

Map BENCH_MODE, BENCH_BASELINES, BENCH_DATASETS, BENCH_SAMPLES,
BENCH_NEW_TOKENS, and BENCH_OUTPUT_DIR to runner local variables before parsing
CLI so explicit CLI values win.

- [x] Step 3: Replace config generation with environment arrays

Refactor run_pair to build KEY=value assignments and execute:

~~~bash
env KEY=value-list bash scripts/run.sh "$b" > "$log" 2>&1
~~~

Keep data conversion, output cleanup, logs, PASS/FAIL counts, strict collector,
and dry-run output. Remove set_env, config_for, gen_config, and all
outputs/.../configs creation.

- [x] Step 4: Run representative tests and dry-run

Run: pytest -q tests/test_representative_runner_contract.py tests/test_master_config_contract.py

Run: bash scripts/run_representative_100.sh --baselines dflash --datasets xsum --dry-run

Expected: the plan renders from the pointer/master and no per-baseline config
directory is created.

### Task 6: Delete old configs and document the master template

**Files:**
- Delete: every tracked config/*.env file; config/master.path is the only config file retained.
- Create: docs/fast_infer_master.example.env
- Modify: README.md
- Modify: docs/README.md
- Modify: docs/representative_100_benchmark.md
- Modify: all docs/baselines/*.md files that mention config/<baseline>.env
- Modify: AGENTS.md where it describes the active config contract
- Modify: tests that assert deleted config files exist

**Interfaces:**
- docs/master_config.md contains a complete copy-paste master template with
  all canonical keys consumed by the loader.
- Documentation instructs operators to edit only
  /workspace/shared_storage/config/fast_infer_master.env.
- The two supplied B200 model paths are used in the template.

- [x] Step 1: Add a stale-reference documentation test

~~~python
def test_active_docs_use_master_config_only():
    for path in [ROOT / "README.md", ROOT / "docs/README.md", ROOT / "AGENTS.md"]:
        text = path.read_text(encoding="utf-8")
        assert "master.path" in text
        assert "config/<baseline>.env" not in text
~~~

- [x] Step 2: Write the complete Vietnamese template

The template must include runtime/cache, offline flags, canonical model paths,
data/output, common run settings, all baseline groups, representative settings,
B200 smoke settings, and:

~~~bash
MODEL_TARGET="/workspace/shared_storage/model/Llama3.1-8B-Instruct"
MODEL_DFLASH_DRAFT="/workspace/shared_storage/model/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
~~~

- [x] Step 3: Update commands and docs

Replace bash scripts/run.sh dflash config/dflash_gsm8k.env with DFLASH_MODE=gsm8k
in the master. Replace baseline-specific config tables with master variable
groups and document the pointer/loader flow.

- [x] Step 4: Delete old tracked config files

Delete only the explicit tracked files under config/ after launcher references
are removed. Do not delete model, dataset, output, cache, or external artifacts.

- [x] Step 5: Scan for stale active references

Run:

~~~bash
rg -n --hidden -S 'config/[a-z0-9_]+\.env|CONFIG_FILE|FAST_INFER_CONFIG_OVERLAY' scripts tests README.md docs AGENTS.md config
~~~

Expected: no active launcher or user-facing document references a deleted
per-baseline config or overlay.

### Task 7: Full verification and B200 handoff

**Files:**
- Modify: task_plan.md
- Modify: findings.md
- Modify: progress.md

- [x] Step 1: Run static checks

~~~bash
for f in scripts/*.sh scripts/common/*.sh; do bash -n "$f"; done
python3 -m compileall -q scripts tests
git diff --check
~~~

- [x] Step 2: Run the repository tests

Run: pytest -q tests

Expected: all repository tests pass with zero failures.

- [x] Step 3: Run fake-runtime launcher sweep

Use a temporary master and fake Python executable to capture every launcher's
arguments. Confirm the same offline flags, runtime cache paths, model aliases,
and output root reach each child.

- [ ] Step 4: Run B200 preflight after installing the external master

~~~bash
source scripts/common/config.sh
fast_infer_load_master
python3 scripts/check_b200_env.py --baselines dflash --target-gpu B200
~~~

Expected: both local model directories resolve without any internet request.

- [ ] Step 5: Run DFlash smoke on B200

~~~bash
bash scripts/run.sh dflash --smoke
~~~

Expected: the launcher reads the pointer/master, passes the local target/draft
paths into infer_dflash.py, and writes the configured JSONL output.

- [ ] Step 6: Record exact verification evidence

Update the persistent planning files with test counts, shell/compile results,
stale-reference scan results, and the remaining edge that cannot be verified
locally: actual CUDA DFlash execution on the B200 server.
