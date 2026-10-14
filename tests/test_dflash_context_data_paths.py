from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_dflash_context_experiments as campaign

DATA_VARIABLES = ("DATA_INPUT", "DATA_FILE", "LONG_BENCH_DATA_DIR", "LONG_BENCH_DATA_FILE")


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    for name in DATA_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    directory = tmp_path / "corpus"
    directory.mkdir()
    for dataset in ("gov_report", "multi_news", "qmsum"):
        (directory / f"{dataset}.jsonl").write_text(json.dumps({
            "id": f"{dataset}-1", "context": "test context", "input": "summarize",
            "answers": ["test answer"],
        }) + "\n")
    return directory


@pytest.mark.parametrize("kind", ("directory", "file"))
def test_paths_collapse_canonical_input_and_legacy_alias(corpus, monkeypatch, kind):
    source = corpus if kind == "directory" else corpus / "gov_report.jsonl"
    monkeypatch.setenv("DATA_INPUT", str(source))
    monkeypatch.setenv("DATA_FILE", str(source / ".") if kind == "directory" else str(source))
    assert campaign._paths(campaign._parser().parse_args([])) == (
        (source, None) if kind == "directory" else (None, source)
    )


@pytest.mark.parametrize("option", ("--data-dir", "--data-file"))
def test_explicit_data_option_overrides_environment(corpus, monkeypatch, option):
    monkeypatch.setenv("DATA_INPUT", "/stale/input")
    monkeypatch.setenv("DATA_FILE", "/another/stale/input")
    source = corpus if option == "--data-dir" else corpus / "gov_report.jsonl"
    assert campaign._paths(campaign._parser().parse_args([option, str(source)])) == (
        (source, None) if option == "--data-dir" else (None, source)
    )


def test_explicit_directory_and_file_remain_ambiguous(corpus):
    args = campaign._parser().parse_args([
        "--data-dir", str(corpus), "--data-file", str(corpus / "gov_report.jsonl"),
    ])
    with pytest.raises(ValueError, match="--data-dir.*--data-file"):
        campaign._paths(args)


def test_different_canonical_data_sources_are_rejected_with_paths(corpus, monkeypatch):
    monkeypatch.setenv("DATA_INPUT", str(corpus))
    monkeypatch.setenv("DATA_FILE", str(corpus / "gov_report.jsonl"))
    with pytest.raises(ValueError, match="DATA_INPUT.*DATA_FILE") as error:
        campaign._paths(campaign._parser().parse_args([]))
    assert str(corpus) in str(error.value)


@pytest.mark.parametrize("variable", DATA_VARIABLES)
@pytest.mark.parametrize("kind", ("directory", "file"))
def test_environment_aliases_recognize_actual_source_type(corpus, monkeypatch, variable, kind):
    source = corpus if kind == "directory" else corpus / "gov_report.jsonl"
    monkeypatch.setenv(variable, str(source))
    assert campaign._paths(campaign._parser().parse_args([])) == (
        (source, None) if kind == "directory" else (None, source)
    )


def test_canonical_data_ignores_longbench_defaults(corpus, monkeypatch):
    monkeypatch.setenv("DATA_INPUT", str(corpus))
    monkeypatch.setenv("LONG_BENCH_DATA_DIR", "/unrelated/default/directory")
    monkeypatch.setenv("LONG_BENCH_DATA_FILE", "/unrelated/default/file.jsonl")
    samples, paths = campaign._load_samples(campaign._parser().parse_args([]))
    assert len(samples) == 3
    assert paths == sorted(corpus.glob("*.jsonl"))


def test_real_phase_launcher_loads_directory_master_without_false_conflict(corpus, tmp_path):
    master = tmp_path / "master.env"
    master.write_text(f"DATA_INPUT={shlex.quote(str(corpus))}\n")
    # Use the real shell launcher/config/runtime. Replace only GPU entrypoint
    # execution with a Python probe of the real parser and corpus loader.
    probe = tmp_path / "python-probe"
    probe.write_text(f"#!{sys.executable}\n" + '''
import json, os, subprocess, sys
if sys.argv[1:2] == ["-c"]:
    raise SystemExit(subprocess.call([sys.executable, *sys.argv[1:]]))
sys.path.insert(0, os.path.dirname(sys.argv[1]))
import run_dflash_context_experiments as campaign
args = campaign._parser().parse_args(sys.argv[2:])
samples, paths = campaign._load_samples(args)
print(json.dumps({"rows": len(samples), "paths": [str(p) for p in paths],
                  "legacy_alias": os.environ.get("DATA_FILE")}))
''')
    probe.chmod(0o755)
    environment = dict(os.environ)
    for name in (*DATA_VARIABLES, "FAST_INFER_MASTER_LOADED"):
        environment.pop(name, None)
    environment.update(FAST_INFER_MASTER_CONFIG=str(master), FAST_INFER_PYTHON=str(probe))
    result = subprocess.run([
        "bash", str(ROOT / "scripts/run_dflash_context_experiments.sh"), "--mode", "prepare",
    ], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["legacy_alias"] == str(corpus)
    assert payload["rows"] == 3
    assert payload["paths"] == [str(p) for p in sorted(corpus.glob("*.jsonl"))]


def test_campaign_routes_every_phase_to_audited_corpus_despite_xsum_master(tmp_path):
    replica = tmp_path / "repo"
    (replica / "scripts/common").mkdir(parents=True)
    for name in (
        "scripts/run_dflash_context_A_b200.sh", "scripts/run_dflash_context_experiments.sh",
        "scripts/common/config.sh", "scripts/common/runtime.sh",
    ):
        shutil.copy2(ROOT / name, replica / name)
    (replica / "data").symlink_to(ROOT / "data", target_is_directory=True)
    selected = ROOT / "data/longbench_100_14k"
    xsum = tmp_path / "xsum.jsonl"
    xsum.write_text(json.dumps({"id": "xsum-1", "context": "wrong corpus"}) + "\n")
    master = tmp_path / "master.env"
    master.write_text(
        f"DATA_INPUT={shlex.quote(str(xsum))}\n"
        f"DCE_DATA_DIR={shlex.quote(str(selected))}\n"
    )
    capture = tmp_path / "phase_paths.jsonl"
    probe = tmp_path / "python-probe"
    probe.write_text(f"#!{sys.executable}\n" + f'''
import json, os, sys
from pathlib import Path
if sys.argv[1:2] in (["-c"], ["-"]):
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
sys.path.insert(0, {str(ROOT / "scripts")!r})
import run_dflash_context_experiments as campaign
sys.argv.extend(["--output-root", {str(replica / "outputs/dflash_context_interventions")!r}])
args = campaign._parser().parse_args(sys.argv[2:])
directory, file = campaign._paths(args)
sys.argv = [sys.argv[1], *sys.argv[2:]]
with Path({str(capture)!r}).open("a") as handle:
    handle.write(json.dumps({{"mode": args.mode, "directory": str(directory), "file": str(file)}}) + "\\n")
if args.mode == "preflight":
    # Only GPU/package checks are substituted; corpus loading is real.
    from TrainingFree.context_interventions import environment
    def cpu_preflight(**kwargs):
        samples, _ = campaign._load_samples(args)
        return {{"status": "ready", "data": {{"record_count": len(samples)}}}}
    environment.run_preflight = cpu_preflight
    raise SystemExit(campaign.main())
if args.mode == "prepare":
    raise SystemExit(campaign.main())
# GPU phases and gates are substituted: this test checks data routing only.
''')
    probe.chmod(0o755)
    environment = dict(os.environ)
    for name in (*DATA_VARIABLES, "FAST_INFER_MASTER_LOADED", "DCE_DATA_DIR", "DCE_RESUME"):
        environment.pop(name, None)
    environment.update(FAST_INFER_MASTER_CONFIG=str(master), FAST_INFER_PYTHON=str(probe))
    result = subprocess.run([
        "bash", str(replica / "scripts/run_dflash_context_A_b200.sh"), "_campaign", "test-routing",
    ], env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    observed = [json.loads(line) for line in capture.read_text().splitlines()]
    assert [row["mode"] for row in observed] == [
        "preflight", "prepare", "validate", "pilot", "dev", "lock", "confirm", "report",
    ]
    assert all(row["directory"] == str(selected) and row["file"] == "None" for row in observed)
    assert "campaign complete" in result.stdout
