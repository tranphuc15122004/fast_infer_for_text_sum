"""Regression coverage using real local Qwen/DFlash weights, without downloads."""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts", ROOT / "externals/dflash"):
    sys.path.insert(0, str(path))

from AMR_DFlash import pipeline
from AMR_DFlash.artifacts import load_torch, read_jsonl, sha256_file, write_jsonl
from AMR_DFlash.checkpoint import load_memory_checkpoint, save_memory_checkpoint, snapshot_fingerprint
from AMR_DFlash.evaluation import evaluate_one_fixed_state
from AMR_DFlash.inference import AMRDFlashEngine
from AMR_DFlash.runtime import load_runtime
from amr_dflash.cli import _parser, _run_inference

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from dflash.model import DFlashDraftModel

    root = tmp_path_factory.mktemp("amr-assets")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(29)
    params = dict(vocab_size=8, hidden_size=16, intermediate_size=32,
                  num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=128, eos_token_id=None, pad_token_id=0)
    Qwen3ForCausalLM(Qwen3Config(**params, num_hidden_layers=4)).save_pretrained(root / "target")
    draft_config = Qwen3Config(**params, num_hidden_layers=5)
    draft_config.block_size = 16
    draft_config.dflash_config = {"target_layer_ids": [0, 1, 2], "mask_token_id": 1}
    draft_config.num_target_layers = 3
    DFlashDraftModel(draft_config).save_pretrained(root / "draft")
    tok = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "a": 2, "b": 3,
                              "c": 4, "d": 5, "e": 6, "f": 7}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(root / "target")
    config = yaml.safe_load((ROOT / "src/AMR_DFlash/configs/pilot.yaml").read_text())
    config["model"]["dtype"] = "float32"
    config["memory"].update(raw_budget=4, num_slots=2, local_window=1, index_dim=4, min_context_tokens=0)
    config["training"].update(selector_steps=2, compressor_steps=2, max_candidates=6)
    saved = {key: os.environ.get(key) for key in ("TARGET_MODEL", "DRAFT_MODEL")}
    os.environ.update(TARGET_MODEL=str(root / "target"), DRAFT_MODEL=str(root / "draft"))
    yield config, root
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    torch.set_num_threads(threads)


def capture(config, root, source, **overrides):
    kwargs = dict(input_path=source, run_root=root, max_samples=None, max_new_tokens=32,
                  max_input_tokens=0, max_states_per_document=3, device=CPU)
    kwargs.update(overrides)
    return pipeline.capture_run(config, **kwargs)


@pytest.fixture
def captured(assets, tmp_path):
    config, _ = assets
    source = tmp_path / "input.jsonl"
    write_jsonl(source, [{"id": "x" * 120, "prompt": "a b c d e f " * 2, "split": "train"},
                         {"id": "validation", "prompt": "a b c d e f " * 3, "split": "validation"},
                         {"id": "holdout", "prompt": "a b c d e f " * 4, "split": "holdout"}])
    root = tmp_path / "run"
    capture(config, root, source)
    return config, root, source


def test_prefill_projects_only_one_logit_row_but_preserves_all_features(assets):
    from AMR_DFlash.model import extract_target_features

    config, _ = assets
    target, tokenizer, draft, memory, _ = load_runtime(config, device=CPU)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7] * 2])
    with torch.no_grad():
        expected = target(input_ids=ids, output_hidden_states=True, return_dict=True)
        expected_features = draft.project_context(extract_target_features(expected, draft.target_layer_ids))
    rows = []
    handle = target.get_output_embeddings().register_forward_pre_hook(lambda _, args: rows.append(args[0].shape[1]))
    try:
        engine = AMRDFlashEngine(target, tokenizer, draft, memory, device=CPU, mode="dense")
        result = engine.generate(ids, max_new_tokens=1, eos_token_ids=None, capture_states=True)
        assert rows == [1]
        assert result.output_ids[0, -1] == expected.logits[0, -1].argmax()
        assert torch.allclose(result.projected_context, expected_features)
        rows.clear()
        evaluate_one_fixed_state(target, draft, memory, context_ids=ids,
                                 projected_context=expected_features,
                                 anchor_id=int(result.output_ids[0, -1]), remaining_output_budget=32,
                                 mode="dense", eos_token_ids=set())
        assert rows == [1, 16, 16]  # prefix, draft head, full verifier
    finally:
        handle.remove()


def test_regenerated_multiturn_manifest_reaches_real_capture_and_verifier_labels(assets, tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    from AMR_DFlash.training_data import prepare_manifest

    config, asset_root = assets
    target_path = tmp_path / "target-with-chat"
    shutil.copytree(asset_root / "target", target_path)
    tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}"
        "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    )
    tokenizer.save_pretrained(target_path)
    monkeypatch.setenv("TARGET_MODEL", str(target_path))
    inputs = {}
    for split, context in (("train", "a b c"), ("validation", "d e f")):
        inputs[split] = [write_jsonl(tmp_path / f"{split}.jsonl", [{
            "id": split, "source": "sharegpt",
            "conversations": [{"role": "user", "content": context},
                              {"role": "assistant", "content": "b c d"},
                              {"role": "user", "content": "e f a b c"},
                              {"role": "assistant", "content": "SHOULD NOT ENTER PROMPT"}],
        }])]
    prepared = prepare_manifest(inputs=inputs, output_dir=tmp_path / "prepared",
                                tokenizer_path=target_path, min_input_tokens=5,
                                max_input_tokens=24, progress=False)
    run = tmp_path / "capture"
    capture(config, run, prepared["manifest"], max_input_tokens=24)
    documents = read_jsonl(run / "documents.jsonl")
    assert len(documents) == 2
    for sample, document in zip(pipeline.read_records_for_pipeline(Path(prepared["manifest"]), max_samples=None), documents):
        rendered, ids = pipeline._tokenize_prompt(tokenizer, sample, max_input_tokens=24)
        bundle = load_torch(run / document["bundle"])
        assert "SHOULD NOT ENTER PROMPT" not in rendered
        assert "assistant: b c d" in rendered
        assert torch.equal(bundle["trajectory_ids"][:ids.shape[1]], ids[0])
        assert document["source_content_sha256"] == pipeline._source_content_hash(sample["raw"], rendered)
    pipeline.generate_candidates_for_run(config, run_root=run)
    report = pipeline.label_candidates(config, run_root=run, device=CPU)
    assert report["training_signal"]["by_split"]["train"]["uncensored_teacher_rows"] > 0
    assert (run / "training_signal.json").is_file()


@pytest.mark.parametrize("field,value", [("draft_fingerprint", {"sha256": "changed"}),
                                         ("dtype", "bfloat16"), ("attention_backend", "eager")])
def test_feature_bank_rejects_changed_projection_or_execution_contract(captured, field, value):
    config, root, _ = captured
    _, _, draft, _, metadata = load_runtime(config, device=CPU)
    bundle = load_torch(root / read_jsonl(root / "states.jsonl")[0]["bundle"])
    changed = {**metadata, field: value}
    with pytest.raises(ValueError, match="feature|capture"):
        pipeline._validate_feature_bundle(bundle, changed, draft)


@pytest.mark.parametrize("records", [
    [{"id": "duplicate", "prompt": "a b", "split": "train"},
     {"id": "duplicate", "prompt": "a b", "split": "holdout"}],
    [{"id": "first", "prompt": "a b", "split": "train"},
     {"id": "second", "prompt": "a b", "split": "holdout"}],
    [{"id": "first", "document_id": "source", "prompt": "a b", "split": "train"},
     {"id": "second", "document_id": "source", "prompt": "c d", "split": "holdout"}],
])
def test_capture_rejects_duplicate_ids_or_cross_split_sources_before_loading(assets, tmp_path, monkeypatch, records):
    config, _ = assets
    source = write_jsonl(tmp_path / "bad.jsonl", records)
    monkeypatch.setenv("TARGET_MODEL", str(tmp_path / "absent"))
    with pytest.raises(ValueError, match="duplicate|overlap|split"):
        capture(config, tmp_path / "run", source, max_samples=1)


def test_runtime_seed_controls_initial_weights_even_if_ambient_rng_differs(assets):
    config, _ = assets
    weights = []
    for ambient_seed in (111, 222):
        torch.manual_seed(ambient_seed)
        _, _, _, memory, _ = load_runtime(config, device=CPU)
        weights.append({key: value.clone() for key, value in memory.state_dict().items()})
    assert all(torch.equal(weights[0][key], weights[1][key]) for key in weights[0])


@pytest.mark.parametrize("overrides", [{"max_new_tokens": 4}, {"max_input_tokens": 8},
                                       {"max_states_per_document": 1}])
def test_resume_rejects_changed_caps_without_rewriting_manifest(captured, overrides):
    config, root, source = captured
    before = (root / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="resume|contract"):
        capture(config, root, source, resume=True, **overrides)
    assert (root / "manifest.json").read_bytes() == before


def test_long_state_ids_have_distinct_candidate_files_and_complete_keys(captured):
    config, root, _ = captured
    pipeline.generate_candidates_for_run(config, run_root=root)
    states = read_jsonl(root / "states.jsonl")
    rows = read_jsonl(root / "candidates.jsonl")
    assert len({row["positions_ref"] for row in rows}) == len(states)
    assert all(row["candidate_id"] in load_torch(root / row["positions_ref"]) for row in rows)


def test_checkpoint_accepts_relocated_bytes_but_rejects_changed_weights(assets, tmp_path):
    config, root = assets
    _, _, _, memory, metadata = load_runtime(config, device=CPU)
    checkpoint = save_memory_checkpoint(tmp_path / "memory.pt", memory, metadata=metadata)
    shutil.copytree(root / "target", tmp_path / "target-copy")
    relocated = {**metadata, "target_fingerprint": snapshot_fingerprint(tmp_path / "target-copy")}
    load_memory_checkpoint(checkpoint, memory, expected_metadata=relocated, device=CPU)
    changed = copy.deepcopy(relocated)
    changed["target_fingerprint"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="fingerprint"):
        load_memory_checkpoint(checkpoint, memory, expected_metadata=changed, device=CPU)


def test_rollout_records_identity_provenance_and_decode_tpot(assets, tmp_path):
    config, _ = assets
    source = write_jsonl(tmp_path / "input.jsonl", [{"id": "doc-1", "prompt": "a b c d e f", "split": "holdout"}])
    output = tmp_path / "output.jsonl"
    args = _parser().parse_args(["--device", "cpu", "infer", "--input", str(source),
                                "--output", str(output), "--mode", "dense", "--max-new-tokens", "8"])
    summary = _run_inference(args, config)
    row = read_jsonl(output)[0]
    assert row["sample_id"] == row["document_id"] == "doc-1"
    assert row["split"] == "holdout"
    for field in ("prompt_hash", "run_config_hash", "input_manifest_sha256", "memory_checkpoint_sha256"):
        assert field in row
    assert summary["run_config_hash"] == row["run_config_hash"]
    assert row["tpot_ms"] == pytest.approx(row["decode_time_ms"] / (row["output_tokens"] - 1), abs=0.002)


def test_ttft_includes_feature_projection_and_memory_updates_are_timed(assets, monkeypatch):
    config, _ = assets
    target, tokenizer, draft, memory, _ = load_runtime(config, device=CPU)
    project = draft.project_context
    update = memory.compressor.update
    def slow_project(*args, **kwargs):
        time.sleep(0.025)
        return project(*args, **kwargs)
    def slow_update(*args, **kwargs):
        time.sleep(0.025)
        return update(*args, **kwargs)
    monkeypatch.setattr(draft, "project_context", slow_project)
    monkeypatch.setattr(memory.compressor, "update", slow_update)
    engine = AMRDFlashEngine(target, tokenizer, draft, memory, device=CPU, mode="amr", use_cost_gate=False)
    result = engine.generate(torch.tensor([[2, 3, 4, 5, 6, 7]]), max_new_tokens=3, eos_token_ids=None)
    assert result.ttft_ms - result.prefill_ms >= 20
    assert result.memory_update_ms >= 20
    assert result.selector_ms >= result.memory_update_ms
    assert result.decode_ms == pytest.approx(result.e2e_ms - result.ttft_ms)


@pytest.mark.parametrize("setting,value", [("seed", 18), ("dtype", "bfloat16"),
                                          ("train_fraction", 0.7)])
def test_resume_rejects_changed_seed_precision_and_split_policy(captured, setting, value):
    config, root, source = captured
    changed = copy.deepcopy(config)
    section = "model" if setting == "dtype" else "data" if setting == "train_fraction" else "training"
    changed[section][setting] = value
    with pytest.raises(ValueError, match="resume.*contract"):
        capture(changed, root, source, resume=True)


def test_resume_can_extend_sample_limit_and_move_identical_snapshots(captured, tmp_path, monkeypatch):
    config, _, source = captured
    root = tmp_path / "extend"
    capture(config, root, source, max_samples=1)
    original = (root / "manifest.json").read_bytes()
    for env in ("TARGET_MODEL", "DRAFT_MODEL"):
        destination = tmp_path / env.lower()
        shutil.copytree(os.environ[env], destination)
        monkeypatch.setenv(env, str(destination))
    result = capture(config, root, source, max_samples=3, resume=True)
    assert result["new_documents"] == 2
    assert (root / "manifest.json").read_bytes() == original
    assert len({row["state_id"] for row in read_jsonl(root / "states.jsonl")}) == result["states"]


@pytest.mark.parametrize("corruption", ["duplicate_state", "mismatched_split", "legacy_feature"])
def test_downstream_rejects_duplicate_states_mislabeled_splits_and_legacy_features(captured, corruption):
    config, root, _ = captured
    states = read_jsonl(root / "states.jsonl")
    if corruption == "duplicate_state":
        write_jsonl(root / "states.jsonl", [*states, states[0]])
    elif corruption == "mismatched_split":
        states[0]["split"] = "holdout"
        write_jsonl(root / "states.jsonl", states)
    else:
        bundle_path = root / states[0]["bundle"]
        bundle = load_torch(bundle_path)
        bundle.pop("feature_contract")
        torch.save(bundle, bundle_path)
    with pytest.raises(ValueError, match="duplicate|split|feature"):
        pipeline.evaluate_fixed_run(config, run_root=root, split="all", mode="dense", checkpoint_path=None,
                                    output_path=None, max_states=1, overwrite=False, device=CPU)


def make_training_fixture(config, root):
    """Real verifier teachers; controlled preferences exercise nonzero selector gradients."""
    pipeline.generate_candidates_for_run(config, run_root=root)
    pipeline.label_candidates(config, run_root=root, device=CPU)
    labels = read_jsonl(root / "candidate_labels.jsonl")
    state = next(row["state_id"] for row in labels if row["split"] == "train" and not row["censored"])
    selected = [row for row in labels if row["state_id"] == state][:2]
    pair = {"record_type": "preference", "state_id": state,
            "document_id": selected[0]["document_id"], "split": "train", "weight": 1.0,
            "capture_contract_sha256": selected[0]["capture_contract_sha256"],
            "candidate_labels_sha256": sha256_file(root / "candidate_labels.jsonl"),
            "positive_candidate_id": selected[0]["candidate_id"],
            "negative_candidate_id": selected[1]["candidate_id"], "test_fixture": True}
    write_jsonl(root / "preferences.jsonl", [pair])
    return labels


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_real_capture_label_train_and_all_modes_pipeline(assets, tmp_path, dtype):
    config = copy.deepcopy(assets[0])
    config["model"]["dtype"] = dtype
    source = write_jsonl(tmp_path / "input.jsonl", [
        {"id": "x" * 120, "prompt": "a b c d e f " * 2, "split": "train"},
        {"id": "valid", "prompt": "a b c d e f " * 3, "split": "validation"},
        {"id": "test", "prompt": "a b c d e f " * 4, "split": "holdout"},
    ])
    root = tmp_path / "run"
    capture(config, root, source)
    labels = make_training_fixture(config, root)
    teachers = [row for row in labels if row.get("teacher_logits_ref")]
    assert len({row["teacher_logits_ref"] for row in teachers}) == len(teachers)
    snapshots = []
    for ambient_seed in (111, 222):
        torch.manual_seed(ambient_seed)
        checkpoint = tmp_path / f"selector-{ambient_seed}.pt"
        result = pipeline.train_selector(config, run_root=root, checkpoint_in=None,
                                         checkpoint_out=checkpoint, steps=2, device=CPU)
        assert result["steps"] == 2
        snapshots.append(load_torch(checkpoint)["memory_state_dict"])
    assert all(torch.equal(snapshots[0][key], snapshots[1][key]) for key in snapshots[0])
    checkpoint = tmp_path / "compressor.pt"
    result = pipeline.train_compressor(config, run_root=root, checkpoint_in=tmp_path / "selector-111.pt",
                                       checkpoint_out=checkpoint, steps=2, device=CPU)
    assert result["steps"] == 2 and result["final_loss"] >= 0
    baseline = None
    workload_hash = None
    run_hashes = set()
    for mode in ("dense", "selection", "compressor", "amr"):
        args = _parser().parse_args(["--device", "cpu", "infer", "--input", str(source),
                                    "--output", str(tmp_path / f"{mode}.jsonl"), "--mode", mode,
                                    "--checkpoint", str(checkpoint), "--split", "holdout",
                                    "--max-new-tokens", "8", "--disable-cost-gate"])
        summary = _run_inference(args, config)
        row = read_jsonl(tmp_path / f"{mode}.jsonl")[0]
        assert row["memory_checkpoint_sha256"] == sha256_file(checkpoint)
        assert row["document_id"] == "test"
        baseline = row["generated_token_ids"] if baseline is None else baseline
        workload_hash = row["workload_hash"] if workload_hash is None else workload_hash
        assert row["generated_token_ids"] == baseline
        assert row["workload_hash"] == workload_hash
        expected_rate = round(row["decode_committed_tokens"] * 1000 / row["decode_time_ms"], 3)
        assert summary["decode_committed_tok_s"] == expected_rate
        run_hashes.add(row["run_config_hash"])
        fixed = pipeline.evaluate_fixed_run(config, run_root=root, split="validation", mode=mode,
                                             checkpoint_path=checkpoint, output_path=tmp_path / f"fixed-{mode}.jsonl",
                                             max_states=1, overwrite=False, device=CPU)
        assert fixed["states"] == 1
    assert len(run_hashes) == 4


def test_training_rejects_preference_split_forgery(captured, tmp_path):
    config, root, _ = captured
    make_training_fixture(config, root)
    pairs = read_jsonl(root / "preferences.jsonl")
    pairs[0]["split"] = "holdout"
    write_jsonl(root / "preferences.jsonl", pairs)
    with pytest.raises(ValueError, match="split"):
        pipeline.train_selector(config, run_root=root, checkpoint_in=None,
                                 checkpoint_out=tmp_path / "bad.pt", steps=1, device=CPU)
    assert not (tmp_path / "bad.pt").exists()


def test_label_rejects_real_changed_draft_projection(captured, tmp_path, monkeypatch):
    from dflash.model import DFlashDraftModel

    config, root, _ = captured
    copy_path = tmp_path / "changed-draft"
    shutil.copytree(os.environ["DRAFT_MODEL"], copy_path)
    changed = DFlashDraftModel.from_pretrained(copy_path, local_files_only=True)
    with torch.no_grad():
        changed.fc.weight.add_(0.1)
    changed.save_pretrained(copy_path)
    monkeypatch.setenv("DRAFT_MODEL", str(copy_path))
    with pytest.raises(ValueError, match="capture.*contract"):
        pipeline.label_candidates(config, run_root=root, device=CPU)
    assert not (root / "candidate_labels.jsonl").exists()


def test_compressor_rejects_modified_teacher_logits(captured, tmp_path):
    config, root, _ = captured
    labels = make_training_fixture(config, root)
    selector = tmp_path / "selector.pt"
    pipeline.train_selector(config, run_root=root, checkpoint_in=None, checkpoint_out=selector, steps=1, device=CPU)
    for row in labels:
        if row.get("teacher_logits_ref") and row["split"] == "train":
            path = root / row["teacher_logits_ref"]
            teacher = load_torch(path)
            teacher["target_logits"].add_(1)
            torch.save(teacher, path)
    with pytest.raises(ValueError, match="teacher.*fingerprint"):
        pipeline.train_compressor(config, run_root=root, checkpoint_in=selector,
                                  checkpoint_out=tmp_path / "bad.pt", steps=1, device=CPU)
    assert not (tmp_path / "bad.pt").exists()


def test_single_token_tpot_is_unknown(assets, tmp_path):
    config, _ = assets
    source = write_jsonl(tmp_path / "input.jsonl", [{"id": "first", "prompt": "a b", "split": "holdout"}])
    args = _parser().parse_args(["--device", "cpu", "infer", "--input", str(source),
                                "--output", str(tmp_path / "output.jsonl"), "--mode", "dense", "--max-new-tokens", "1"])
    _run_inference(args, config)
    row = read_jsonl(tmp_path / "output.jsonl")[0]
    assert row["tpot_ms"] is None
    assert row["decode_committed_tokens"] == 0
