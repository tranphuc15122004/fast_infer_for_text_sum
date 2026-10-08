"""Exercise AMR-DFlash launcher routing against temporary offline assets."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config


ROOT = Path(__file__).resolve().parents[1]


def _write_qwen_snapshot(path: Path, *, layers: int, draft: bool) -> None:
    path.mkdir(parents=True)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    if draft:
        config.block_size = 16
        config.dflash_config = {"target_layer_ids": [0, 1], "mask_token_id": 0}
        config.num_target_layers = 2
    config.to_json_file(path / "config.json")
    (path / "model.safetensors").write_bytes(b"preflight-only")
    if not draft:
        tokenizer = Tokenizer(
            WordLevel(
                vocab={"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "hello": 3},
                unk_token="[UNK]",
            )
        )
        tokenizer.pre_tokenizer = Whitespace()
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            eos_token="[EOS]",
        )
        fast_tokenizer.save_pretrained(path)


def test_run_sh_routes_amr_and_loads_master_model_paths_offline(tmp_path: Path):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    _write_qwen_snapshot(target, layers=4, draft=False)
    _write_qwen_snapshot(draft, layers=5, draft=True)
    master = tmp_path / "master.env"
    master.write_text(
        f"MODEL_TARGET='{target}'\nMODEL_DFLASH_DRAFT='{draft}'\nFI_DEVICE=cpu\nFI_OFFLINE=1\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "FAST_INFER_MASTER_CONFIG": str(master),
            "FAST_INFER_PYTHON": str(ROOT / ".venv" / "bin" / "python"),
            "AMR_DEVICE": "cpu",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run.sh"), "amr_dflash", "preflight"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout[result.stdout.index("{") :])
    assert report["target_model"] == str(target.resolve())
    assert report["draft_model"] == str(draft.resolve())
    assert report["draft_layers"] == 5
    assert report["block_size"] == 16
    assert report["tokenizer_vocab_size"] == 4
    assert report["tokenizer_eos_token_id"] == 2
    assert report["local_files_only"] is True
