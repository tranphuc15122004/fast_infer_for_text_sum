"""End-to-end smoke với snapshot Qwen3-4B thật đã có trong HF cache.

Test này không chạy trong suite mặc định vì cần nhiều phút và khoảng 8 GB
model weights. Bật tường minh bằng ``MR_DFLASH_RUN_REAL_QWEN3_4B=1``.
Mặc định test chạy CPU; có thể truyền snapshot khác qua
``MR_DFLASH_QWEN3_4B_PATH``.
"""

from __future__ import annotations

import os
import json
import math
from pathlib import Path

import pytest
import torch

from MR_DFlash.checkpoint import warm_start_draft_model
from MR_DFlash.config import DataConfig, ModelConfig, RunConfig, TrainingConfig
from MR_DFlash.data import DFlashFeatureDataset, save_feature_file
from MR_DFlash.inference import MRDFlashInferenceEngine
from MR_DFlash.mr_model import MRDFlashDraftModel
from MR_DFlash.run_train import build_online_model
from MR_DFlash.trainer import Trainer
from MR_DFlash.training import (
    MRDFlashTrainStrategy,
    OnlineMRDFlashModel,
    build_mr_draft_spec_from_target_config,
)


def _qwen3_4b_snapshot() -> Path | None:
    configured = os.environ.get("MR_DFLASH_QWEN3_4B_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        return path if (path / "config.json").exists() else None

    cache_root = Path(
        os.environ.get(
            "HF_HOME",
            str(Path.home() / ".cache" / "huggingface"),
        )
    ) / "hub" / "models--Qwen--Qwen3-4B"
    ref = cache_root / "refs" / "main"
    if not ref.exists():
        return None
    commit = ref.read_text(encoding="utf-8").strip()
    if not commit:
        return None
    snapshot = (cache_root / "snapshots" / commit).resolve()
    return snapshot if (snapshot / "config.json").exists() else None


def test_real_qwen3_4b_mr_dflash_end_to_end_cpu(tmp_path: Path) -> None:
    """Capture thật, train thật, reload checkpoint thật và inference thật."""
    if os.environ.get("MR_DFLASH_RUN_REAL_QWEN3_4B") != "1":
        pytest.skip("set MR_DFLASH_RUN_REAL_QWEN3_4B=1 để chạy model Qwen3-4B thật")
    model_path = _qwen3_4b_snapshot()
    if model_path is None:
        pytest.skip("không tìm thấy snapshot Qwen3-4B local; không tải model qua mạng")

    from transformers import AutoModelForCausalLM

    torch.set_num_threads(int(os.environ.get("MR_DFLASH_CPU_THREADS", "8")))
    torch.manual_seed(123)
    device = torch.device("cpu")
    dtype = torch.bfloat16

    target = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device=device).eval()
    target_config = target.config
    vocab_size = int(target_config.vocab_size)
    mask_token_id = 151669 if vocab_size > 151669 else vocab_size - 1
    seq_len = 32
    input_ids = torch.arange(100, 100 + seq_len, dtype=torch.long).view(1, -1)
    loss_mask = torch.ones((1, seq_len), dtype=dtype)

    # Năm layer target trải đều là feature contract tương ứng config MR-DFlash.
    layer_ids = [1, 9, 17, 25, 33]
    assert max(layer_ids) < int(target_config.num_hidden_layers)
    with torch.no_grad():
        target_outputs = target(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden_states = torch.cat(
        [target_outputs.hidden_states[layer_id + 1] for layer_id in layer_ids],
        dim=-1,
    )
    assert hidden_states.shape == (1, seq_len, len(layer_ids) * int(target_config.hidden_size))
    assert torch.isfinite(hidden_states.float()).all()

    feature_dir = tmp_path / "features"
    save_feature_file(
        str(feature_dir / "qwen3_4b_sample.ckpt"),
        {
            "input_ids": input_ids[0],
            "loss_mask": loss_mask[0],
            "hidden_states": hidden_states[0],
        },
    )

    spec = build_mr_draft_spec_from_target_config(
        target_config,
        draft_num_hidden_layers=1,
        block_size=4,
        target_layer_ids=layer_ids,
        num_stages=2,
        hca_compression_ratio=128,
        csa_compression_ratio=4,
        local_window=128,
        csa_top_k=64,
    )
    spec.mask_token_id = mask_token_id
    draft = MRDFlashDraftModel(spec).to(device=device, dtype=dtype)
    copied = draft.init_from_target(target)
    assert copied, "init_draft_from_target phải copy được layer Qwen3"

    model = OnlineMRDFlashModel(
        draft,
        target_lm_head=target.get_output_embeddings(),
        target_embed_tokens=target.get_input_embeddings(),
        mask_token_id=mask_token_id,
        block_size=4,
        num_anchors=4,
        loss_decay_gamma=7.0,
        objective_chunk_blocks=0,
        loss_type="dflash",
        attention_backend="sdpa",
    )
    run_cfg = RunConfig(
        run_id="real-qwen3-4b-mr-dflash-smoke",
        output_dir=str(tmp_path / "train_out"),
        model=ModelConfig(
            target_model_path=str(model_path),
            architecture="mr_dflash",
            draft_num_hidden_layers=1,
            block_size=4,
            mask_token_id=mask_token_id,
            target_layer_ids=layer_ids,
            torch_dtype="bfloat16",
            mr_num_stages=2,
            hca_compression_ratio=128,
            csa_compression_ratio=4,
            memory_local_window=128,
            csa_top_k=64,
        ),
        data=DataConfig(
            hidden_states_path=str(feature_dir),
            max_length=seq_len,
        ),
        training=TrainingConfig(
            strategy="mr_dflash",
            num_epochs=1,
            max_steps=1,
            batch_size=1,
            accumulation_steps=1,
            learning_rate=6e-4,
            warmup_ratio=0.0,
            max_grad_norm=1.0,
            num_anchors=4,
            loss_decay_gamma=7.0,
            objective_chunk_blocks=0,
            attention_backend="sdpa",
            loss_type="dflash",
            save_interval=1,
            log_interval=1,
        ),
    )

    # Kiểm tra đường build từ config tương tự run_train, rồi train model đã
    # init từ target; target embeddings/lm_head vẫn frozen trong optimizer.
    class _Tokenizer:
        def convert_tokens_to_ids(self, token: str) -> int:
            return mask_token_id if token == "[MASK]" else -1

    configured_model = build_online_model(
        run_cfg,
        tokenizer=_Tokenizer(),
        target_config=target_config,
        embed_tokens=target.get_input_embeddings(),
        lm_head=target.get_output_embeddings(),
        device=device,
    )
    assert isinstance(configured_model, OnlineMRDFlashModel)

    dataset = DFlashFeatureDataset(
        str(feature_dir),
        max_len=seq_len,
        run_id="real-qwen3-4b-smoke",
    )
    summary = Trainer(
        run_cfg,
        MRDFlashTrainStrategy(model),
        dataset,
        device=device,
    ).fit()
    assert summary["global_step"] == 1
    assert torch.isfinite(draft.memory.adapter.hca.weight.float()).all()
    metrics_path = tmp_path / "train_out" / "metrics.jsonl"
    last_metrics = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[-1])
    assert math.isfinite(float(last_metrics["loss"]))

    checkpoint = tmp_path / "train_out" / "checkpoint_final.pt"
    draft_weights = tmp_path / "train_out" / "draft_final.pt"
    assert checkpoint.exists()
    assert draft_weights.exists()

    # Reload weights-only vào instance mới để chứng minh seam train → serving.
    restored = MRDFlashDraftModel(spec).to(device=device, dtype=dtype)
    missing, unexpected = warm_start_draft_model(
        restored,
        str(draft_weights),
        strategy_name="mr_dflash",
    )
    assert missing == []
    assert unexpected == []

    engine = MRDFlashInferenceEngine(
        target,
        restored,
        mask_token_id=mask_token_id,
        device=device,
    )
    prefix = input_ids[:, :8]
    prefill = engine.prefill(prefix)
    assert prefill.memory.total_tokens == prefix.shape[1]
    draft_output = engine.draft_block(prefix, prefill.memory)
    assert draft_output.proposed_ids.shape == (1, 3)
    assert torch.isfinite(draft_output.logits.float()).all()
    verified = engine.verify(prefix, draft_output.proposed_ids, prefill.memory)
    assert verified.accepted_ids.shape[1] >= 1
    assert torch.isfinite(verified.target_logits.float()).all()
    generated = engine.generate(prefix, max_new_tokens=2)
    assert generated.input_ids.shape == (1, prefix.shape[1] + 2)
    assert generated.memory.total_tokens == generated.input_ids.shape[1]
    assert generated.accepted_proposal_tokens >= 0
    assert int(generated.input_ids.min()) >= 0
    assert int(generated.input_ids.max()) < vocab_size

    print(
        "REAL_QWEN3_4B_MR_DFLASH_SMOKE_PASS "
        f"model={model_path} train_steps={summary['global_step']} "
        f"generated_tokens={generated.input_ids[0, -2:].tolist()} "
        f"accepted_proposal_tokens={generated.accepted_proposal_tokens}"
    )
