import pytest
from Benchmark.vllm_all_baselines import METHODS, DRAFT_ARGUMENTS, build_speculative_config
from Benchmark.common.vllm_pilot import select_pilot_methods, install_phase_a_vllm_compat


def test_vllm_methods_includes_dflash_phase_a():
    assert "dflash_phase_a" in METHODS
    assert DRAFT_ARGUMENTS.get("dflash_phase_a") == "dflash_model"


def test_select_pilot_methods_allows_dflash_phase_a():
    selected = select_pilot_methods(
        "vanilla_vllm,dflash,dflash_phase_a",
        available_methods=METHODS,
        reference="vanilla_vllm",
    )
    assert "dflash_phase_a" in selected


def test_build_speculative_config_for_dflash_phase_a():
    fake_config = {"block_size": 16}
    spec, meta = build_speculative_config(
        "dflash_phase_a",
        "/path/to/dflash",
        fake_config,
        target_num_hidden_layers=36,
    )
    assert spec is not None
    assert spec["method"] == "dflash"
    assert spec["num_speculative_tokens"] == 16
    assert meta["vllm_method"] == "dflash"


def test_install_phase_a_vllm_compat_idempotent():
    # Should run without error and be idempotent
    try:
        install_phase_a_vllm_compat()
        install_phase_a_vllm_compat()
    except RuntimeError as e:
        if "NVIDIA driver" in str(e) or "driver is too old" in str(e):
            pytest.skip(f"Skipping due to host CUDA driver mismatch: {e}")
        raise


def test_phase_a_gather_and_sdpa_simulation():
    import torch
    import torch.nn.functional as F

    num_blocks = 200
    block_size = 16
    num_kv_heads = 2
    num_heads = 16
    head_dim = 128

    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
    value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)

    total_tokens = 3000
    num_prompt_blocks = (total_tokens + block_size - 1) // block_size
    block_table = torch.arange(5, 5 + num_prompt_blocks)

    focused_indices = torch.sort(torch.randperm(total_tokens)[:2048]).values

    phys_blocks = block_table[focused_indices // block_size]
    offsets = focused_indices % block_size

    k_focused = key_cache[phys_blocks, offsets]
    v_focused = value_cache[phys_blocks, offsets]
    assert k_focused.shape == (2048, num_kv_heads, head_dim)

    # Query tokens (17 tokens)
    q = torch.randn(1, num_heads, 17, head_dim)
    k_q = torch.randn(17, num_kv_heads, head_dim)
    v_q = torch.randn(17, num_kv_heads, head_dim)

    k_all = torch.cat([k_focused, k_q], dim=0).unsqueeze(0)
    v_all = torch.cat([v_focused, v_q], dim=0).unsqueeze(0)

    head_rep = num_heads // num_kv_heads
    k_in = k_all.repeat_interleave(head_rep, dim=2).transpose(1, 2)
    v_in = v_all.repeat_interleave(head_rep, dim=2).transpose(1, 2)

    attn_output = F.scaled_dot_product_attention(q, k_in, v_in)
    assert attn_output.shape == (1, num_heads, 17, head_dim)
    attn_output = attn_output.transpose(1, 2).reshape(17, num_heads * head_dim)
    assert attn_output.shape == (17, 2048)


