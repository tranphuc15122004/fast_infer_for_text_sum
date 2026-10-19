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


def test_select_focused_paged_blocks_vllm():
    from Benchmark.common.vllm_pilot import select_focused_paged_blocks_vllm
    import torch

    # 1. Short context (within budget <= 2048): returns all full blocks
    short_blocks = select_focused_paged_blocks_vllm(total_seq_len=1000, block_size=16, budget_blocks=128)
    assert len(short_blocks) == 1000 // 16
    assert (short_blocks == torch.arange(1000 // 16)).all()

    # 2. Long context with exact multiple (16000 tokens, 1000 blocks)
    long_blocks_exact = select_focused_paged_blocks_vllm(
        total_seq_len=16000,
        block_size=16,
        budget_blocks=128,
        chunk_blocks=8,
        sink_blocks=1,
    )
    assert len(long_blocks_exact) == 128
    assert 0 in long_blocks_exact  # Sink block 0 must be included
    assert 999 in long_blocks_exact  # Recent block must be included

    # 3. Long context with remainder (16005 tokens, 1000 full blocks, r=5)
    long_blocks_rem = select_focused_paged_blocks_vllm(
        total_seq_len=16005,
        block_size=16,
        budget_blocks=128,
        chunk_blocks=8,
        sink_blocks=1,
    )
    assert len(long_blocks_rem) == 127
    assert 0 in long_blocks_rem  # Sink block 0 must be included
    assert 999 in long_blocks_rem  # Recent block must be included

    # 4. Multi-query chunk reps scoring
    num_chunks = (1000 + 7) // 8
    hidden_size = 64
    chunk_reps = torch.randn(num_chunks, hidden_size)
    q = torch.randn(4, hidden_size)  # 4 accepted queries
    scored_blocks = select_focused_paged_blocks_vllm(
        total_seq_len=16000,
        block_size=16,
        budget_blocks=128,
        chunk_blocks=8,
        sink_blocks=1,
        chunk_reps=chunk_reps,
        q=q,
        query_pooling="max",
    )
    assert len(scored_blocks) == 128
    assert 0 in scored_blocks


def test_phase_a_block_table_reordering_and_offset_parity():
    from Benchmark.common.vllm_pilot import select_focused_paged_blocks_vllm
    import torch

    block_size = 16
    num_query_tokens = 17

    # Test both r = 0 and r = 5
    for total_seq_len in (16000, 16005):
        r = total_seq_len % block_size
        total_blocks = (total_seq_len + num_query_tokens + block_size - 1) // block_size
        orig_row = torch.arange(100, 100 + total_blocks)  # unique physical block IDs

        focused_logical = select_focused_paged_blocks_vllm(
            total_seq_len=total_seq_len,
            block_size=block_size,
            budget_blocks=128,
            chunk_blocks=8,
            sink_blocks=1,
        )
        M = len(focused_logical)
        focused_phys = orig_row[focused_logical]

        q_start_block = total_seq_len // block_size
        q_end_block = (total_seq_len + num_query_tokens - 1) // block_size

        new_row = orig_row.clone()
        new_row[:M] = focused_phys
        new_row[M] = orig_row[q_start_block]
        if q_end_block > q_start_block:
            new_row[M + 1] = orig_row[q_end_block]

        focused_seq_len = M * block_size + r + num_query_tokens

        # Verify that total attended tokens is <= 2065 (8x smaller than 16017)
        assert focused_seq_len <= 128 * 16 + 17

        # Verify exact offset parity for query tokens:
        # Query tokens are at offsets 0..16
        for query_off in range(num_query_tokens):
            orig_token_pos = total_seq_len + query_off
            orig_phys_block = orig_row[orig_token_pos // block_size].item()
            orig_token_offset = orig_token_pos % block_size

            # In the reordered block table, query token starts after focused context
            new_token_pos = M * block_size + r + query_off
            new_phys_block = new_row[new_token_pos // block_size].item()
            new_token_offset = new_token_pos % block_size

            # Physical block and token offset MUST MATCH EXACTLY!
            assert new_phys_block == orig_phys_block, f"Mismatch at query_off={query_off}: {new_phys_block} != {orig_phys_block}"
            assert new_token_offset == orig_token_offset, f"Mismatch at query_off={query_off}: {new_token_offset} != {orig_token_offset}"



