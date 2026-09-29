import torch

from scripts.dflash_tree_cpu import (
    build_draft_tree,
    build_tree_attention_mask,
    walk_verified_tree,
)


def test_best_first_tree_contains_parent_closed_prefixes_under_budget():
    token_ids = [[10, 11], [20, 21], [30, 31]]
    log_probs = [
        [-0.1, -1.0],
        [-0.2, -0.8],
        [-0.3, -0.7],
    ]

    tree = build_draft_tree(token_ids, log_probs, budget=5)

    assert len(tree.nodes) == 5
    by_id = {node.node_id: node for node in tree.nodes}
    for node in tree.nodes:
        if node.parent_id is not None:
            assert node.parent_id in by_id
            assert by_id[node.parent_id].depth == node.depth - 1
    assert tree.nodes[0].token_id == 10


def test_tree_attention_mask_allows_only_past_bonus_and_ancestors():
    tree = build_draft_tree(
        [[10, 11], [20, 21]],
        [[-0.1, -0.2], [-0.1, -0.2]],
        budget=3,
    )

    mask, position_ids = build_tree_attention_mask(
        tree, past_length=4, dtype=torch.float32, device=torch.device("cpu")
    )

    assert mask.shape == (1, 1, 4, 8)
    assert position_ids.tolist() == [[4, 5, 6, 5]]
    # Bonus query sees four cached tokens and itself, but no draft nodes.
    assert torch.isfinite(mask[0, 0, 0, :5]).all()
    assert torch.isneginf(mask[0, 0, 0, 5:]).all()
    # A sibling cannot attend to the other sibling.
    assert torch.isneginf(mask[0, 0, 3, 6])


def test_verifier_walk_accepts_matching_branch_then_emits_target_bonus():
    tree = build_draft_tree(
        [[10, 11], [20, 21]],
        [[-0.1, -0.2], [-0.1, -0.2]],
        budget=3,
    )
    # Nodes are popped in best-first order: 10, (10,20), 11.
    logits = torch.full((1 + len(tree.nodes), 32), -100.0)
    logits[0, 10] = 10.0
    logits[1, 20] = 10.0
    logits[2, 7] = 10.0

    accepted, bonus = walk_verified_tree(tree, logits)

    assert [tree.nodes[i].token_id for i in accepted] == [10, 20]
    assert bonus == 7
