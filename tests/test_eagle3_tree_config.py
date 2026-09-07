import pytest

from scripts.eagle3_infer_qwen3 import resolve_eagle_tree_config


def test_total_token_is_capped_by_generated_tree_capacity():
    config = resolve_eagle_tree_config(total_token=32, depth=0, top_k=4)

    assert config["max_total_token"] == 5
    assert config["total_token"] == 5
    assert config["adjusted"] is True


def test_valid_total_token_is_kept():
    config = resolve_eagle_tree_config(total_token=32, depth=8, top_k=4)

    assert config["max_total_token"] == 133
    assert config["total_token"] == 32
    assert config["adjusted"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_token": 0, "depth": 8, "top_k": 4},
        {"total_token": 32, "depth": -1, "top_k": 4},
        {"total_token": 32, "depth": 8, "top_k": 0},
    ],
)
def test_tree_config_rejects_non_positive_values(kwargs):
    with pytest.raises(ValueError):
        resolve_eagle_tree_config(**kwargs)
