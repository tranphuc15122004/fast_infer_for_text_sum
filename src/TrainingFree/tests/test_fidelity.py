import pytest
import torch

from src.TrainingFree.fidelity import (
    fake_quantize_source_span_,
    fake_quantize_symmetric,
    install_source_cache_quantizer,
)


def test_symmetric_fake_quantization_uses_source_axis_scale_and_signed_levels() -> None:
    values = torch.tensor([[-1.0], [-0.6], [0.0], [0.6], [1.0]])

    quantized = fake_quantize_symmetric(values, bits=4, quant_dim=0)

    expected = torch.tensor([[-1.0], [-4.0 / 7.0], [0.0], [4.0 / 7.0], [1.0]])
    torch.testing.assert_close(quantized, expected, atol=1e-6, rtol=0.0)


def test_source_fake_quantization_leaves_prefix_suffix_and_other_kv_heads_unchanged() -> None:
    cache_tensor = torch.tensor(
        [[
            [[0.25], [0.10], [0.25], [0.75]],
            [[0.50], [0.10], [0.25], [0.80]],
        ]],
        dtype=torch.bfloat16,
    )
    original = cache_tensor.clone()

    fake_quantize_source_span_(
        cache_tensor,
        source_start=1,
        source_end=3,
        bits=4,
        kv_heads=(1,),
    )

    torch.testing.assert_close(cache_tensor[:, 0], original[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(cache_tensor[:, 1, 0], original[:, 1, 0], rtol=0, atol=0)
    torch.testing.assert_close(cache_tensor[:, 1, 3], original[:, 1, 3], rtol=0, atol=0)
    assert not torch.equal(cache_tensor[:, 1, 1:3], original[:, 1, 1:3])


class _FakeLayerCache:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys = keys
        self.values = values


class _FakeDynamicCache:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.layers = [_FakeLayerCache(keys, values)]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        layer = self.layers[layer_idx]
        layer.keys = torch.cat((layer.keys, key_states), dim=-2)
        layer.values = torch.cat((layer.values, value_states), dim=-2)
        return layer.keys, layer.values


def test_cache_hook_quantizes_source_once_and_keeps_appended_generated_kv_bf16() -> None:
    keys = torch.tensor([[[[0.25], [0.10], [0.25]]]], dtype=torch.bfloat16)
    values = torch.tensor([[[[0.5], [0.10], [0.25]]]], dtype=torch.bfloat16)
    cache = _FakeDynamicCache(keys.clone(), values.clone())
    prefix_and_source_keys = keys.clone()
    prefix_and_source_values = values.clone()

    install_source_cache_quantizer(
        cache,
        source_start=1,
        source_end=3,
        k_bits=4,
        v_bits=8,
    )
    generated_key = torch.tensor([[[[0.375]]]], dtype=torch.bfloat16)
    generated_value = torch.tensor([[[[0.625]]]], dtype=torch.bfloat16)
    returned_keys, returned_values = cache.update(generated_key, generated_value, 0)

    torch.testing.assert_close(returned_keys[..., 0, :], prefix_and_source_keys[..., 0, :], rtol=0, atol=0)
    torch.testing.assert_close(returned_values[..., 0, :], prefix_and_source_values[..., 0, :], rtol=0, atol=0)
    assert not torch.equal(returned_keys[..., 1:3, :], prefix_and_source_keys[..., 1:3, :])
    torch.testing.assert_close(returned_keys[..., 3:, :], generated_key, rtol=0, atol=0)
    torch.testing.assert_close(returned_values[..., 3:, :], generated_value, rtol=0, atol=0)


@pytest.mark.parametrize("bits", (2, 3, 5))
def test_fake_quantization_rejects_unconfigured_bit_widths(bits: int) -> None:
    with pytest.raises(ValueError, match="bits"):
        fake_quantize_symmetric(torch.ones(3, 2), bits=bits)


def test_16_bit_fake_quantization_is_an_exact_noop() -> None:
    values = torch.randn(2, 5, 7, 4, dtype=torch.bfloat16)

    quantized = fake_quantize_symmetric(values, bits=16)

    torch.testing.assert_close(quantized, values, rtol=0, atol=0)
