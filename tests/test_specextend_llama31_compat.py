import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPECEXTEND = ROOT / "externals/SpecExtend/specextend"
sys.path.insert(0, str(SPECEXTEND))


def test_target_attention_accepts_llama31_rope_scaling():
    from transformers import LlamaConfig
    from shared.modeling_llama_kv_target import LlamaAttention

    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=131072,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
    )

    attention = LlamaAttention(config)

    assert attention.rotary_emb.inv_freq.numel() == config.hidden_size // config.num_attention_heads // 2
