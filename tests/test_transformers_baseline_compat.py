import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str, pythonpath: list[Path]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in pythonpath]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def test_fastkv_registers_snapkv_for_modern_transformers():
    proc = _run_python(
        """
import transformers
from baselines import monkeypatch
monkeypatch.replace_llama('snapkv')
from transformers.models.llama import modeling_llama
assert modeling_llama.LlamaAttention.forward.__name__ != 'forward'
print(transformers.__version__)
""",
        [ROOT / "scripts", ROOT / "externals" / "FastKV"],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_fastkv_runs_tiny_modern_mistral_forward():
    proc = _run_python(
        """
import torch
from transformers import MistralConfig, MistralForCausalLM
from baselines.monkeypatch import replace_mistral

config = MistralConfig(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    sliding_window=None,
    attention_dropout=0.0,
)
model = MistralForCausalLM(config).eval()
replace_mistral('snapkv')
for layer in model.model.layers:
    layer.self_attn.config.window_size = 8
    layer.self_attn.config.max_capacity_prompt = 16
    layer.self_attn.config.kernel_size = 5
    layer.self_attn.config.pooling = 'avgpool'
    layer.self_attn.config.merge = 'pivot'
    layer.self_attn.config.retain_rate = 1.0
    layer.self_attn.config.eviction_mode = 'a2sf'
with torch.no_grad():
    output = model(input_ids=torch.randint(0, 128, (1, 12)), use_cache=True)
assert output.logits.shape == (1, 12, 128)
""",
        [ROOT / "scripts", ROOT / "externals" / "FastKV"],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_gemfilter_loader_imports_with_modern_transformers():
    proc = _run_python(
        """
import transformers
import my_utils.load_model
print(transformers.__version__)
""",
        [ROOT / "scripts", ROOT / "externals" / "GemFilter"],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_snapkv_short_prefill_does_not_apply_a_large_window():
    proc = _run_python(
        """
import torch
from baselines.snapkv.utils import SnapKVCluster
cluster = SnapKVCluster(
    window_size=1024,
    max_capacity_prompt=2048,
    retain_rate=0.1,
    eviction_mode='proportional',
)
q = torch.randn(1, 2, 6, 4)
k = torch.randn_like(q)
v = torch.randn_like(q)
out_k, out_v = cluster.update_kv(q, q, v, None, 1)
assert out_k.shape == k.shape
assert out_v.shape == v.shape
""",
        [ROOT / "externals" / "FastKV"],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
