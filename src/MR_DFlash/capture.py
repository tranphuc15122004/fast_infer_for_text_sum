"""Capture offline target features cho train DFlash (bản HF, thay SGLang).

Tương ứng ``scripts/prepare_hidden_states.py`` + capture contract DFlash của
SpecForge nhưng chạy thẳng bằng Hugging Face ``AutoModelForCausalLM`` (không
cần SGLang server). Với mỗi mẫu hội thoại:

- render toàn bộ chuỗi (prompt + assistant) bằng chat template của tokenizer,
  dựng ``loss_mask`` cho nội dung assistant;
- chạy target model một lượt (teacher-forcing toàn chuỗi) lấy hidden states
  tại các layer ``target_layer_ids``; feature tại layer ``l`` lấy từ
  ``outputs.hidden_states[l + 1]`` (giống ``extract_context_feature`` của
  SpecForge với offset = 1);
- concat theo chiều feature → ``hidden_states``, lưu file ``.ckpt`` chứa
  ``input_ids`` / ``loss_mask`` / ``hidden_states``.

Note: hidden states của toàn chuỗi được lưu (kể cả prompt) vì block draft cần
context feature của mọi vị trí trước anchor. Capture không cần lm_head/embed
của target (chúng được nạp ở bước train).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .data import build_sample, iter_jsonl, save_feature_file


def resolve_layer_ids(layer_ids: Optional[List[int]], num_layers: int) -> List[int]:
    if layer_ids:
        for value in layer_ids:
            if value < 0 or value >= num_layers:
                raise ValueError(
                    f"target_layer_ids phải thuộc [0,{num_layers}), got {value}"
                )
        return list(layer_ids)
    from .config import build_target_layer_ids

    return build_target_layer_ids(num_layers, 1)


def _extract_context_feature(
    hidden_states: List[torch.Tensor],
    layer_ids: List[int],
) -> torch.Tensor:
    """Concat hidden states sau layer ``layer_id`` (offset = +1)."""
    offset = 1
    selected = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected, dim=-1)


class HFTargetCapture:
    """Capture feature target bằng HF model (eval, no-grad)."""

    def __init__(
        self,
        target_model_path: str,
        layer_ids: List[int],
        *,
        cache_dir: str = "./cache",
        trust_remote_code: bool = False,
        torch_dtype: str = "bfloat16",
        device: str = "auto",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[torch_dtype]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            target_model_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            target_model_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

        num_layers = int(self.model.config.num_hidden_layers)
        self.layer_ids = resolve_layer_ids(layer_ids, num_layers)
        self.context_feature_dim = len(self.layer_ids) * int(
            self.model.config.hidden_size
        )

    def capture_one(
        self,
        input_ids: List[int],
    ) -> torch.Tensor:
        """Chạy target trên toàn chuỗi → hidden concat (1, seq, feat)."""
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=ids,
                output_hidden_states=True,
                use_cache=False,
            )
        features = _extract_context_feature(outputs.hidden_states, self.layer_ids)
        return features.float().cpu() if self.device.type == "cpu" else features.cpu()


def capture_dataset(
    *,
    target_model_path: str,
    data_path: str,
    output_path: str,
    max_length: int,
    layer_ids: Optional[List[int]] = None,
    num_samples: Optional[int] = None,
    cache_dir: str = "./cache",
    trust_remote_code: bool = False,
    torch_dtype: str = "bfloat16",
    device: str = "auto",
) -> Dict[str, int]:
    """Capture toàn bộ dataset → các file ``.ckpt`` dưới ``output_path``.

    Trả về thống kê {captured, skipped_invalid, skipped_no_consecutive}.
    """
    capturer = HFTargetCapture(
        target_model_path,
        layer_ids or [],
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device=device,
    )
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"captured": 0, "skipped_invalid": 0, "skipped_no_consecutive": 0}
    for index, row in enumerate(iter_jsonl(data_path)):
        if num_samples is not None and stats["captured"] >= num_samples:
            break
        sample = build_sample(row, capturer.tokenizer, max_length)
        if sample is None:
            stats["skipped_invalid"] += 1
            continue
        try:
            features = capturer.capture_one(sample["input_ids"])
        except Exception as exc:  # capture fail → bỏ mẫu, ghi rõ lý do
            print(f"[capture] skip sample {sample['id']!r}: {exc!r}")
            stats["skipped_invalid"] += 1
            continue
        seq = len(sample["input_ids"])
        if features.shape[1] != seq:
            print(
                f"[capture] skip {sample['id']!r}: feature len {features.shape[1]} "
                f"!= input len {seq}"
            )
            stats["skipped_invalid"] += 1
            continue
        if features.shape[1] < 2:
            stats["skipped_no_consecutive"] += 1
            continue

        tensors = {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "loss_mask": torch.tensor(sample["loss_mask"], dtype=torch.long),
            "hidden_states": features[0],
        }
        save_feature_file(str(out_dir / f"sample_{index:08d}.ckpt"), tensors)
        stats["captured"] += 1
        if stats["captured"] % 25 == 0:
            print(f"[capture] {stats['captured']} mẫu ...")
    return stats


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture offline target features cho train DFlash (HF)."
    )
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    stats = capture_dataset(
        target_model_path=args.target_model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        max_length=args.max_length,
        layer_ids=args.target_layer_ids,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
        device=args.device,
    )
    print(f"[capture] xong: {stats}")


if __name__ == "__main__":
    main()
