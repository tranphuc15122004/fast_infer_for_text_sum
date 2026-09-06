"""Export an MR-DFlash training checkpoint as an HF DFlash draft directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-draft-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import sys

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "externals" / "dflash"))
    from dflash.model import DFlashDraftModel

    model = DFlashDraftModel.from_pretrained(
        args.base_draft_model, device_map="cpu", torch_dtype=torch.bfloat16
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("draft_state_dict")
    if state is None:
        raise ValueError(f"checkpoint không có draft_state_dict: {args.checkpoint}")
    state = {
        (key[len("draft_model."):] if key.startswith("draft_model.") else key): value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"checkpoint không khớp: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    print({"output": str(output), "global_step": payload.get("global_step"), "status": "ok"})


if __name__ == "__main__":
    main()
