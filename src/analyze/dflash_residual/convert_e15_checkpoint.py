"""Convert the released HF DFlash draft to MR-DFlash warm-start format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import sys

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "externals" / "dflash"))
    from dflash.model import DFlashDraftModel

    model = DFlashDraftModel.from_pretrained(
        args.draft_model, device_map="cpu", torch_dtype=torch.bfloat16
    )
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "mr_dflash_draft_weights_v1",
        "draft_state_dict": state,
        "source_model": args.draft_model,
        "keys": len(state),
    }, output)
    output.with_suffix(".manifest.json").write_text(
        json.dumps({
            "experiment": "E15",
            "conversion": "HF_DFlash_to_MR_DFlash_warm_start",
            "source_model": args.draft_model,
            "output": str(output),
            "keys": len(state),
            "status": "ok",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "keys": len(state), "status": "ok"}))


if __name__ == "__main__":
    main()
