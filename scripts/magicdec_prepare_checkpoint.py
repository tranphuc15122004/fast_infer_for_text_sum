#!/usr/bin/env python3
"""Download + convert a HF checkpoint to MagicDec's model.pth format.

MagicDec's benchmark scripts need a converted checkpoint whose FOLDER NAME
matches an entry in Engine/SnapKV/model.py (e.g. "tinyllama", "llama-3.1-8b",
"qwen2.5-7b", ...). Usage:

  python scripts/magicdec_prepare_checkpoint.py \
      --repo-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --model-key tinyllama \
      --out-dir ~/.cache/huggingface/magicdec/tinyllama
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common.paths import ROOT

MAGICDEC = ROOT / "externals" / "MagicDec"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--model-key", required=True,
                        help="folder name = ModelArgs key, e.g. tinyllama")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_pth = out_dir / "model.pth"
    if model_pth.exists():
        print(f"model.pth already exists: {model_pth}")
        return

    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(MAGICDEC) + ":" + env.get("PYTHONPATH", "")

    def run(cmd: list[str]) -> None:
        print("+ " + " ".join(cmd))
        subprocess.run(cmd, cwd=MAGICDEC, env=env, check=True)

    dl_args = [sys.executable, "download.py", "--repo_id", args.repo_id,
               "--out_dir", str(out_dir)]
    if args.hf_token:
        dl_args += ["--hf_token", args.hf_token]
    run(dl_args)

    # convert_hf_checkpoint.py outputs into the same dir; ensure the model-key
    # dir name is used so Transformer.from_name resolves correctly.
    run([sys.executable, "convert_hf_checkpoint.py",
         "--checkpoint_dir", str(out_dir)])
    if not model_pth.exists():
        # converter may place model.pth one level down; report location.
        cands = list(out_dir.rglob("model.pth"))
        if cands:
            print(f"model.pth found at: {cands[0]}")
            return
        raise SystemExit("Conversion produced no model.pth")
    print(f"Checkpoint ready: {model_pth}")


if __name__ == "__main__":
    main()
