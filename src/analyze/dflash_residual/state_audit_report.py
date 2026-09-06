"""Report paired reference-prefix/on-policy E17-A state audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .causal_diagnosis import paired_state_comparison
from .io import read_trace_jsonl


def report(result: dict) -> str:
    lines = [
        "# E17-A — Reference-prefix versus on-policy state audit",
        "",
        "Cùng source/document, target Qwen3-4B, DFlash checkpoint, Top-16, native block 16, context cap 1024, bfloat16/SDPA trên T4. Reference mode thêm gold assistant prefix; on-policy mode giữ collector deployment hiện tại. So sánh ghép theo document, bootstrap document 500 lần.",
        "",
    ]
    for fold, item in result["folds"].items():
        lines.extend([
            f"## {fold}",
            "",
            "| State | Docs | Blocks | Rows | MAT_O16 | R16@3 | R16@4 | R16@5 | R16@6 | R16@7 | R16@8 | J16@3 | J16@4 | J16@5 | J16@6 | J16@7 | J16@8 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for mode in ("on_policy", "reference"):
            x = item[mode]
            r = x["marginal_recall"]
            j = x["joint_survival"]
            values = [mode, x["documents"], x["blocks"], x["rows"], f"{x['mat_o16']:.4f}"]
            values += [f"{r.get(str(pos), 0.0):.4f}" for pos in range(3, 9)]
            values += [f"{j.get(str(pos), 0.0):.4f}" for pos in range(3, 9)]
            lines.append("| " + " | ".join(map(str, values)) + " |")
        lines.extend(["", "Reference minus on-policy bootstrap:"])
        boot = item["bootstrap"]["reference_minus_on_policy"]
        for key in ("delta_mat_o16", "delta_r16_3", "delta_r16_8", "delta_j16_3", "delta_j16_8"):
            x = boot[key]
            lines.append(f"- `{key}`: mean `{x['mean']:.4f}`, 95% CI `{x['ci95']}`.")
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "Multi-News cho state effect lớn và CI dương ở late prefix position/J16. GovReport có point estimate MAT_O16 dương nhưng CI rộng chứa 0, còn R16/J16 differences không nhất quán. Vì vậy E17-A chỉ đạt **workload-specific pilot evidence**, chưa đạt gate mechanism general trên ≥2 datasets.",
        "",
        "## Decision",
        "",
        "Không chạy E18 intervention ở phase này. Không được gọi on-policy alignment là proposal; cần rotated fold lớn hơn hoặc server run trước khi can thiệp training.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", action="append", required=True, help="NAME=ON_POLICY_PATH,REFERENCE_PATH")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    folds = {}
    for spec in args.fold:
        name, paths = spec.split("=", 1)
        on_path, ref_path = paths.split(",", 1)
        folds[name] = paired_state_comparison(
            read_trace_jsonl(on_path), read_trace_jsonl(ref_path), bootstrap_samples=500
        )
    result = {"status": "ok", "folds": folds}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(report(result), encoding="utf-8")
    print(f"E17-A report: {output}")


if __name__ == "__main__":
    main()
