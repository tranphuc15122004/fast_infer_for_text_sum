# Thực nghiệm residual headroom của DFlash/DFlash2

Package này triển khai P0–P4 cho câu hỏi:

> Khi context dài lên, DFlash mất token đúng khỏi candidate set, hay DFlash2
> không chọn token đúng dù token đó vẫn còn trong candidate set?

## Contract

Collector ghi một JSONL row cho mỗi `(sample_id, round_index,
draft_position)`. `candidate_token_ids` là Top-M thật theo score giảm dần;
không được chèn target token vào list. `target_token_source` phải ghi rõ
`verifier_posterior` hoặc `canonical_continuation`. `accepted_draft_len` là số
draft token được accept liên tiếp, không bao gồm target fallback token.

DFlash2 selection có thể ghi file tối giản:

```json
{"run_id":"r1","sample_id":"s1","round_index":0,"draft_position":1,"selected_token_id":123}
```

File này được join bằng đúng bốn khóa; duplicate hoặc thiếu khóa sẽ fail.

## P0 — alignment

Chạy official benchmark adapter hiện có và collector custom trên cùng sample
canonical. Output benchmark của `scripts/infer_dflash.py` có
`acceptance_lengths`, trong đó mỗi phần tử gồm một target fallback nên analyzer
trừ một:

```bash
python3 -m src.analyze.dflash_residual.run \
  --phase p0 \
  --official outputs/dflash_official.jsonl \
  --custom outputs/dflash_custom_acceptance.jsonl \
  --output outputs/dflash_residual/p0
```

`PASS` chỉ khi hai bên có acceptance dương, có đủ block và MAT gần nhau.

## Tạo trace P2

Collector cần target/draft snapshot và CUDA-compatible Transformers/DFlash;
không chạy trên host dev T4 theo CPU workflow của repo:

```bash
python3 -m src.analyze.dflash_residual.trace_dflash \
  --target-model "$MODEL_TARGET" \
  --draft-model "$MODEL_DFLASH_DRAFT" \
  --input data/representative_100/cnn_dailymail_representative.jsonl \
  --output outputs/dflash_residual/cnn_trace.jsonl \
  --max-samples 100 \
  --max-new-tokens 32 \
  --top-m 16 \
  --context-lengths 1024,2048,4096,8192,16384 \
  --truncate-side right
```

Collector giữ native block size của checkpoint. Nếu native block size là 16,
số proposal thực tế được ghi trong `max_depth` theo semantics của official
runner; không tự gắn thêm slot để làm đẹp cho bảng.

## P1–P4

Không có DFlash2 selection vẫn chạy được P1, P2 và P4; P3 khi đó trả
`UNAVAILABLE`:

```bash
python3 -m src.analyze.dflash_residual.run \
  --phase all \
  --trace outputs/dflash_residual/cnn_trace.jsonl \
  --dflash2-selection outputs/dflash_residual/dflash2_selection.jsonl \
  --output outputs/dflash_residual/cnn_all
```

Mỗi run có `metrics.json`, `metrics.csv`, `report.md`. P2 thêm
`p2/coverage.csv` và `p2/recall_heatmap.png` nếu matplotlib có sẵn. P3 báo
`candidate_miss_rows`, `selection_error_rows`, `MAT_D`, `MAT_D2`, `MAT_O16`,
`G_sel`/`oracle_headroom` và `rho_D2`; nếu có ít nhất hai context bin đủ dữ
liệu, P3 thêm `p3/rho_by_context.png`. P4 fit:

```text
hit ~ 1 + log1p(context_length) + draft_position
          + log1p(context_length) * draft_position
```

Bootstrap theo document. H4 chỉ PASS nếu upper 95% CI của hệ số tương tác
âm; dữ liệu sparse trả `INCONCLUSIVE`.

## Synthetic CPU smoke

```bash
python3 - <<'PY'
from pathlib import Path
from src.analyze.dflash_residual.run import run_synthetic
run_synthetic(Path("/tmp/dflash_residual_synthetic.jsonl"), documents=6)
PY
python3 -m src.analyze.dflash_residual.run \
  --phase all \
  --trace /tmp/dflash_residual_synthetic.jsonl \
  --output /tmp/dflash_residual_result \
  --bootstrap-samples 200 \
  --min-documents 5
```

Synthetic fixture chỉ kiểm tra plumbing/statistics; không được dùng làm
evidence cho H1–H4.
