# Phase 1 Dataset Analysis Design

**Goal:** Quét toàn bộ raw ShareGPT và ArXiv để báo cáo phân phối dữ liệu trước khi quyết định sampling/split, tách hoàn toàn khỏi prepare/build, vLLM regenerate và hidden-cache.

## Scope

- Analyzer đọc raw source theo streaming, không random sample và không ghi đè raw source.
- Analyzer tạo JSON report, row-level audit JSONL và biểu đồ PNG từ toàn bộ record hợp lệ/lỗi.
- `run_b200_vllm_phase1.sh` không chạy `prepare_server_data.py` hoặc `build_pilot_dataset.py`; nó chỉ tiêu thụ `PREPARED_ROOT/normalized/*.jsonl` đã được duyệt.
- Sampling/split train-val-test vẫn giữ trong `prepare_server_data.py` cho bước sau, nhưng không còn là một phần ngầm của Phase 1 launcher.

## Analysis contract

Analyzer phải ghi:

- tổng số record đọc, hợp lệ, lỗi và duplicate ID;
- thống kê theo source;
- raw character length và tokenizer token length khi tokenizer local có sẵn;
- quantile P50/P75/P90/P95/P99/max;
- ShareGPT: số turn, role counts, prompt/last-user length;
- ArXiv: document/reference length và source fields;
- các record lỗi/missing field vào JSONL riêng;
- các figure reproducible bằng matplotlib/seaborn.

Analyzer không được tạo `train_prompts.jsonl`, `val_prompts.jsonl`, `test_prompts.jsonl`, regenerate output hoặc target feature cache.

## Output layout

```text
<analysis-root>/
├── summary.json
├── records.jsonl
├── issues.jsonl
└── figures/
    ├── token_length_distribution.png
    ├── token_length_ecdf.png
    ├── source_composition.png
    └── conversation_structure.png
```

## Validation

- Unit tests dùng fixture nhỏ, không cần model hoặc internet.
- Smoke analysis kiểm tra output schema, non-random ordering và figure files.
- Full scan chỉ chạy khi raw source path tồn tại trên server và sau khi smoke analysis pass.
