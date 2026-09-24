# Build dữ liệu MR-DFlash 50k theo độ dài prompt

Builder quét toàn bộ raw ShareGPT và ArXiv, chuẩn hóa về JSONL hiện hành, loại ID trùng trong từng nguồn, rồi đếm token trên prompt sau Qwen3-4B chat template. Chỉ các prompt dài tối đa `10 × 1024 = 10.240` token mới đủ điều kiện lấy mẫu.

Các bin dùng biên đóng bên phải:

| Bin | Token prompt | Quota tổng |
|---:|---:|---:|
| 0 | 0–2.048 | 10.000 |
| 1 | 2.049–4.096 | 10.000 |
| 2 | 4.097–6.144 | 10.000 |
| 3 | 6.145–8.192 | 10.000 |
| 4 | 8.193–10.240 | 10.000 |

Tổng quota theo nguồn là 25.000 ShareGPT và 25.000 ArXiv. Tỷ lệ nguồn chỉ được cố định trên toàn bộ dữ liệu; thành phần của từng bin được tính từ capacity thực tế để đạt đủ quota. Trong mỗi source-bin, mẫu có thứ hạng SHA-256 nhỏ nhất từ seed và ID được chọn, không lặp ID. Nếu không thể thỏa đồng thời quota nguồn và bin, builder ghi rõ phần thiếu trong report và dừng build.

## 1. Preview toàn bộ dữ liệu trên B200

Preview không lấy mẫu và không tạo split. Nó chuẩn hóa, đo token cho toàn bộ record hợp lệ duy nhất và ghi báo cáo/quota feasibility. Chạy trên server có source và tokenizer local:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 scripts/mr_dflash/build_balanced_dataset.py \
  --sharegpt-source /workspace/storage-shared/nlp/tungdd11/tungdecoder/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json \
  --arxiv-source /workspace/storage-shared/nlp/dungdx4/datasets/arxiv/train.label.jsonl \
  --tokenizer /workspace/storage-shared/models/Qwen3-4B \
  --output-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_50k_50_50_10k \
  --batch-size 128 \
  --seed 42 \
  --preview-only
```

Xem `reports/length_distribution.json`, `reports/length_distribution.csv`, `reports/length_distribution.png` và `manifests/preview_manifest.json`. Preview báo tổng record, ID trùng, số prompt đủ điều kiện theo từng bin/source, quota matrix và phần thiếu nếu không khả thi. Nếu tiến trình bị ngắt, chạy lại cùng lệnh với `--resume`.

## 2. Build sau khi đã xem preview

Chỉ chạy bước này khi preview trên đúng raw source/tokenizer/seed đã được xem và quota khả thi. `--confirm-build` yêu cầu xác nhận tường minh; `--resume` dùng lại normalization và token-length cache đã tính ở bước preview.

```bash
python3 scripts/mr_dflash/build_balanced_dataset.py \
  --sharegpt-source /workspace/storage-shared/nlp/tungdd11/tungdecoder/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json \
  --arxiv-source /workspace/storage-shared/nlp/dungdx4/datasets/arxiv/train.label.jsonl \
  --tokenizer /workspace/storage-shared/models/Qwen3-4B \
  --output-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_50k_50_50_10k \
  --batch-size 128 \
  --seed 42 \
  --resume \
  --confirm-build
```

Builder từ chối build nếu thiếu preview, nếu preview không khớp input/tokenizer/seed, hoặc nếu quota không khả thi. Khi thiếu dữ liệu, không được bật chế độ rút ngắn hay lặp mẫu; cần xem report và quyết định lại kế hoạch riêng.

## 3. Artifact

```text
<output-root>/
├── normalized/
│   ├── train_prompts.jsonl
│   ├── val_prompts.jsonl
│   └── test_prompts.jsonl
├── manifests/
│   ├── preview_manifest.json
│   └── build_manifest.json
├── reports/
│   ├── length_distribution.csv
│   ├── length_distribution.json
│   └── length_distribution.png
└── .build_work/                 # token-length cache để audit/resume
```

Mỗi JSONL row theo canonical prompt schema: `id`, `source`, `conversations`, `metadata`. Metadata có `prompt_token_length`, `prompt_length_bin`, `split` và trường provenance từ normalizer. Không sửa raw source. Tổng split là train 45.000, val 2.500, test 2.500; mỗi source và mỗi bin đều giữ biên 90/5/5.

## 4. Giới hạn khi train/tokenize

Trần 10.240 áp dụng cho prompt. Để giữ prompt dài cùng phần assistant output, regeneration, validate, tokenize, cache và `data.max_length` của training config cần dùng khoảng `16.384` token. Giới hạn 8.192 hiện có sẽ từ chối hoặc cắt mất các prompt ở hai bin cuối.
