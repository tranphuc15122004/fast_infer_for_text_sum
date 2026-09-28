# Build dữ liệu MR-DFlash 50k theo độ dài prompt

Builder quét toàn bộ raw ShareGPT và ArXiv, chuẩn hóa về JSONL hiện hành, loại ID trùng trong từng nguồn, rồi đếm token trên prompt sau Qwen3-4B chat template. Chỉ các prompt dài tối đa `10 × 1024 = 10.240` token mới đủ điều kiện lấy mẫu.

Các bin dùng biên đóng bên phải:

| Bin | Token prompt | Quota tổng |
|---:|---:|---:|
| 0 | 0–2.048 | 14.000 |
| 1 | 2.049–4.096 | 9.000 |
| 2 | 4.097–6.144 | 9.000 |
| 3 | 6.145–8.192 | 9.000 |
| 4 | 8.193–10.240 | 9.000 |

Tổng quota theo nguồn là 16.660 ShareGPT và 33.340 ArXiv (xấp xỉ 33/67). Bin 0–2k có 14.000 mẫu; bốn bin dài có 9.000 mẫu mỗi bin. Với preview hiện tại, giới hạn bin cho phép tối đa 16.672 ShareGPT; quota 16.660 giữ split 90/5/5 chính xác theo từng nguồn. Thành phần mỗi source-bin được tính từ capacity đã đo thực tế. Trong mỗi source-bin, mẫu có thứ hạng SHA-256 nhỏ nhất từ seed và ID được chọn, không lặp ID. Nếu không thể thỏa đồng thời quota nguồn và bin, builder ghi rõ phần thiếu trong report và dừng build.

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
  --output-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_50k_33_67_14k_9k \
  --batch-size 128 \
  --seed 42 \
  --preview-only
```

Xem `reports/length_distribution.json`, `reports/length_distribution.csv`, `reports/length_distribution.png` và `manifests/preview_manifest.json`. Preview báo tổng record, ID trùng, số prompt đủ điều kiện theo từng bin/source, quota matrix và phần thiếu nếu không khả thi. Nếu tiến trình bị ngắt, chạy lại cùng lệnh với `--resume`.

## 2. Build sau khi đã xem preview

Chỉ chạy bước này khi preview trên đúng raw source/tokenizer/seed đã được xem và quota khả thi. `--confirm-build` yêu cầu xác nhận tường minh; `--resume` dùng lại normalization và token-length cache đã tính ở bước preview. Thay đổi quota không làm mất hiệu lực cache; raw source, tokenizer và seed vẫn phải khớp.

```bash
python3 scripts/mr_dflash/build_balanced_dataset.py \
  --sharegpt-source /workspace/storage-shared/nlp/tungdd11/tungdecoder/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json \
  --arxiv-source /workspace/storage-shared/nlp/dungdx4/datasets/arxiv/train.label.jsonl \
  --tokenizer /workspace/storage-shared/models/Qwen3-4B \
  --output-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_50k_33_67_14k_9k \
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


## 5. Snapshot dataset đã build trên B200 (2026-09-24)

### Đường dẫn và artifact

Dataset hoàn tất trên server tại:

    /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_50k_50_50_10k/
    ├── normalized/
    │   ├── train_prompts.jsonl
    │   ├── val_prompts.jsonl
    │   └── test_prompts.jsonl
    ├── manifests/
    │   ├── preview_manifest.json
    │   └── build_manifest.json
    └── reports/
        ├── length_distribution.json
        ├── length_distribution.csv
        └── length_distribution.png

Tên thư mục phase1_50k_50_50_10k được giữ từ lần cấu hình đầu và không mô tả quota cuối cùng. Cấu hình và kết quả chuẩn cần đọc từ manifests/build_manifest.json; không suy ra tỷ lệ nguồn từ tên thư mục.

Log build:

    /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_50k_50_50_10k.build_14k_9k.log

### Nguồn, tokenizer và cách đo

| Thành phần | Giá trị |
|---|---|
| ShareGPT raw | /workspace/storage-shared/nlp/tungdd11/tungdecoder/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json |
| ArXiv raw | /workspace/storage-shared/nlp/dungdx4/datasets/arxiv/train.label.jsonl |
| Tokenizer | /workspace/storage-shared/models/Qwen3-4B |
| Seed | 42 |
| Batch size khi scan | 128 |
| Trần prompt | 10.240 token, gồm prompt sau chat template |
| Cách render | Qwen3 chat template, add_generation_prompt=True, enable_thinking=False; tokenizer gọi với add_special_tokens=False, không truncate |

ShareGPT giữ toàn bộ hội thoại làm ngữ cảnh đến message user cuối cùng, gồm các lượt assistant trước đó; response assistant sau user cuối không đưa vào prompt. Sampling tái lập: xếp hạng SHA-256 ổn định theo seed, source và ID bên trong từng bin; không lặp ID. Split cũng tái lập theo seed.

### Kết quả scan đầy đủ

Scan toàn bộ dữ liệu trước khi lấy mẫu:

| Source | Đã chuẩn hóa | ID duy nhất | ID trùng | Đủ điều kiện <=10.240 | Vượt trần |
|---|---:|---:|---:|---:|---:|
| ShareGPT | 92.825 | 92.825 | 0 | 92.799 | 26 |
| ArXiv | 202.703 | 202.703 | 0 | 142.404 | 60.299 |
| Tổng | 295.528 | 295.528 | 0 | 235.203 | 60.325 |

Số đủ điều kiện theo bin:

| Bin prompt (token, biên đóng) | ShareGPT có sẵn | ArXiv có sẵn | Tổng có sẵn | Quota lấy |
|---|---:|---:|---:|---:|
| 0–2.048 | 90.127 | 12.071 | 102.198 | 14.000 |
| 2.049–4.096 | 2.497 | 30.349 | 32.846 | 9.000 |
| 4.097–6.144 | 130 | 41.842 | 41.972 | 9.000 |
| 6.145–8.192 | 24 | 33.781 | 33.805 | 9.000 |
| 8.193–10.240 | 21 | 24.361 | 24.382 | 9.000 |

Quota thực tế theo source x bin:

| Bin | ShareGPT lấy | ArXiv lấy | Tổng |
|---|---:|---:|---:|
| 0–2.048 | 14.000 | 0 | 14.000 |
| 2.049–4.096 | 2.485 | 6.515 | 9.000 |
| 4.097–6.144 | 130 | 8.870 | 9.000 |
| 6.145–8.192 | 24 | 8.976 | 9.000 |
| 8.193–10.240 | 21 | 8.979 | 9.000 |
| Tổng | 16.660 | 33.340 | 50.000 |

Tỷ lệ nguồn toàn tập là ShareGPT 33,32%, ArXiv 66,68%. Không ép tỷ lệ 50/50 trong từng bin: ShareGPT gần như không có prompt trên 4.096 token, nên bin ngắn đầu tiên hoàn toàn là ShareGPT và các bin dài chủ yếu là ArXiv. Preview báo feasible=True.

### Split cuối cùng và định dạng

Build kết thúc với total=50.000: train=45.000, val=2.500, test=2.500. Các biên 90/5/5 được giữ chính xác theo tổng từng source và theo tổng từng bin:

| Nhóm | Train | Val | Test | Tổng |
|---|---:|---:|---:|---:|
| ShareGPT | 14.994 | 833 | 833 | 16.660 |
| ArXiv | 30.006 | 1.667 | 1.667 | 33.340 |
| Bin 0–2.048 | 12.600 | 700 | 700 | 14.000 |
| Mỗi bin còn lại | 8.100 | 450 | 450 | 9.000 |
| Toàn tập | 45.000 | 2.500 | 2.500 | 50.000 |

Các con số theo source và theo bin là các biên tổng; từng ô giao source x bin được phân bổ xác định để đồng thời giữ hai biên này.

Mỗi file là JSONL, một object canonical mỗi dòng. Schema cấp cao:

    {"id":"...", "source":"sharegpt", "conversations":[{"role":"user", "content":"..."}], "metadata":{"prompt_token_length":123, "prompt_length_bin":0, "split":"train"}}

source là sharegpt hoặc arxiv; conversations là danh sách message theo role/content; metadata gồm provenance normalizer cùng prompt_token_length, prompt_length_bin và split. build_manifest.json lưu quota, cấu hình tokenization, số lượng theo nguồn/split và SHA-256 danh sách ID mỗi split; dùng manifest làm nguồn xác thực khi audit.

### Ghi chú vận hành đã gặp

- Log build cuối xác nhận done total=50000 splits={'train': 45000, 'val': 2500, 'test': 2500}.
- Transformers cảnh báo một chuỗi dài 221.055 token vượt model_max_length=131.072. Scan vẫn hoàn tất; các prompt trên trần 10.240 bị loại khỏi tập lấy mẫu (26 ShareGPT, 60.299 ArXiv).
- Khi resume normalizer, không dùng str.splitlines() để tách JSONL: ký tự Unicode U+2028/U+2029 trong content có thể bị hiểu nhầm là ranh giới record. Dùng raw.split("\n") trong prepare_sharegpt.py và prepare_arxiv.py; hai file trên server đã được sửa như vậy để resume cache chính xác.
- Cache identity của normalization/token length chỉ phụ thuộc raw inputs, tokenizer, seed và trần prompt. Đổi quota không làm cache cũ mất hiệu lực; giữ helper _source_cache_identity khi đồng bộ/sửa builder.
