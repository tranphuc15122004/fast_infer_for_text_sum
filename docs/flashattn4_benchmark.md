# Quy trình Benchmark các baseline với backend Transformer FlashAttention-4 (FA4)

Tài liệu này hướng dẫn chi tiết cách sử dụng quy trình Benchmark các baseline tăng tốc suy luận (`vanilla_hf`, `eagle3`, `dflash`, `domino`, `dspark`) sử dụng backend Transformers native + FlashAttention-4 (FA4), hỗ trợ cả hai tập dữ liệu:
1. **Dữ liệu tiếng Việt (`eval_100`)**: gồm 4 dataset (`vietnews`, `wikilingua`, `vims`, `vlsp`), mỗi tập 100 mẫu.
   - **Đường dẫn trên server**: `/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/eval_100`
   - **Đường dẫn trên máy local**: `/home/tuantb/fast_infer_text_sum_Viet/datasets/eval_100` (đã được liên kết qua `data/eval_100` và `datasets/eval_100`)
2. **Dữ liệu LongBench tiếng Anh (`longbench_100_14k`)**: gồm 5 dataset (`gov_report`, `qmsum`, `multi_news`, `lcc`, `repobench-p`).

---

## 1. Tổng quan phương pháp và cấu hình

- **Cơ chế**: Native Transformers, batch size = 1, greedy decoding (`do_sample=False`). Toàn bộ target và draft attention đều dispatch qua kernel `flash_attention_4` (không fallback sang eager hay SDPA).
- **Môi trường yêu cầu cho GPU inference**:
  - Python 3.12 hệ thống
  - PyTorch 2.13/2.14 + CUDA 13.0
  - `flash-attn-4[cu13]==4.0.0b32` (hoặc b15 có shim tương thích)
  - GPU NVIDIA Blackwell B200 (SM100+)
  - Môi trường server offline, không nạp vLLM trong process đo.
- **5 Baseline so sánh**:
  - `vanilla_hf`: Baseline tham chiếu chuẩn (`model.generate()`, BF16, FA4).
  - `eagle3`: Speculative decoding AR profile (`total_token=17, depth=16, top_k=1`).
  - `dflash`: Block-level speculation (`block_size=16`, greedy).
  - `domino`: DFlash draft với causal correction và CUDA graph (`DraftCorrectionGraphRunner`).
  - `dspark`: SpecForge checkpoint evaluator với threshold 0.0.

---

## 2. Đường dẫn dữ liệu và Model

### Đường dẫn Dữ liệu tiếng Việt (`eval_100`)
- **Server**: `/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/eval_100`
- **Local**: `/home/tuantb/fast_infer_text_sum_Viet/datasets/eval_100` hoặc `data/eval_100`
- **Cấu trúc file**:
  - `vietnews_100.jsonl` (100 mẫu)
  - `wikilingua_100.jsonl` (100 mẫu)
  - `vims_100.jsonl` (100 mẫu)
  - `vlsp_100.jsonl` (100 mẫu)
  - `manifest.json` (chứa checksum SHA-256 và thống kê độ dài văn bản)

### Prompt template tiếng Việt
Mọi mẫu tiếng Việt được định dạng tự động qua prompt:
```text
Hãy tóm tắt văn bản sau bằng tiếng Việt. Chỉ trả lời bằng bản tóm tắt:

{document}
```
Sau đó được bọc qua chat template của tokenizer target checkpoint (ví dụ Qwen3).

### Checkpoints trên Server B200
Khai báo trong master config (`fast_infer_master_Viet.env` hoặc `fast_infer_master.env`):
- `MODEL_TARGET`: `/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B`
- `MODEL_EAGLE_DRAFT`: `/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B_eagle3`
- `MODEL_DFLASH_DRAFT`: `/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B-DFlash-b16`
- `MODEL_DOMINO_DRAFT`: `/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B-Domino-b16`
- `MODEL_DSPARK_DRAFT`: `/workspace/storage-shared/nlp/dungdx4/BERT/Dspark-Qwen3-4B-b7`

---

## 3. Kiểm tra trên máy Dev Local (CPU)

Do máy local (`tuantb@teslaT4`) dùng GPU T4 driver cũ không chạy được stack CUDA 13/FA4/Blackwell, quy trình kiểm tra trên máy local tập trung vào:
- Xác thực tính toàn vẹn của dataset và checksum từ `manifest.json`.
- Kiểm tra render prompt tiếng Việt và LongBench.
- Kiểm tra parser CLI và mapping cấu hình.

Chạy test bộ dữ liệu:
```bash
PYTHONPATH=src:scripts pytest tests/test_benchmark_vietnamese_fa4.py
```

---

## 4. Chạy Benchmark trên Server B200

Chuyển vào thư mục repo trên server và export đường dẫn master config:
```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum
export FAST_INFER_MASTER_CONFIG=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master_Viet.env
```

### Bước 1: Preflight check (Không tải model, kiểm tra GPU & Dataset)
Kiểm tra môi trường B200, FA4 import, và tính toàn vẹn của bộ dữ liệu tiếng Việt:
```bash
FI_GPU_IDS=0 bash scripts/run_fa4_benchmark.sh \
  --data-dir /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/eval_100 \
  --datasets all \
  --preflight-only
```
*(Nếu muốn preflight dữ liệu LongBench gốc, chỉ cần bỏ tham số `--data-dir`)*

### Bước 2: Smoke test (1 mẫu, kiểm tra thông suốt cả 5 baseline)
Chạy thử 1 mẫu của dataset `vietnews` với 64 token sinh ra:
```bash
FI_GPU_IDS=0 bash scripts/run_fa4_benchmark.sh \
  --data-dir /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/eval_100 \
  --mode smoke --datasets vietnews --samples-per-dataset 1 \
  --max-new-tokens 64 --warmup-tokens 64
```

### Bước 3: Representative benchmark (Chạy tập đại diện phân tầng độ dài)
Chạy 2 mẫu cho mỗi dataset trên cả 4 bộ tiếng Việt (tổng cộng 8 prompt x 5 baseline):
```bash
FI_GPU_IDS=0 bash scripts/run_fa4_benchmark.sh \
  --data-dir /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/eval_100 \
  --mode representative --datasets all --samples-per-dataset 2 \
  --max-new-tokens 512 --warmup-tokens 512 --repetitions 3
```

Hoặc chạy 20 mẫu/dataset (80 mẫu x 5 baseline):
```bash
FI_GPU_IDS=0 bash scripts/run_fa4_benchmark.sh \
  --data-dir /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/eval_100 \
  --mode representative --datasets all --samples-per-dataset 20 \
  --max-new-tokens 512 --warmup-tokens 512 --repetitions 1
```

### Bước 4: Full benchmark (100 mẫu x 4 dataset)
Chạy toàn bộ 100 mẫu cho cả 4 dataset tiếng Việt:
```bash
FI_GPU_IDS=0 bash scripts/run_fa4_benchmark.sh \
  --data-dir /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/eval_100 \
  --mode full --datasets all \
  --max-new-tokens 512 --repetitions 1
```

---

## 5. Chạy trên Dữ liệu LongBench Canonical (Tiếng Anh)

Khi muốn chuyển sang đánh giá trên LongBench tiếng Anh:
```bash
FI_GPU_IDS=0 bash scripts/run_fa4_benchmark.sh \
  --data-dir data/longbench_100_14k \
  --datasets all \
  --mode representative --samples-per-dataset 20 \
  --max-new-tokens 512
```
Runner tự động nhận diện LongBench manifest (`schema_version: longbench-canonical-v1`) và chuyển sang template prompt tiếng Anh của từng dataset (`gov_report`, `qmsum`, `multi_news`, `lcc`, `repobench-p`).

---

## 6. Output và Báo cáo Kết quả

Mỗi lần chạy sẽ tạo một thư mục con dưới `outputs/fa4_native_benchmark/<run-id>/` gồm:
- `report_vi.md`: Báo cáo tóm tắt Markdown tiếng Việt hiển thị bảng so sánh (ROUGE-L, BLEU-4, Latency E2E, TTFT, TPOT, Tok/s, DSR, ESR, Tỷ lệ accept %, Greedy exact match, Token LCS overlap).
- `metrics_summary.csv`: Bảng số liệu dạng CSV cho tất cả các baseline.
- `results.jsonl`: Kết quả chi tiết của từng mẫu (output text, token IDs, latency từng bước, acceptance counters).
- `samples.jsonl`: Danh sách các mẫu được chọn kèm số lượng token.
- `run_report.json`: Toàn bộ metadata, thông số GPU, phiên bản package và runtime validation.
