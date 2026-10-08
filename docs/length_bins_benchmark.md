# Tài liệu Bộ dữ liệu Benchmark phân vùng theo độ dài (Length Bins Benchmark)

Tài liệu này mô tả chi tiết thiết kế, ý nghĩa thực nghiệm, cấu trúc và cách sử dụng bộ dữ liệu **phân vùng theo độ dài (Length Bins)** phục vụ đo đạc và so sánh hiệu năng các kỹ thuật tăng tốc suy luận ngữ cảnh dài (Long-Context LLM Inference).

---

## 1. Đường dẫn lưu trữ

- **Đường dẫn canonical trên server sản xuất (B200 / Production Server):**
  ```text
  /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/length_bins/
  ```
- **Đường dẫn local / repository:**
  ```text
  data/length_bins/
  ```

---

## 2. Mục tiêu & Cơ sở khoa học

Trong suy luận mô hình ngôn ngữ lớn (LLM Inference), hành vi phần cứng và hiệu năng thuật toán thay đổi căn bản theo chiều dài ngữ cảnh:
1. **Pha Prefill:** Chi phí tính toán tự chú ý (Self-Attention) tỷ lệ thuận với bậc hai chiều dài ngữ cảnh ($O(N^2)$). Khi prompt ngắn (< 2k token), Prefill diễn ra gần như tức thì (< 50ms) và Compute-bound chưa bộc lộ; nhưng khi prompt vượt 8k hay 12k token, thời gian Prefill chiếm phần lớn tổng thời gian xử lý và tiêu tốn lượng VRAM khổng lồ.
2. **Pha Decode:** Bị nghẽn bởi băng thông bộ nhớ (Memory Bandwidth-bound) khi phải nạp toàn bộ KV Cache qua mỗi bước sinh token. Ngữ cảnh càng dài, dung lượng KV Cache càng lớn, làm giảm tốc độ sinh token per second.
3. **Các kỹ thuật tăng tốc:**
   - **Speculative Decoding (EAGLE-3, DFlash, MagicDec, SpecExtend):** Khi prompt quá ngắn, chi phí sinh draft và xác thực có thể gây phản tác dụng (overhead > speedup). Khi prompt dài, tỷ lệ chấp nhận draft token và khả năng duy trì context quyết định mức speedup thực tế.
   - **Sparse Attention & Context Prefill (MInference, FlexPrefill, SpecPrefill):** Phát huy sức mạnh tối đa ở dải dài (> 8k token), giúp giảm tải tính toán Prefill từ $O(N^2)$ về gần tuyến tính.
   - **KV Cache Compression & Eviction (FastKV, GemFilter, RocketKV):** Tiết kiệm dung lượng bộ nhớ và băng thông ở dải ngữ cảnh dài, nhưng cần kiểm soát nghiêm ngặt suy giảm chất lượng nội dung tóm tắt (ROUGE / Edit Similarity).

Để quan sát chính xác **điểm chuyển tiếp (transition point)** và **đường cong mở rộng hiệu năng (scaling curve)** của từng kỹ thuật, bộ dữ liệu được chia thành **5 phân vùng độ dài độc lập**, mỗi phân vùng gồm **50 mẫu**.

---

## 3. Ý nghĩa chi tiết của 5 phân vùng

| Phân vùng | Tệp dữ liệu | Khoảng Token | Token TB | Ý nghĩa đo lường & Điểm nghẽn phần cứng |
|:---:|---|:---:|:---:|---|
| **Bin 1** | `bin1_00k_02k.jsonl` | `[0 – 2,000)` | **1,484** | **Overhead Check:** Kiểm tra độ trễ nền (baseline latency) và chi phí phụ (overhead) của các thuật toán phức tạp khi ngữ cảnh ngắn. |
| **Bin 2** | `bin2_02k_04k.jsonl` | `[2,000 – 4,000)` | **2,964** | **Transition Zone:** Vùng chuyển tiếp; Prefill bắt đầu tăng tỷ trọng. Đo điểm bắt đầu có lãi tốc độ (speedup > 1.0x). |
| **Bin 3** | `bin3_04k_08k.jsonl` | `[4,000 – 8,000)` | **5,791** | **Compression Zone:** Ngưỡng ngữ cảnh phổ biến nhất. Các phương pháp nén prompt (LLMLingua) và KV pruning phát huy tác dụng rõ nét. |
| **Bin 4** | `bin4_08k_12k.jsonl` | `[8,000 – 12,000)` | **9,703** | **Attention Scaling:** Prefill scaling $O(N^2)$ rõ rệt. Sparse Attention đạt ưu thế vượt trội (1.5x – 2.5x speedup). |
| **Bin 5** | `bin5_12k_16k.jsonl` | `[12,000 – 16,000)` | **13,774** | **Memory Stress Test:** Thử thách cực hạn VRAM và băng thông bộ nhớ. Đánh giá khả năng chống OOM và độ giữ chất lượng (ROUGE-L). |

---

## 4. Thống kê chi tiết từng tệp dữ liệu

Tất cả các mẫu được trích xuất từ dữ liệu nguồn canonical LongBench (`data/longbench_200/`), tokenized theo Qwen3-4B trên toàn bộ prompt hoàn chỉnh:

| Tệp | Số mẫu | Dung lượng | Token Min | Median | Mean | Token Max | Tỷ lệ tác vụ | Phân bố Dataset nguồn |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|---|
| `bin1_00k_02k.jsonl` | 50 | 415 KB | 303 | 1,617 | 1,483.8 | 1,994 | Tóm tắt: 25, Code: 25 | `lcc`: 25, `multi_news`: 25 |
| `bin2_02k_04k.jsonl` | 50 | 817 KB | 2,010 | 2,871 | 2,963.9 | 3,970 | Tóm tắt: 26, Code: 24 | `gov_report`: 13, `lcc`: 12, `multi_news`: 12, `repobench-p`: 12, `qmsum`: 1 |
| `bin3_04k_08k.jsonl` | 50 | 1.5 MB | 4,000 | 5,823 | 5,790.6 | 7,767 | Tóm tắt: 30, Code: 20 | Cân bằng 10 mẫu/dataset: `gov_report` (10), `qmsum` (10), `multi_news` (10), `repobench-p` (10), `lcc` (10) |
| `bin4_08k_12k.jsonl` | 50 | 2.4 MB | 8,012 | 9,633 | 9,703.1 | 11,878 | Tóm tắt: 31, Code: 19 | `gov_report`: 15, `qmsum`: 14, `repobench-p`: 14, `lcc`: 5, `multi_news`: 2 |
| `bin5_12k_16k.jsonl` | 50 | 3.3 MB | 12,117 | 13,649 | 13,774.0 | 15,843 | Tóm tắt: 33, Code: 17 | `gov_report`: 15, `qmsum`: 15, `repobench-p`: 15, `multi_news`: 3, `lcc`: 2 |

*Tổng cộng: **250 mẫu** (tương đương ~8.2 MB).*

File metadata kiểm tra toàn vẹn và checksum SHA256: [`data/length_bins/manifest.json`](../data/length_bins/manifest.json).

---

## 5. Cấu trúc trường bản ghi (Record Schema)

Mỗi record trong file `.jsonl` tuân thủ chuẩn schema LongBench của dự án:

```json
{
  "id": "gov_report_3ac4f8eab6a5d29a",
  "dataset": "gov_report",
  "task_type": "summarization",
  "source_split": "test",
  "source_index": 4,
  "context": "The Congressional Research Service...",
  "input": "",
  "answers": [
    "Multiyear procurement (MYP) and block buy contracting (BBC)..."
  ],
  "reference_output": "Multiyear procurement (MYP) and block buy contracting (BBC)...",
  "input_tokens": 10697,
  "length_bin": 3,
  "length_bin_name": "bin4_08k_12k"
}
```

- Mọi mẫu trong mỗi file jsonl được sắp xếp thứ tự tăng dần theo `input_tokens`.

---

## 6. Hướng dẫn Benchmark và Chạy thực nghiệm

### 6.1 Chạy 1 baseline trên 1 phân vùng cụ thể

```bash
# Ví dụ chạy baseline eagle3 trên bin 3 (4k-8k tokens)
DATA_INPUT=data/length_bins/bin3_04k_08k.jsonl bash scripts/run.sh eagle3

# Chạy trên server B200 với đường dẫn canonical
DATA_INPUT=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/length_bins/bin5_12k_16k.jsonl bash scripts/run.sh dflash
```

### 6.2 Quét toàn bộ 5 phân vùng cho 1 baseline

```bash
DATA_DIR="/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/length_bins"
# Nếu chạy local: DATA_DIR="data/length_bins"

for bin_file in ${DATA_DIR}/bin*.jsonl; do
    echo "=========================================================="
    echo "Benchmarking ${bin_file}"
    echo "=========================================================="
    DATA_INPUT="${bin_file}" bash scripts/run.sh <baseline>
done
```

### 6.3 Tái lập (Reproducibility)

Để tạo lại tập dữ liệu này từ dữ liệu nguồn với cùng seed:

```bash
python3 scripts/build_length_bins.py \
  --source-dir data/longbench_200 \
  --output-dir data/length_bins \
  --samples-per-bin 50 \
  --seed 42
```
