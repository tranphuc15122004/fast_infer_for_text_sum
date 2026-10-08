# Bộ dữ liệu Benchmark phân vùng theo độ dài (Length Bins Benchmark)

Tập dữ liệu này gồm **250 mẫu** được trích xuất từ LongBench gốc (`data/longbench_200/`), chia thành **5 phân vùng độ dài độc lập (50 mẫu/khoảng)** từ 0 đến 16,000 tokens. Mỗi khoảng đại diện cho một ngưỡng hoạt động phần cứng và đặc tính thuật toán suy luận khác nhau.

---

## 1. Đường dẫn lưu trữ

- **Đường dẫn canonical trên server sản xuất:**
  ```text
  /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/length_bins/
  ```
- **Đường dẫn repo / local:**
  ```text
  data/length_bins/
  ```

---

## 2. Ý nghĩa kỹ thuật của 5 phân vùng độ dài

| Phân vùng | Tệp dữ liệu | Khoảng Token | Token TB | Ý nghĩa đo lường & Hardware Bottleneck |
|:---:|---|:---:|:---:|---|
| **Bin 1** | [`bin1_00k_02k.jsonl`](bin1_00k_02k.jsonl) | `[0 – 2,000)` | **1,484** | **Overhead Check:** Prefill tức thì (< 50ms). Đo chi phí phụ của speculative decoding (draft overhead) và xem các cơ chế nén có bị suy hao chất lượng không đáng có không. |
| **Bin 2** | [`bin2_02k_04k.jsonl`](bin2_02k_04k.jsonl) | `[2,000 – 4,000)` | **2,964** | **Transition Zone:** Vùng chuyển tiếp, prefill bắt đầu chiếm tỷ trọng rõ rệt. Xác định ngưỡng bắt đầu có lãi tốc độ (speedup > 1.0x). |
| **Bin 3** | [`bin3_04k_08k.jsonl`](bin3_04k_08k.jsonl) | `[4,000 – 8,000)` | **5,791** | **Compression Zone:** Ngữ cảnh tiêu chuẩn. Các phương pháp nén ngữ cảnh (LLMLingua, Semantic Selection) và KV Cache (FastKV, GemFilter) thể hiện rõ hiệu năng. |
| **Bin 4** | [`bin4_08k_12k.jsonl`](bin4_08k_12k.jsonl) | `[8,000 – 12,000)` | **9,703** | **Attention Scaling:** Prefill scaling $O(N^2)$ thể hiện mạnh. Sparse Attention (MInference, FlexPrefill, SpecPrefill) đạt ưu thế vượt trội. |
| **Bin 5** | [`bin5_12k_16k.jsonl`](bin5_12k_16k.jsonl) | `[12,000 – 16,000)` | **13,774** | **Memory Stress Test:** Thử thách cực hạn VRAM và băng thông bộ nhớ. Đánh giá khả năng chống OOM và duy trì chất lượng tóm tắt (ROUGE-L). |

---

## 3. Thống kê chi tiết từng tệp dữ liệu

| Tệp | Số mẫu | Dung lượng | Min | Median | Max | Phân bố tác vụ | Phân bố Dataset nguồn |
|---|:---:|:---:|:---:|:---:|:---:|---|---|
| `bin1_00k_02k.jsonl` | 50 | 415 KB | 303 | 1,617 | 1,994 | Summarization: 25, Code: 25 | `lcc`: 25, `multi_news`: 25 |
| `bin2_02k_04k.jsonl` | 50 | 817 KB | 2,010 | 2,871 | 3,970 | Summarization: 26, Code: 24 | `gov_report`: 13, `lcc`: 12, `multi_news`: 12, `repobench-p`: 12, `qmsum`: 1 |
| `bin3_04k_08k.jsonl` | 50 | 1.5 MB | 4,000 | 5,823 | 7,767 | Summarization: 30, Code: 20 | Đều 10 mẫu/dataset: `gov_report` (10), `qmsum` (10), `multi_news` (10), `repobench-p` (10), `lcc` (10) |
| `bin4_08k_12k.jsonl` | 50 | 2.4 MB | 8,012 | 9,633 | 11,878 | Summarization: 31, Code: 19 | `gov_report`: 15, `qmsum`: 14, `repobench-p`: 14, `lcc`: 5, `multi_news`: 2 |
| `bin5_12k_16k.jsonl` | 50 | 3.3 MB | 12,117 | 13,649 | 15,843 | Summarization: 33, Code: 17 | `gov_report`: 15, `qmsum`: 15, `repobench-p`: 15, `multi_news`: 3, `lcc`: 2 |

Metadata chuẩn, hash SHA256 từng file và danh sách ID được lưu tại [`manifest.json`](manifest.json).

---

## 4. Cấu trúc trường trong JSONL (Schema)

Mỗi dòng là một đối tượng JSON tuân thủ schema LongBench canonical:

```json
{
  "id": "gov_report_3ac4f8eab6a5d29a",
  "dataset": "gov_report",
  "task_type": "summarization",
  "source_split": "test",
  "source_index": 4,
  "context": "Nội dung văn bản dài...",
  "input": "",
  "answers": ["Tóm tắt chuẩn tham chiếu..."],
  "reference_output": "Tóm tắt chuẩn tham chiếu...",
  "input_tokens": 10697,
  "length_bin": 3,
  "length_bin_name": "bin4_08k_12k"
}
```

- Các mẫu trong mỗi file đã được sắp xếp tăng dần theo `input_tokens` để khi chạy `--limit` nhỏ vẫn trải đều từ biên dưới lên biên trên.

---

## 5. Hướng dẫn sử dụng & Tái lập

### Chạy kiểm thử một baseline với 1 phân vùng độ dài

```bash
# Chạy local
DATA_INPUT=data/length_bins/bin3_04k_08k.jsonl bash scripts/run.sh eagle3

# Chạy trên server với đường dẫn canonical
DATA_INPUT=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/length_bins/bin5_12k_16k.jsonl bash scripts/run.sh dflash
```

### Chạy toàn bộ ma trận 5 phân vùng cho 1 baseline

```bash
for bin_file in data/length_bins/bin*.jsonl; do
    echo "=== Running baseline on ${bin_file} ==="
    DATA_INPUT="${bin_file}" bash scripts/run.sh <baseline>
done
```

### Tái lập / Tạo lại tập dữ liệu

Script tạo dữ liệu nằm tại [`scripts/build_length_bins.py`](../../scripts/build_length_bins.py):

```bash
python3 scripts/build_length_bins.py \
  --source-dir data/longbench_200 \
  --output-dir data/length_bins \
  --samples-per-bin 50 \
  --seed 42
```
