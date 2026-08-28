# Plug-and-play dữ liệu cho các baseline

## Bộ canonical LongBench 1.000 mẫu

Bộ test chính thức của repo nằm ở `data/longbench_200/`, gồm năm file JSONL:

| File | Loại | Số mẫu |
|---|---|---:|
| `gov_report.jsonl` | summarization | 200 |
| `qmsum.jsonl` | summarization/query | 200 |
| `multi_news.jsonl` | summarization | 200 |
| `lcc.jsonl` | code completion | 200 |
| `repobench-p.jsonl` | code completion | 200 |

`manifest.json` lưu seed, tokenizer, source count, token statistics, danh sách
ID và checksum của từng file. Nguồn LongBench được mirror local; builder không
tải internet.

Schema chung của mỗi record:

```json
{
  "id": "lcc_4e5c...",
  "dataset": "lcc",
  "source_split": "test",
  "source_index": 37,
  "task_type": "code_completion",
  "context": "...",
  "input": "...",
  "answers": ["..."],
  "reference_output": "...",
  "input_tokens": 1234,
  "length_bin": 3
}
```

Prompt chính thức được render từ `scripts/common/longbench_prompts.json` ở lúc load;
raw record không gắn chết prompt của một baseline. `input_tokens` là độ dài
prompt sau render bằng tokenizer Llama 3.1 đã dùng để build. LCC và RepoBench-P
được chọn 40 mẫu trong mỗi 5 bin độ dài.

Build lại từ mirror local:

```bash
python scripts/build_longbench_200.py \
  --source-dir /path/to/LongBench \
  --tokenizer /path/to/Meta-Llama-3.1-8B-Instruct \
  --output-dir data/longbench_200 \
  --seed 42
python scripts/validate_longbench_200.py \
  --data-dir data/longbench_200 --expected-count 200
```

Để xem nhanh cấu trúc và nội dung mẫu của cả năm dataset:

```bash
python scripts/show_longbench_200.py --samples 1
```

Có thể giới hạn dataset và độ dài preview, ví dụ:

```bash
python scripts/show_longbench_200.py \
  --dataset lcc --dataset repobench-p \
  --samples 2 --context-chars 300 --field-chars 160
```

Khi đánh giá, dùng `DATA_INPUT=data/longbench_200/<dataset>.jsonl`. Loader
chung tự render prompt từ `dataset/context/input`; reference chung là
`reference_output`. Ba task summarization dùng ROUGE/BLEU; hai task code
completion dùng `code_exact_match` và `code_edit_similarity`, không dùng ROUGE.

## Dữ liệu tùy biến

Để chạy một baseline với **dữ liệu của bạn**, chỉ cần bỏ file jsonl vào thư mục
này và trỏ `DATA_FILE` trong `config/<baseline>.env` tới file đó:

```bash
# ví dụ
DATA_FILE="data/user_prompts.jsonl"
```

## Định dạng file (jsonl, mỗi dòng 1 record JSON)

```json
{"id": 0, "prompt": "...", "answer": "..."}          // khuyến nghị
{"id": 0, "question": "...", "answer": "..."}
{"id": 1, "text": "...", "keyword": "some_entity"}   // dạng tài liệu/summarization
{"id": 2, "turns": ["user prompt", "assistant reply"]} // dạng chat (EAGLE-style)
```

Các trường được loader chung (`scripts/common/data_loader.py`) tự nhận diện:
`prompt` → `question` → `instruction` → `text` → `turns[0]`. Trường tuỳ chọn:
`answer` (đáp án tham chiếu), `keyword` (entity kỳ vọng còn sống sót — dùng cho
verify retention của nhóm compression/KV).

## Reference cho ROUGE

Khi record có một trong các trường `reference` / `summary` / `answer`, script
infer sẽ tự tính **ROUGE-1/2/L** (F1) của summary sinh ra so với reference và
ghi `rouge1/rouge2/rougeL` vào từng record + `mean_rouge*` vào bản `summary`
cuối file. Triển khai: `scripts/common/rouge.py` (pure-Python, không cần cài
package).

```json
{"id": 0, "text": "<long doc>", "reference": "<tóm tắt mẫu>"}   // summarization
{"id": 0, "prompt": "...", "answer": "..."}                      // QA cũng tính ROUGE
```

Nếu không có reference, các key ROUGE đơn giản không xuất hiện trong output.

## Trạng thái hỗ trợ theo baseline

| Baseline | Đọc `DATA_FILE`? | Ghi chú |
|---|---|---|
| LLMLingua | ✅ | nén prompt rồi target sinh; dùng `prompt`/`text` |
| FastKV | ✅ | mỗi record 1 lần generate (Llama/Mistral) |
| GemFilter | ✅ | mỗi record 1 lần generate (model theo config) |
| speculative_prefill | ✅ | batch qua vLLM |
| MInference | ✅ | mỗi record 1 lần generate |
| EAGLE-3 | ✅ | dùng `--question-file` (định dạng `turns`) |
| RocketKV | ⚠️ | smoke = kernel; full pipeline dùng LongBench config riêng |
| MagicDec | ⚠️ | benchmark dùng dataset pg19 nội bộ; muốn data riêng phải sửa repo |
| SpecExtend | ✅ | đọc jsonl có trường `text` (format summarization) |
| LongSpec | ⚠️ | cần longbench jsonl đã tiền xử lý + GPU 80GB |
| HiGOE | ❌ | pipeline riêng theo dataset folder (QMSum/GovReport...) |
| DFlash | ❌ | chỉ hỗ trợ dataset builtin (gsm8k/math500/humaneval/mbpp/mt-bench) |

## Chuẩn hoá output

Mọi script ghi kết quả vào `outputs/<baseline>_*.jsonl` theo schema thống nhất
(`externals/baseline_repo_guide.md` §13): input_tokens, retained_tokens,
output_tokens, selector_latency_ms, ttft/e2e_ms, throughput_tok_s, ... + bản
`summary` cuối file, kèm verify PASS/FAIL.

## Trích tập mẫu đại diện

`data/representative_100/` là bộ legacy 4 dataset, chỉ dùng để tái hiện các run
cũ. Sau khi có các file normalized trong `data/normalized/`, có thể trích khoảng
100 mẫu cho mỗi dataset bằng:

```bash
python data/extract_representative_samples.py \
  --normalized-dir data/normalized \
  --output-dir data/representative_100 \
  --samples-per-dataset 100
```

Script chọn các điểm cách đều sau khi xếp theo độ dài `document`, nên bao phủ
được tài liệu ngắn, trung bình và dài mà không phụ thuộc random seed. Mặc định
độ dài là số từ và không cần tải model/tokenizer. Kết quả gồm một file riêng
cho từng dataset và `manifest.json`. Có thể dùng
`--length-metric tokens --tokenizer Qwen/Qwen3-4B`
để chọn theo token length của Qwen. Record có `document` hoặc `reference` rỗng
sẽ được bỏ qua và số lượng được ghi trong `manifest.json`.
