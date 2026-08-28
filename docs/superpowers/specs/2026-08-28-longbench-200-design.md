# Thiết kế bộ dữ liệu LongBench canonical 1.000 mẫu

## Mục tiêu

Thay thế bộ `data/representative_100` bằng một test set cố định gồm năm task
LongBench, mỗi task 200 request. Mọi baseline và evaluator dùng cùng `id`,
context, input và reference; prompt được render theo task ở bước đánh giá.

## Phạm vi

Năm task được chốt:

| Task | Loại | Source test | Canonical |
|---|---|---:|---:|
| `gov_report` | summarization | 200 | 200 |
| `qmsum` | summarization/query | 200 | 200 |
| `multi_news` | summarization | 200 | 200 |
| `lcc` | code completion | 500 | 200 |
| `repobench-p` | code completion | 500 | 200 |

`gov_report`, `qmsum`, `multi_news` giữ nguyên toàn bộ 200 mẫu test. `lcc` và
`repobench-p` được sort theo số token của prompt canonical, chia năm bin theo
thứ tự và lấy 40 mẫu/bin bằng `random.Random(seed)`. Không lọc theo ngưỡng
token sau khi chọn vì như vậy có thể làm thay đổi số lượng canonical.

## Kiến trúc dữ liệu

Source được đọc từ thư mục local/cache qua `--source-dir`; builder không gọi
network. Mặc định có thể dùng cache LongBench đã mirror trên server. Pipeline:

```text
LongBench JSONL local
        ↓
validate source fields + stable ID
        ↓
render official task prompt + tokenize
        ↓
stratified fixed selection
        ↓
data/longbench_200/*.jsonl + manifest.json
        ↓
baseline adapter / task-aware evaluator
```

Prompt template được lưu trong `scripts/common/longbench_prompts.json`, lấy theo prompt
LongBench đang vendored trong baseline repo. Raw canonical record không lưu
prompt đã render để tránh gắn dataset với một baseline; builder vẫn dùng template
để tính `input_tokens` đúng theo input thực tế.

## Canonical schema

Mỗi dòng trong `data/longbench_200/<dataset>.jsonl` là JSON object với các field
bắt buộc:

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

- `id`: ổn định theo dataset, source index và nội dung; được dùng để join output.
- `context`/`input`: giữ nguyên semantics LongBench.
- `answers`: giữ toàn bộ reference gốc.
- `reference_output`: reference đầu tiên, dùng làm field chung cho evaluator.
- `task_type`: `summarization` hoặc `code_completion`.
- `input_tokens`: số token của prompt template chính thức, không phải số từ.
- `length_bin`: `null` cho ba task giữ toàn bộ; giá trị `0..4` cho hai task stratified.

Các field tùy chọn của LongBench như `language` và `all_classes` được giữ trong
`metadata` để schema gốc không bị mất nhưng không làm thay đổi interface chung.

## Manifest và reproducibility

`manifest.json` ghi `schema_version`, `seed`, tokenizer reference, prompt config
hash, source directory, source count, selected count, length statistics, danh
sách ID theo dataset và SHA-256 của từng output JSONL. Manifest là artifact bắt
buộc để kiểm tra mọi baseline chạy cùng request set.

Builder ghi ra thư mục mới và từ chối overwrite nếu output đã tồn tại, trừ khi
có `--force`. Có checkpoint manifest tạm sau mỗi dataset và progress logging;
chạy lại với cùng tham số có thể tiếp tục/ghi sang thư mục khác mà không sửa
source.

## Đánh giá task-aware

- Ba task summarization dùng ROUGE-1/2/L như pipeline hiện tại.
- Hai task code completion dùng normalized exact match và edit similarity trên
  phần code sinh ra; không dùng ROUGE.
- Speed metrics (`input_tokens`, retained tokens, TTFT/E2E...) vẫn dùng schema
  output chung cho cả hai task.

## Tương thích runner

Thư mục canonical có tên file `<dataset>.jsonl`, không dùng hậu tố
`_representative`. Runner/collector mới sẽ nhận data directory configurable,
discover file theo manifest và join reference bằng `id`. Các adapter legacy có
thể đọc alias `document= context` và `reference= reference_output` ở lớp chuyển
đổi, không sửa raw canonical.

## Kiểm thử và rollout

1. Test fixture 100 dòng (20/task) kiểm tra schema, ID, render prompt, token
   length và sampling reproducibility; test phải fail trước implementation.
2. Build small-scale từ source cache, phân tích count, missing, duplicate,
   length bins và spot-check cả summarization/code.
3. Chỉ sau khi người dùng duyệt phân tích small-scale mới build đủ 1.000 mẫu.
4. Validate full artifact, dry-run runner/collector và quét stale reference
   trước khi đổi default benchmark khỏi `representative_100`.
