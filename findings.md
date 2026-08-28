# Findings & Decisions

## Requirements
- Xây bộ test mới thay cho `data/representative_100`.
- Các baseline phải dùng cùng một tập request cố định, có ID canonical.
- Chuẩn hóa dữ liệu về một format chung để thuận tiện đánh giá.
- Theo tài liệu người dùng cung cấp: 5 task LongBench (`gov_report`, `qmsum`, `multi_news`, `lcc`, `repobench-p`), mỗi task 200 mẫu, tổng 1.000.
- `lcc` và `repobench-p` lấy stratified theo input-token length: 5 bins × 40 mẫu.
- Không cố định prompt riêng của một baseline trong raw dataset; prompt renderer xử lý sau.

## Research Findings
- Repo hiện có `data/normalized/` và `data/representative_100/` cho 4 task summarization, mỗi task 100 mẫu; runner `scripts/run_representative_100.sh` hard-code thư mục và hậu tố `_representative`.
- Loader chung `scripts/common/data_loader.py` nhận `prompt/question/instruction/document/text/turns`, và reference qua `reference/summary/answer`; chưa có canonical LongBench `context/input/reference_output`.
- Collector `scripts/collect_metrics.py` cũng hard-code pattern `_representative.jsonl` và `data/representative_100`.
- Existing runner tự tạo prompt `Summarize the following document...`; cách này chỉ phù hợp summarization, không phù hợp LCC/RepoBench-P.
- AGENTS yêu cầu server offline, không tự tải model/dataset; local wheel/model/cache phải có sẵn.
- Source LongBench đã có trong `/home/tuantb/.cache/huggingface/datasets/fast_infer_text_sum/FastKV/data/LongBench`: đúng 200 dòng cho `gov_report`, `qmsum`, `multi_news` và 500 dòng cho `lcc`, `repobench-p`.
- Prompt config chính thức cho cả 5 task đã vendored ở `externals/RocketKV/eval/longbench_utils/config/dataset2prompt.json` và `externals/GemFilter/eval/LongBench/config/dataset2prompt.json`.
- Snapshot tokenizer `meta-llama/Meta-Llama-3.1-8B-Instruct` có trong HF cache; builder có thể chạy offline với đường dẫn snapshot.

## Proposed Canonical Record
Mỗi record JSONL sẽ có các field ổn định:

```json
{
  "id": "longbench_lcc_<stable-id>",
  "dataset": "lcc",
  "source_split": "test",
  "source_index": 12,
  "task_type": "code_completion",
  "context": "...",
  "input": "...",
  "answers": ["..."],
  "reference_output": "...",
  "input_tokens": 1234,
  "length_bin": 2
}
```

`reference_output` là bản chuẩn hóa từ `answers[0]` (nếu có), còn `answers` được giữ nguyên để evaluator task-specific dùng exact-match/edit similarity hoặc nhiều reference. Các dataset summarization dùng `task_type: summarization`; LCC/RepoBench-P dùng `code_completion`.

## Open Questions
- Tokenizer mặc định triển khai là snapshot local của Meta-Llama-3.1-8B-Instruct; CLI vẫn cho phép override.
- Rollout runner sẽ giữ task-aware metric; baseline không có adapter code-completion không được tính điểm chất lượng cho LCC/RepoBench-P.
- User đã phê duyệt full-scale trong lượt 2026-08-28; bộ canonical chính thức đã được ghi tại `data/longbench_200`.

## Small-scale Analysis (2026-08-28)
- Output tạm: `/tmp/fast_infer_longbench_small`.
- Kết quả: 5 file × 20 record = 100 record; validator pass; 100 ID unique; tất cả record có schema chung và reference không rỗng.
- Bin distribution: `gov_report`, `qmsum`, `multi_news`, `lcc`, `repobench-p` đều có 4 record/bin trong bản partial 20; LCC/RepoBench-P giữ đúng 5 bin.
- Token stats (Llama 3.1 prompt tokens): GovReport mean 9,547.55 (min 3,493, max 18,608); QMSum mean 13,616.20 (5,962–25,420); Multi-News mean 2,373.40 (791–5,175); LCC mean 2,719.95 (1,366–5,792); RepoBench-P mean 9,974.50 (3,385–19,587).
- Spot-check: summarization rows render official summary/query prompts; LCC/RepoBench-P rows render code-completion prompts and preserve code references.
- Determinism ở checkpoint small-scale: focused sampling test pass; một lệnh rebuild dự phòng dừng trước khi ghi RepoBench-P.

## Full-scale Verification (2026-08-28)
- `data/longbench_200/` có 5 JSONL và `manifest.json`, tổng 1.000 record.
- `validate_longbench_200.py --expected-count 200` pass; mọi ID unique, schema đầy đủ, checksum manifest đúng.
- LCC và RepoBench-P đều có length-bin distribution `40/40/40/40/40`.
- Rebuild độc lập tại `/tmp/fast_infer_longbench_full_repeat_v2` với cùng seed và tokenizer tạo output byte-identical cho cả 5 JSONL và manifest.
- `pytest -q tests/test_longbench_dataset.py` pass; toàn repo vẫn có collection errors từ test vendored thiếu optional dependencies.

## Resources
- `data/README.md`
- `scripts/common/data_loader.py`
- `scripts/run_representative_100.sh`
- `scripts/collect_metrics.py`
- `/home/tuantb/.codex/attachments/99b12101-d46d-4dfb-9003-b1d80d684e78/pasted-text.txt`
- `AGENTS.md`
