# Plug-and-play dữ liệu cho các baseline

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
