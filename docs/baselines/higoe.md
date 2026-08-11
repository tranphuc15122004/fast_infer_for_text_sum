# HiGOE

Evidence/proposition-aware semantic selection (graph + GNN + retriever) cho
long-document summarization. Dựa trên `externals/HiGOE`.

## Env & cài đặt

- Env: **`envs/legacy`** — dùng chung với FastKV/RocketKV/GemFilter/SpecExtend (torch 2.4.1 + dgl 2.5.0 vendored + langchain 0.2.10).
- `uv sync --project envs/legacy --locked`

## Chạy smoke (không cần dataset/API key)

```bash
bash scripts/run.sh higoe
```
Verify: mọi module HiGOE import được + Contriever round-trip (dummy docs, CPU-ok).

## Full pipeline (cần chuẩn bị)

Thứ tự theo README repo: `graph_construction.py → knowledge_synthesizer_ppr.py →
training_preparation.py → train (lưu ý repo thiếu train.py, chỉ có train_lossnew.py)
→ eval.py → sum_eval.py`, mỗi bước `--cuda 0 --dataset <qmsum|wcep|booksum|govreport|squality>`.

Yêu cầu:
- **Dataset**: QMSum (Yale-LILY), WCEP-10 (`ccdv/WCEP-10`), BookSum
  (`kmfoda/booksum`), GovReport (`ccdv/govreport-summarization`), SQuALITY (nyu-mll)
  → đặt đúng đường dẫn cố định mà code đọc (`./data/SQuALITY/data/v1-3/txt/test.jsonl`, ...).
- **LLM judge**: bước graph construction + knowledge synthesis gọi
  `get_llm_response_via_api` (openai==0.28) — cần API key cấu hình trong `utils.py`,
  hoặc chuyển sang `get_llm_response_via_local` (model Llama-2-7b-chat, fp16).
- **Retriever**: `facebook/contriever` (tự tải).

## Output

Smoke: `outputs/higoe_smoke.jsonl` (imports_ok, retrieval_ok).

## Troubleshooting

- Repo không phải package — chạy từ thư mục `externals/HiGOE` (wrapper đã cd).
- Stack hiện đại có thể lệch nhẹ API với code gốc (torch 1.12); nếu full pipeline
  lỗi do version, cần môi trường cũ trên máy chuyên dụng.
