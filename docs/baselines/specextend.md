# SpecExtend

Drop-in long-context speculative decoding (target-guided, training-free) cho
long-document summarization. Dựa trên `externals/SpecExtend`.

## Env & cài đặt

- Env: **venv chung** tại `.venv` (Python 3.12, dependency từ `requirements.txt`).
- `bash scripts/setup_venv.sh --offline`
- Full cần wheel `flash-attn` tương thích CUDA/GPU có sẵn trong wheelhouse local.

## Model

| Vai trò | Model |
|---|---|
| Target | `meta-llama/Meta-Llama-3.1-8B-Instruct` (`llama3_1_8b`) |
| Draft (EAGLE-3) | `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` |

Checkpoint EAGLE-3 phải đi qua `run_eagle.py`; không thể dùng như draft classic.
Smoke giới hạn 1 mẫu, 512 token input và 16 token output, nhưng vẫn dùng đúng
cặp model trong paper. Cần GPU đủ cho target 8B + EAGLE-3; nếu thiếu VRAM,
smoke sẽ fail rõ ràng thay vì tự đổi sang TinyLlama/Vicuna.

Lưu ý khi diễn giải paper: Figure 1 dùng Llama-3.1-8B-Instruct + EAGLE-3 để
đo hiệu năng/bộ nhớ; kết quả headline 3.86x ở phần reasoning là
DeepSeek-R1-Distill-Llama-8B + EAGLE-3 trên AIME-24. Các bảng long-document
GovReport/PG-19/BookSum dùng Vicuna-7B + Vicuna-68M.

Implementation note: nhánh Llama-3.1 dùng loader EAGLE-3 chính thức và ghép
SpecExtend target hybrid-tree attention/RoPE compatibility. Đây là đường triển
khai tương thích để chạy đúng cặp model; chưa tuyên bố tái lập 3.86x hoặc toàn
bộ CMR benchmark của paper cho đến khi chạy trên GPU phù hợp và đối chiếu đủ
metric.

## Chạy smoke / thật

```bash
bash scripts/run.sh specextend   # smoke: Llama-3.1-8B + EAGLE-3
```

Cấu hình trong master: `SPECEXTEND_SCRIPT=run_eagle.py`,
`SPECEXTEND_MODEL_NAME`, `MODEL_TARGET`, `MODEL_EAGLE_DRAFT`,
`SPECEXTEND_DATA_FILE` (jsonl có trường `text`), `SPECEXTEND_MAX_SAMPLES`,
`SPECEXTEND_MAX_NEW_TOKENS`, `SPECEXTEND_MAX_INPUT_TOKENS`,
`SPECEXTEND_ENABLED`.

Data kèm sẵn: `externals/SpecExtend/specextend/data/govreport/govreport_{512,1K,2K,4K,8K,16K}.jsonl`
(đã nhúng prompt summarize).

## Dữ liệu của bạn

File jsonl định dạng `{"id":.., "text": "..."}` (prompt summarize sẽ được repo
tự gắn). Đặt vào `data/` và trỏ `INPUT_FILE="data/<file>.jsonl"`.

## Output

`outputs/specextend_smoke.jsonl` — returncode + số dòng summary sinh được.

## Troubleshooting

- Nếu OOM: giảm `MAX_GEN_LEN`/`MAX_INPUT_TOKENS`, hoặc chuyển GPU lớn. Không
  thay EAGLE-3 bằng một LLM classic nếu mục tiêu là tái hiện paper.
- Llama-3.1 + EAGLE-3 không chạy trên T4 16 GiB theo preflight VRAM; dùng GPU
  >=20 GiB. Với classic Vicuna path, T4 có thể dùng SDPA/PyTorch fallback nếu
  không có FlashAttention-2; tree kernel vẫn cần kiểm chứng riêng nếu backend
  Triton không compile được.
- `eval_classic.py`/`eval_eagle.py` chạy sweep nhiều độ dài (dành cho GPU lớn).
