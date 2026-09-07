# EAGLE-3

Learned speculative decoding (EAGLE-3, Qwen3) — lossless under target
verification. Dựa trên `externals/EAGLE`.

## Env & cài đặt

- Env: **venv chung** tại `.venv` (Python 3.12, dependency từ `requirements.txt`).
- `bash scripts/setup_venv.sh --offline`.

## Model

| Vai trò | Model | Ghi chú |
|---|---|---|
| Base | `MODEL_TARGET` | đặt snapshot local trong master nếu server offline |
| Draft | `MODEL_EAGLE_DRAFT` | đặt snapshot local trong master nếu server offline |

## Chạy smoke / thật

```bash
bash scripts/run.sh eagle3
```

Cấu hình trong master: `EAGLE_BENCHMARK`, `EAGLE_QUESTION_BEGIN` /
`EAGLE_QUESTION_END`, `EAGLE_MAX_NEW_TOKENS`, `EAGLE_TOTAL_TOKENS` /
`EAGLE_DEPTH` / `EAGLE_TOP_K`.

Phân biệt hai loại token:

- `EAGLE_MAX_NEW_TOKENS` là ngân sách output của mỗi lần generate.
- `EAGLE_TOTAL_TOKENS` là số node của speculative tree, không phải độ dài
  output. Với `depth=D` và `top_k=K`, giá trị tối đa hợp lệ là
  `1 + K + D * K * K`. Nếu cấu hình lớn hơn, launcher tự hạ về giới hạn
  này và in cảnh báo; ví dụ `total_token=10000` với `depth=8`, `top_k=4` chỉ
  tạo được tối đa `133` node và là nguyên nhân của lỗi
  `selected index k out of range`.

## Dữ liệu của bạn (plug-and-play)

Set `DATA_FILE` trong env (ghi đè question file). Định dạng bắt buộc (EAGLE chat):

```json
{"id": 0, "turns": ["user prompt"]}
```

```bash
DATA_FILE="data/user_prompts.jsonl" bash scripts/run.sh eagle3
```

## Output

`outputs/eagle3_qwen3_qa.jsonl` — per-question: new_tokens, tree_steps,
accept_length, eagle/naive tok/s, speedup + summary cuối.

## Troubleshooting

- Base/draft config mismatch → wrapper tự kiểm tra hidden_size/heads/vocab trước khi load.
- GPU RAM: Qwen3-4B fp16 ~9GB; cần GPU ≥ 12GB.
