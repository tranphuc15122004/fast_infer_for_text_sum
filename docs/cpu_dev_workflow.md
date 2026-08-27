# Dev/Debug trên CPU — máy tuantb@teslaT4

> Mục đích: hướng dẫn AI và developer dùng máy T4 này làm môi trường **dev trên CPU**
> cho stack cu130, và debug script baseline trước khi chạy số liệu thật trên server cu13.

## 1. Bối cảnh (tại sao chỉ CPU)

| Hạng mục | Giá trị | Ý nghĩa |
|---|---|---|
| GPU | Tesla T4 15GB | chỉ dùng cho smoke khi driver phù hợp |
| Driver | 550.163.01 | max CUDA 12.4 |
| Stack đã cài | torch 2.11.0+cu130, vllm 0.24.0 | **cần driver 570+** → không chạy GPU được |
| Quyền | `tuantb` **không có sudo** | không nâng driver, không cài package hệ thống |
| `torch.cuda.is_available()` | luôn `False` | đừng cố sửa GPU trên máy này |

→ Quyết định (2026-08-27): **giữ stack cu130, dev trên CPU**. Số liệu benchmark
chạy trên server cu13 để giữ tính công bằng (không hạ cu124 vì kéo cascade
vllm/transformers cũ, lệch version với server).

`.venv` ở máy T4 chỉ mô phỏng dependency/API của server B200. Production B200
không activate `.venv`; chạy `python3` trực tiếp sau khi đã có đủ wheel/cache
server. Có thể mô phỏng profile bằng:

```bash
FAST_INFER_PYTHON="$PWD/.venv/bin/python" \
  bash scripts/run_b200_smoke.sh --preflight-only
```

## 2. Môi trường

- Venv: `.venv/` (Python 3.12.13), đã cài **`requirements.local.txt`** (không phải
  `requirements.txt`).
- `requirements.local.txt` = bản sao `requirements.txt` đã:
  - Thay `vllm @ file://...` → `vllm==0.24.0` (PyPI), torch/audio/vision bỏ `+cu130`.
  - `nixl-cu13` hạ 1.3.0 → 1.2.0 (khớp `nixl==1.2.0`).
  - Comment các package không có trên PyPI / cần hệ thống: `deep_ep`, `eviseq`,
    `flashinfer-jit-cache(+cu130)`, `pyrouge`, `python-apt(+ubuntu4.1)`,
    `dbus-python`, `PyGObject`, `mooncake-transfer-engine` (cp310-only),
    `flash_attn` (sdist build fail: nvcc 12.4 vs torch cu130).
  - `deepspeed==0.19.3` cài OK từ sdist (pure wheel).
- Preflight: `FAST_INFER_VENV="$PWD/.venv" "$PWD/.venv/bin/python" scripts/check_shared_env.py`
  → PASS trừ `flash_attn` (không cài được trên máy này).
- uv tạo venv **không có pip** → `pip list` = 0 dòng là bình thường; dùng `uv pip list --python .venv/bin/python`.

## 3. Chạy baseline trên CPU

Cách chung (baseline có fallback CPU):
```bash
CUDA_VISIBLE_DEVICES="" DEVICE=cpu SMOKE=1 bash scripts/run_<baseline>.sh
```

- `llmlingua`: hỗ trợ CPU đầy đủ (`infer_llmlingua.py` tự fallback cpu khi CUDA off).
  Đã verify: `bash scripts/run_llmlingua.sh` chạy 2 sample ~3 phút/sample (Qwen2.5-1.5B).
- Các baseline khác: kiểm tra script có đọc `--device`/`DEVICE` không; nếu hardcode
  `.to("cuda")` thì chưa chạy được CPU (ghi nhận và để dành cho server).
- `CUDA_VISIBLE_DEVICES=""` quan trọng: tránh torch cu130 init CUDA với driver cũ
  (warning ồn + một số package như bitsandbytes có thể treo).

## 4. Bug đã sửa — đừng tái lập

### 4.1 Check Python 3.12 (runtime.sh + setup_venv.sh)
BẪY: `raise SystemExit("...") if cond else None` parse thành `raise (X if cond else None)`.
Khi version ĐÚNG là 3.12 → biểu thức = `None` → `raise None` → `TypeError: exceptions
must derive from BaseException` → check luôn fail.
ĐÚNG: `sys.exit(1) if cond else None` (khi đúng version, biểu thức = None, không sao).

### 4.2 transformers 5.x apply_chat_template
Trong transformers 5.x, `apply_chat_template(..., tokenize=True, return_tensors="pt")`
mặc định `return_dict=True` → trả **`BatchEncoding`** (không còn tensor thô như 4.x).
→ `.shape[1]`/`.to(device)` trả về object lạ → AttributeError.
SỬA: thêm `return_dict=False` (tương thích cả 4.x và 5.x):
```python
ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", return_dict=False,
).to(device)
```
Đã sửa ở: `infer_llmlingua.py` (2 chỗ), `eagle3_infer_qwen3.py`, `infer_flexprefill.py`.
Khi gặp `AttributeError: 'shape'`/`__getattr__` từ tokenization_utils_base → đây là lỗi này.

## 5. Checklist debug script mới / sửa baseline

1. Chạy preflight trước: `check_shared_env.py` (import-only, không tải model).
2. Mở rộng `CUDA_VISIBLE_DEVICES=""` khi chạy để tránh CUDA init.
3. Nếu script dùng `apply_chat_template` → kiểm tra `return_dict=False`.
4. Nếu script dùng `.to("cuda")` hardcode → baseline cần GPU, để cho server.
5. Kiểm tra model cache: `ls ~/.cache/huggingface/hub` (ưu tiên dùng snapshot local,
   máy có internet nhưng gated model cần token).
6. Chạy `--smoke` với `MAX_SAMPLES=1..2`, `MAX_NEW_TOKENS<=32` để nhanh.
7. Output phải theo schema `io_util.JsonlWriter` (kết thúc bằng record summary).

## 6. Giới hạn (đừng lãng phí thời gian)

- `flash_attn` chưa cài → baseline cần flash-attn không chạy được (kể cả CPU nếu
  script import thẳng `flash_attn`).
- vllm 0.24.0 kernels cần GPU → chạy vllm inference trên máy này không khả thi.
- `setup_venv.sh --check` luôn báo lỗi thiếu local requirement sources
  (`deep_ep`/`eviseq`/vllm wheel) trên máy này — bình thường, không phải lỗi venv.
- Không cài package cần `sudo` (libdbus-1-dev, libcairo2-dev...) — máy không có quyền.
