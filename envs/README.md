# Environment layout for baseline verification

Repo quản lý một số **uv venv** độc lập trong `envs/<group>/` (mỗi nhóm có
`pyproject.toml` + `uv.lock` đã commit). Lý do không gộp thành 1 env duy nhất:
một số baseline bị **khóa cứng** bởi dependency đã biên dịch/khớp chính xác
version (vllm, triton, flashinfer) hoặc `transformers` quá cũ (MagicDec 4.36).

```
envs/
  legacy/      FastKV, RocketKV, GemFilter, SpecExtend, HiGOE
               (transformers 4.45.2, torch 2.4.1+cu124, numpy 1.26)
  specprefill/ speculative_prefill, MInference
               (vllm 0.6.3.post1, transformers 4.50.2, torch 2.4.0)
  magicdec/    MagicDec            (transformers 4.36.2, flashinfer)
  longspec/    LongSpec            (transformers 4.46.3, triton 3.1.0, liger 0.3.1)
core = project gốc (pyproject.toml ở root): EAGLE, dflash, LLMLingua
       (transformers 4.57.1, torch cu126)
```

## Vì sao `envs/legacy` gộp được 5 baseline?

FastKV/RocketKV (pin gốc 4.45.x), GemFilter (pin gốc 4.43.3), SpecExtend (pin
gốc 4.41.0), HiGOE (không pin) đều dùng API `transformers` còn tồn tại ở
4.45.2 (`LlamaAttention`, `LLAMA_ATTENTION_CLASSES`, `AttentionMaskConverter`,
`cache_utils`, `modeling_attn_mask_utils._prepare_4d_causal_attention_mask`...)
nên gộp chung một stack: `transformers==4.45.2` + `torch==2.4.1` + `numpy==1.26`.
Các pin mâu thuẫn đã được thả lỏng: SpecExtend `tokenizers 0.19.1 → 0.20.x`,
`protobuf 3.19.0 → 4.25.1`; HiGOE `dgl` dùng wheel **vendored** tại
`envs/legacy/wheels/` (PyPI chỉ có wheel Windows, index riêng của dgl chặn uv).

## Cách dùng trên máy mới

Yêu cầu: `uv` (>= 0.4), Python 3.11, NVIDIA driver tương thích CUDA.

```bash
# 1) Cài tất cả env từ lock files (tái lập chính xác)
bash scripts/setup_envs.sh
#    - Trên GPU lớn (A100/H100...) thêm flash-attn:
#      EXTRA_FLASH=1 bash scripts/setup_envs.sh

# 2) Chạy smoke test một baseline
bash scripts/run.sh llmlingua        # ví dụ: baseline chạy CPU
bash scripts/run.sh fastkv --smoke   # baseline cần GPU
```

Mỗi wrapper (`scripts/run_<baseline>.sh`) đọc `config/<baseline>.env`, `cd` vào
thư mục gốc của repo baseline nếu cần, set `PYTHONPATH` vào `externals/<repo>`
và chạy `uv run --project <env> --locked python scripts/infer_<baseline>.py`.

## Ghi chú

- **flash-attn**: prebuilt wheel chỉ hỗ trợ sm80+ (A100/H100). T4 (sm75) phải
  build từ source nên mặc định KHÔNG cài; dùng `--extra flash` chỉ trên GPU lớn.
  Code FastKV/SpecExtend đã được patch để import không lỗi khi thiếu flash-attn
  (chỉ các đường chạy flash mới cần nó).
- **dgl**: wheel Linux được **vendor** trong repo (`envs/legacy/wheels/`) vì
  PyPI chỉ có wheel Windows và index riêng của dgl chặn uv (403). Bản này là
  CPU build — đủ cho smoke/verify; full pipeline HiGOE dùng GPU có thể cần bản
  `+cu*` riêng.
- **HF token**: các baseline dùng model gated (Llama) cần `HF_TOKEN` trong env.
- Sau khi sửa `pyproject.toml` của một env: chạy `uv lock --project envs/<g>`
  rồi commit cả `uv.lock` để giữ tái lập được. **Nhớ**: `envs/` trong
  `.gitignore` chỉ chặn `.venv`, còn `pyproject.toml` + `uv.lock` + `wheels/`
  đều được commit.
