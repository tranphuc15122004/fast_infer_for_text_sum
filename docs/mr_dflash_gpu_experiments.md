# MR-DFlash GPU experiments — B200 runbook

Tài liệu này là runbook cho các run cần thực hiện sau khi có GPU được cấp
riêng. Agent không tự chọn GPU hay chiếm GPU của job khác; người chạy phải
điền đúng `CUDA_VISIBLE_DEVICES` sau khi scheduler xác nhận allocation.

## Preflight gần nhất — 2026-09-06

- `nvidia-smi` nhìn thấy Tesla T4 15 GiB, driver `550.163.01`, CUDA `12.4`,
  không có process đang chạy trên GPU.
- Python runtime hiện tại dùng `torch 2.11.0+cu130` nhưng báo
  `torch.cuda.is_available()=False` và `torch.cuda.device_count()=0`, kèm cảnh
  báo không khởi tạo được NVML.
- Vì runtime T4 hiện tại không expose CUDA cho PyTorch, GPU smoke B200 chưa
  được thực hiện tại workspace này; không xem kết quả CPU là bằng chứng GPU.
- CPU simulation bằng `.venv/bin/python` (Python `3.12.13`) chạy được toàn bộ
  MR-DFlash package smoke: `12 passed, 2 skipped` trong khoảng `31s`. Đây không phải validation
  của server B200; `.venv` vẫn chứa PyTorch `cu130` và chỉ được dùng CPU trên
  máy T4 này.
- Real-model smoke đã chạy offline với snapshot Qwen3-4B local: capture 5
  layer (`hidden_size=2560`), train MR-DFlash `1` optimizer step, lưu/nạp
  `draft_final.pt`, rồi chạy đủ `prefill → draft → verify → generate`; kết quả
  `1 passed` trong `211.45s`, `loss=12.5037`, output 2 token finite. Đây là
  functional smoke trên CPU, chưa phải GPU benchmark.
- DDP CPU world-size 2 cũng đã pass bằng `torchrun`; đây chỉ là validation
  process-group/sharding/checkpoint, không thay thế GPU smoke.

## Điều kiện cấp job

Trước mỗi run, người thực hiện phải xác nhận GPU không thuộc job khác và thay
`<ALLOCATED_GPU>` bằng device đã cấp:

```bash
export CUDA_VISIBLE_DEVICES=<ALLOCATED_GPU>
export MR_RUN_ROOT=outputs/mr-dflash-gpu-YYYYMMDD-HHMMSS
```

Không dùng `CUDA_VISIBLE_DEVICES` rỗng, không dùng wildcard GPU và không ghi
đè output của run khác. Lưu stdout, config YAML và commit SHA cùng output.

## Rung 1 — GPU functional smoke

Mục tiêu là xác nhận CUDA dtype/kernel, không dùng để claim speedup:

```bash
cd /home/tuantb/fast_infer_text_sum
export CUDA_VISIBLE_DEVICES=<ALLOCATED_GPU>
export FI_OFFLINE=1
export PYTHONPATH=src
export MR_RUN_ROOT="outputs/mr-dflash-gpu-YYYYMMDD-HHMMSS"

python3 -m MR_DFlash.run_train \
  --config src/MR_DFlash/configs/qwen3_4b_mr_dflash.yaml \
  --device cuda \
  --max-steps 2 \
  --batch-size 1 \
  --num-anchors 8 \
  --num-workers 0 \
  --output-dir "$MR_RUN_ROOT/smoke"
```

Kiểm tra `loss` finite, checkpoint load được và target parameters không xuất
hiện trong `draft_final.pt`.

Smoke 2 GPU (chỉ chạy sau khi đã xác nhận hai GPU cùng allocation):

```bash
export CUDA_VISIBLE_DEVICES=<GPU0>,<GPU1>
torchrun --standalone --nproc_per_node=2 \
  -m MR_DFlash.run_train \
  --config src/MR_DFlash/configs/qwen3_4b_mr_dflash.yaml \
  --device cuda --max-steps 2 --batch-size 1 --num-anchors 8 \
  --num-workers 0 --output-dir "$MR_RUN_ROOT/smoke-2gpu"
```

Sau functional smoke, run train thật dùng `--batch-size 4` trên 1 GPU hoặc
`--batch-size 2` trên mỗi GPU khi dùng 2 GPU; cả hai giữ global batch `4` như
DFlash gốc. Có thể tăng `--num-workers 4` sau khi xác nhận feature store nằm
trên local NVMe.

## Rung 2 — Inference correctness

Sau khi có `draft_final.pt`, chạy greedy generation cùng prompt cố định:

```bash
PYTHONPATH=src python3 -m MR_DFlash.inference \
  --target-model-path Qwen/Qwen3-4B \
  --draft-checkpoint-path "$MR_RUN_ROOT/smoke/draft_final.pt" \
  --prompt "<FIXED_PROMPT>" \
  --mask-token-id 151669 \
  --device cuda \
  --max-new-tokens 64 \
  --local-files-only
```

So sánh output với target greedy decode cùng `max_new_tokens`, ghi
`accepted_proposal_tokens`, số vòng verify và output token ids. Bản reference
hiện chỉ hỗ trợ greedy; sampling cần experiment card riêng.

## Rung 3 — Baseline/variant benchmark

Giữ cố định target, data split, prompt order, seed và `max_new_tokens`. Chạy
DFlash và MR-DFlash với các tham số DFlash giống nhau:

| Nhóm | Kiến trúc | Block | Anchor | LR | Loss decay |
|---|---|---:|---:|---:|---:|
| Baseline | DFlash | 16 | 512 | 6e-4 | 7.0 |
| Variant | MR-DFlash | 16 | 512 | 6e-4 | 7.0 |

MR-specific values cố định ở V1: HCA `128`, CSA `4`, local `128`, Top-k `64`,
2 stages. Mỗi run cần ghi:

- primary: acceptance rate trên cùng token budget;
- guardrails: target quality/ROUGE, draft memory, peak VRAM, prefill/decode
  latency, tokens/s;
- checkpoint, config dump, seed và GPU id.

Không dùng Rung 1 hoặc CPU smoke để kết luận MR-DFlash nhanh hơn hay tốt hơn
DFlash; chúng chỉ xác nhận pipeline hoạt động. Rung 3 mới là benchmark có
ý nghĩa, và phải chạy cùng prompt/seed/token budget/cache target.

## Rung 0 — kiểm tra trước khi cấp GPU

```bash
cd /home/tuantb/fast_infer_text_sum
nvidia-smi
FI_OFFLINE=1 PYTHONPATH=src python3 - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA chưa sẵn sàng trong runtime hiện tại")
print(torch.cuda.get_device_name(0))
PY
```

Nếu có job khác trên GPU được cấp, dừng run và xin allocation khác; không
ghi đè `MR_RUN_ROOT` đã tồn tại.
