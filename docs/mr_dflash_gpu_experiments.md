# MR-DFlash GPU experiments — B200 runbook

Tài liệu này là runbook cho các run cần thực hiện sau khi có GPU được cấp
riêng. Agent không tự chọn GPU hay chiếm GPU của job khác; người chạy phải
điền đúng `CUDA_VISIBLE_DEVICES` sau khi scheduler xác nhận allocation.

## Preflight gần nhất — 2026-09-07

- `nvidia-smi` nhìn thấy Tesla T4 15 GiB, driver `550.163.01`, CUDA `12.4`,
  không có process đang chạy trên GPU.
- Python runtime hiện tại dùng `torch 2.11.0+cu130` nhưng báo
  `torch.cuda.is_available()=False` và `torch.cuda.device_count()=0`, kèm cảnh
  báo không khởi tạo được NVML.
- Preflight lại lúc `2026-09-07 08:50 UTC`: `nvidia-smi` chạy được và GPU vẫn
  trống, nhưng runtime PyTorch không expose CUDA do mismatch driver/cu130; máy
  này không được xem là GPU smoke host.
- Vì runtime T4 hiện tại không expose CUDA cho PyTorch, GPU smoke B200 chưa
  được thực hiện tại workspace này; không xem kết quả CPU là bằng chứng GPU.
- CPU simulation bằng `.venv/bin/python` (Python `3.12.13`) chạy được toàn bộ
  MR-DFlash package smoke: `29 passed, 2 skipped` trong khoảng `31s`. Đây không phải validation
  của server B200; `.venv` vẫn chứa PyTorch `cu130` và chỉ được dùng CPU trên
  máy T4 này.
- Real-model smoke vừa chạy offline với snapshot Qwen3-4B local: capture 5
  layer (`hidden_size=2560`), train MR-DFlash `1` optimizer step, lưu/nạp
  `draft_final.pt`, rồi chạy đủ `prefill → draft → verify → generate`; kết quả
  `1 passed` trong `164.86s`, `loss=12.0915`, output 2 token finite. Đây là
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

Các smoke config tương ứng cũng phải được kiểm tra để phát hiện mismatch
feature width/depth trước khi train pilot:

```bash
python3 -m MR_DFlash.run_train \
  --config src/MR_DFlash/configs/qwen3_4b_dflash_1l.yaml \
  --device cuda --max-steps 2 --batch-size 1 --num-anchors 8 \
  --num-workers 0 --output-dir "$MR_RUN_ROOT/dflash-1l-smoke"
```

```bash
python3 -m MR_DFlash.run_train \
  --config src/MR_DFlash/configs/qwen3_4b_dflash_2l.yaml \
  --device cuda --max-steps 2 --batch-size 1 --num-anchors 8 \
  --num-workers 0 --output-dir "$MR_RUN_ROOT/dflash-2l-smoke"
```

Mỗi `metrics.jsonl` phải có `loss`, `acc`, `step_time_s` và
`tokens_per_second`; trên CUDA phải có thêm `peak_memory_allocated_mb` và
`peak_memory_reserved_mb`. Đây là các số đo pilot, không phải kết luận
speedup.

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
  --local-files-only \
  --timing-json
```

CLI vẫn in decoded text; dòng JSON cuối có `accepted_proposal_tokens`, số vòng
verify, `generated_tokens` và `timings_s={prefill_s,draft_s,verify_s,total_s}`.
So sánh output với target greedy decode cùng `max_new_tokens` và lưu cả token
ids. Bản reference hiện chỉ hỗ trợ greedy; sampling cần experiment card riêng.

## Rung 3 — Baseline/variant benchmark

Giữ cố định target, data split, prompt order, seed và `max_new_tokens`. Chạy
đủ ba cấu hình Qwen3-4B sau; không so MR 2-stage chỉ với DFlash 1-layer:

| Nhóm | Config | Feature layers | Draft depth | Vai trò |
|---|---|---|---:|---|
| DFlash-1L | `qwen3_4b_dflash_1l.yaml` | `[1,9,17,25,33]` | 1 | baseline |
| DFlash-2L | `qwen3_4b_dflash_2l.yaml` | `[1,9,17,25,33]` | 2 | depth/capacity control |
| MR-HCA+CSA | `qwen3_4b_mr_dflash.yaml` | `[1,9,17,25,33]` | 2 stage | variant |

Tất cả cùng block `16`, anchors `512`, LR `6e-4`, loss decay `7.0`,
`max_length=3072`, seed `42` và init policy tương ứng. DFlash-2L là
depth-matched, không phải parameter-count matched tuyệt đối; ghi
`trainable_parameter_count` và peak VRAM của cả ba run.

MR-specific values cố định ở V1: HCA `128`, CSA `4`, local `128`, Top-k `64`,
2 stages, indexer dense warm-up `1000` optimizer steps rồi hard Top-k,
`indexer_num_heads=1` cho baseline. Joint attention MR V1 dùng block-batch
layout `[B*N,K,H]`, project shared KV một lần và gather projected KV cho CSA;
chưa xem đây là benchmark kernel tối ưu. Có thể chạy thêm ablation
`indexer_num_heads=4,8` với seed/data giữ nguyên. Mỗi run cần ghi:

- primary: acceptance rate trên cùng token budget;
- guardrails: target quality/ROUGE, draft memory, peak VRAM, prefill/decode
  latency, tokens/s;
- checkpoint, config dump, seed và GPU id.

Không dùng Rung 1 hoặc CPU smoke để kết luận MR-DFlash nhanh hơn hay tốt hơn
DFlash; chúng chỉ xác nhận pipeline hoạt động. Rung 3 mới là benchmark có
ý nghĩa, và phải chạy cùng prompt/seed/token budget/cache target.

### Ma trận Llama 3.1 8B Instruct theo ngân sách DFlash-5L

Khi benchmark target `meta-llama/Meta-Llama-3.1-8B-Instruct`, dùng cùng năm
feature layers `[1,8,15,22,29]` cho mọi draft. Ma trận tối thiểu:

| Nhóm | Config | Draft | Trainable params | Vai trò |
|---|---|---|---:|---|
| DFlash-5L | `llama3_1_8b_dflash_5l.yaml` | 5 DFlash layers, MLP 12288 | 1,048,626,432 | baseline |
| MR-4S | `llama3_1_8b_mr_dflash.yaml` | 4 MR stages, indexer 4096, MLP 12288 | 1,040,786,432 (-0.75%) | run chính |
| MR-4S exact | `llama3_1_8b_mr_dflash_exact_params.yaml` | 4 MR stages, indexer 5120, MLP 12288 | 1,049,175,040 (+0.052%) | parameter ablation |

`MR-4S` được khuyến nghị cho pilot và latency vì giữ indexer cùng chiều rộng
hidden; `MR-4S exact` kiểm soát số tham số chặt hơn nhưng có retrieval width
lớn hơn 25%, nên phải báo overhead này riêng. Parameter count chỉ là draft
trainable parameters, không tính target Llama bị freeze. Không gộp ba run này
với ma trận Qwen3; feature cache, tokenizer và manifest phải được capture riêng
cho Llama.

Config đặt `mask_token_id: 128002` explicit và khóa `block_size: 16` theo
protocol benchmark chung; checkpoint DFlash Llama gốc dùng block 10 nhưng ta
đang train lại với block 16 để mọi baseline cùng draft budget. Trước
capture/train thật, xác nhận token ID với tokenizer snapshot local. Nếu dùng
đường dẫn local trên B200, truyền `--target-model-path "$MODEL_TARGET"` và lưu
config/manifest thực tế cùng output.

Lệnh smoke/pilot dùng cấu hình chính như sau (baseline DFlash-5L thay tên YAML
và `--output-dir` tương ứng):

```bash
export CUDA_VISIBLE_DEVICES=<ALLOCATED_GPU>
export MODEL_TARGET=/path/to/Llama-3.1-8B-Instruct
export LLAMA_RUN_ROOT="outputs/mr-dflash-llama3-1-8b-$(date +%Y%m%d-%H%M%S)"
FI_OFFLINE=1 PYTHONPATH=src python3 -m MR_DFlash.run_train \
  --config src/MR_DFlash/configs/llama3_1_8b_mr_dflash.yaml \
  --target-model-path "$MODEL_TARGET" --device cuda \
  --max-steps 2 --batch-size 1 --num-anchors 8 --num-workers 0 \
  --output-dir "$LLAMA_RUN_ROOT/smoke"
```

Chỉ sau khi smoke pass mới chạy pilot `50–100` steps bằng cách bỏ
`--max-steps 2`, đặt output riêng và giữ feature cache/seed cố định. Chạy
`llama3_1_8b_dflash_5l.yaml` trước hoặc sau MR không ảnh hưởng, nhưng không
được dùng chung output directory giữa các architecture.

### Pilot ngắn trước serious training

Sau khi ba config smoke đều pass, chạy mỗi cấu hình `50–100` optimizer steps
trên cùng feature store. Mục tiêu là phát hiện OOM, loss không hữu hạn và
chênh lệch overhead trước khi tăng lên `10k` steps. Thu thập:

```text
train: loss, acc, grad_norm, step_time_s, tokens_per_second
memory: peak_memory_allocated_mb, peak_memory_reserved_mb
inference: prefill_s, draft_s, verify_s, total_s, accepted_proposal_tokens,
           generated_tokens, rounds
```

Nếu pilot ổn định ở 3K, serious long-context phải chạy curriculum riêng:
`3072 → 8192 → 16384` với checkpoint/seed được ghi rõ. Không gộp kết quả 3K
với kết luận về 8K/16K+.

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
