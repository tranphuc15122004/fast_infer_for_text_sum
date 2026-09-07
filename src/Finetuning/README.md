# Finetuning DFlash offline

Đây là pipeline self-contained để train draft DFlash cho Qwen3 trên một tiến
trình CPU/GPU. Target model luôn frozen; chỉ `draft_model` được optimizer cập
nhật. Pipeline không tải dataset hoặc model từ internet khi `offline: true`.

## Synthetic smoke

```bash
PYTHONPATH=src .venv/bin/python -m Finetuning.run_train \
  --config src/Finetuning/configs/synthetic_smoke.yaml \
  --device cpu
```

Kết quả nằm trong `outputs/finetuning-synthetic/`, gồm `metrics.jsonl`,
`train.log` và checkpoint draft-only.

## Dữ liệu thật

JSONL tối thiểu:

```json
{"id": "sample-001", "document": "...", "summary": "..."}
```

Chỉnh `qwen3_4b.yaml` hoặc `qwen3_8b.yaml` để trỏ tới snapshot model local và
file JSONL. `run_train` sẽ render chat template, tạo `loss_mask` chỉ trên
summary, capture hidden states theo target layer, validate manifest rồi train.

Nếu đã có feature store, đặt `data.hidden_states_path` và để
`data.train_data_path: null`; hai nguồn không được bật đồng thời.

```bash
PYTHONPATH=src .venv/bin/python -m Finetuning.run_train \
  --config src/Finetuning/configs/qwen3_4b.yaml \
  --device cuda
```

Checkpoint được lưu theo step và có optimizer, scheduler, RNG, config cùng
draft weights để phục vụ resume/evaluation. Không dùng synthetic smoke để suy
ra chất lượng hoặc ROUGE tiếng Việt.
