# Fine-tune DFlash cho tóm tắt tiếng Việt

`src/Finetuning` là pipeline DFlash độc lập cho Qwen3. Nó không import từ
`src/MR_DFlash`; target Qwen luôn frozen, còn optimizer chỉ cập nhật
`DFlashDraftModel`.

Điều cần phân biệt: DFlash học để tăng tốc chính target, không làm target có
chất lượng tóm tắt cao hơn. Vì vậy pipeline train trên **summary do chính
target sinh**, nhưng giữ gold summary của người viết để đo ROUGE sau cùng.

## Quy trình thật

Input gốc UTF-8 JSONL:

```json
{"id":"vi-001","document":"...","summary":"gold summary"}
```

Chạy ba stage riêng biệt. Mọi model đều phải là snapshot local khi
`offline: true`.

```bash
# 1. Sinh trajectory greedy của frozen target. summary trong file output là
#    target trajectory; reference_summary vẫn là gold của dữ liệu gốc.
PYTHONPATH=src python3 -m Finetuning.generate_targets \
  --input /data/vietnamese_train.jsonl \
  --output /work/teacher_train.jsonl \
  --target-model-path /models/Qwen3-4B \
  --max-length 2048 --max-source-tokens 1536 --max-summary-tokens 384 \
  --torch-dtype bfloat16 --device cuda

# 2. Capture hidden state một lần. --num-draft-layers=5 sẽ chọn cùng rule
#    target layer với config khi target_layer_ids: null.
PYTHONPATH=src python3 -m Finetuning.capture_features \
  --input /work/teacher_train.jsonl \
  --output /work/features_train \
  --target-model-path /models/Qwen3-4B \
  --num-draft-layers 5 \
  --max-length 2048 --max-source-tokens 1536 --max-summary-tokens 384 \
  --torch-dtype bfloat16 --device cuda

# Lặp lại stage 1–2 cho validation và thay các path trong config.
PYTHONPATH=src python3 -m Finetuning.run_train \
  --config src/Finetuning/configs/qwen3_4b.yaml --device cuda
```

`run_train` chỉ nhận `data.hidden_states_path` và
`data.eval_hidden_states_path`; nó chủ động từ chối raw JSONL. Nhờ vậy capture
không đồng thời giữ hai bản target trên GPU. Manifest cache kiểm tra nghiêm
ngặt target/tokenizer, target layers, dtype, prompt template và các budget
trước khi train. Cache được publish atomic; có thể chạy lại capture an toàn
sau lỗi, nhưng phải dùng thư mục output chỉ dành cho feature store.

Prompt mặc định là chỉ dẫn tóm tắt tiếng Việt có một placeholder
`{document}`. Nếu thay `data.prompt_template`, phải dùng đúng chuỗi đó cho
**cả generate, capture và evaluate**; manifest sẽ từ chối cache khác contract.

Mỗi checkpoint có thư mục `draft_export/` gồm `draft_state_dict.pt`,
`draft_metadata.json` và `COMPLETE`. Có thể tiếp tục domain adaptation bằng:

```yaml
model:
  draft_init_path: /work/previous-run/qwen3-4b-step1000/draft_export
```

Không kết hợp `draft_init_path` với `--resume-from`. Resume kiểm tra metadata
export (target, layer IDs, mask token, block size và draft config) trước khi
phục hồi optimizer/scheduler/RNG.

## Đánh giá sau train

Đánh giá generation độc lập dùng cùng prompt contract và file validation gốc
hoặc file trajectory (nếu là trajectory nó tự dùng `reference_summary`).

```bash
PYTHONPATH=src python3 -m Finetuning.generation_evaluation \
  --input /work/teacher_eval.jsonl \
  --output /work/dflash_eval.jsonl \
  --target-model-path /models/Qwen3-4B \
  --draft-export /work/output/qwen3-4b-step1000/draft_export \
  --max-length 2048 --max-source-tokens 1536 --max-summary-tokens 384 \
  --torch-dtype bfloat16 --device cuda
```

Output ghi ROUGE-1/2/L F1 với gold summary, `target_token_match` cho từng mẫu,
và summary có `target_exact_rate` cùng paired `speedup`. Với greedy decoding,
`target_exact_rate` phải bằng 1.0 trước khi dùng số speedup để kết luận. Nếu
không đạt, checkpoint hoặc contract target/draft không phù hợp.

## Quy mô và smoke

Five target layers của Qwen3-8B ở BF16, sequence 2048, xấp xỉ 80 MiB hidden
state/mẫu. Bắt đầu với 500 → 2K mẫu, quan sát validation loss/accuracy và
`target_exact_rate`, rồi mới capture tập lớn. Dataset chỉ được mở lazy qua
`DataLoader`; không materialize toàn bộ tensor cache vào RAM.

Smoke CPU không cần snapshot:

```bash
PYTHONPATH=src .venv/bin/python -m Finetuning.run_train \
  --config src/Finetuning/configs/synthetic_smoke.yaml --device cpu
```

Smoke chỉ kiểm tra lifecycle, không đại diện cho ROUGE tiếng Việt, VRAM hay
speedup thực tế.
