# Nhật ký Huấn luyện: DFlash Paper trên 4 GPU B200 (50K Phase 1)

> **Ngày ghi nhận:** 2026-10-05  
> **Trạng thái:** Job đang chạy (Training in progress trên 4 GPU B200)  
> **Tác vụ:** Huấn luyện DFlash 5-layer Draft Model (Qwen3-4B backbone) trên tập 50K Phase 1 ngữ cảnh dài (16K context).

---

## 1. Tóm tắt tiến trình đã hoàn thành trong phiên làm việc

1. **Kiểm tra trạng thái dữ liệu & Cache Phase 1:**
   - Dữ liệu 50K Phase 1 (ArXiv + ShareGPT) đã được sinh và trích xuất cache offline đầy đủ tại:
     `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_phase1_50k_50_50_10k_b200_gpu0_ctx16k_out1k_20260928T132928Z/`
   - Khắc phục lỗi lệch 2 mẫu tập Val (`sharegpt_JYJaytf_0` và `sharegpt_fQhexkP_58`) do 0 supervised tokens bằng `/tmp/val_filtered.jsonl`.
   - Lệnh kiểm tra cấu trúc `verify_feature_cache.py` đã **PASS 100%**: `valid samples=2498 offsets=13340890 shards=79`.
2. **Cập nhật hệ thống Logging & Thanh tiến trình Trainer (`src/MR_DFlash/trainer.py`):**
   - Đã tích hợp thanh tiến trình trực quan `tqdm` (hiển thị `%`, `step/total`, `step/s`, `ETA`).
   - Postfix cập nhật thời gian thực: `loss`, `acc`, `lr`, `tok/s`.
   - In log chi tiết định kỳ và **luôn in ở step 1**: hiển thị rõ `%`, `loss`, `acc`, `lr`, `grad_norm`, `tokens/s`, `step_time` và `peak VRAM`.
   - Bảo vệ giao diện không bị vỡ thanh tiến trình khi lưu checkpoint hoặc đánh giá `evaluate()`.
3. **Thực nghiệm Smoke Test & Tìm điểm ngọt VRAM:**
   - Smoke test 5 steps DDP 4 GPU đạt kết quả xuất sắc: VRAM ~28 GB/card, thông lượng lên tới 92.000 tokens/s.
   - Thử nghiệm `batch_size: 14` gặp OOM ở step 16 do gặp batch toàn mẫu 16K (chạm trần 180 GB và gây nghẽn bộ nhớ 14s/step).
   - **Xác định điểm ngọt tối ưu (Sweet Spot):** Hạ `batch_size` xuống **`8`** mỗi card (tổng global batch size = 32). VRAM đỉnh nằm ở mức an toàn **~105 – 115 GB**, còn dư hơn 65 GB headroom chống OOM, tốc độ đạt cực đại ~80.000 – 90.000 tok/s.
4. **Quyết định chiến lược phân bổ GPU:**
   - Dồn toàn lực **4 card GPU** để hoàn thành trước **Job 1: DFlash Paper**.
   - Thời gian rút ngắn gần 50%: từ ~6.5 tiếng xuống chỉ còn **~3.2 – 3.5 tiếng**.

---

## 2. Thông số Job đang chạy hiện tại

| Thuộc tính | Giá trị cấu hình |
|---|---|
| **Mô hình mục tiêu (Target)** | Qwen3-4B (`/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B`), BF16, Frozen |
| **Kiến trúc Draft** | DFlash 5-layer (`draft_num_hidden_layers: 5`, `draft_intermediate_size: 9728`) |
| **Feature Layers** | `[1, 9, 17, 25, 33]` (Hidden width 12.800) |
| **Thiết bị** | 4 $\times$ NVIDIA B200 (`CUDA_VISIBLE_DEVICES=0,1,2,3`) qua `torchrun` DDP |
| **Batch size** | `batch_size = 8` / rank $\rightarrow$ Global batch size = 32 |
| **Accumulation steps** | 1 |
| **Ngữ cảnh tối đa** | 16.384 tokens (16K context) |
| **Số epoch** | 6 epochs |
| **Tổng số steps dự kiến** | $\approx 8.430$ steps |
| **Thời gian mỗi step** | $\approx 1.3\text{s} – 1.6\text{s}$ |
| **Thời gian chạy ước tính** | $\approx 3.2 – 3.5$ tiếng |
| **Thư mục đầu ra (Output)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/dflash_paper_4gpu` |

---

## 3. Câu lệnh đã khởi chạy

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_for_text_sum-main

export CUDA_VISIBLE_DEVICES=0,1,2,3
export FI_OFFLINE=1
export PYTHONPATH=src

torchrun --standalone --nproc_per_node=4 -m MR_DFlash.run_train \
  --config src/MR_DFlash/configs/train_qwen3_4b_dflash_b200_2gpu_170gb.yaml \
  --output-dir /workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/dflash_paper_4gpu \
  --device cuda \
  --batch-size 8 \
  --accumulation-steps 1
```

---

## 4. Runbook cho Phiên làm việc tiếp theo (Next Session)

Khi mở lại phiên làm việc sau, thực hiện theo thứ tự các bước sau:

### Bước 1: Kiểm tra kết quả huấn luyện DFlash Paper
1. **Kiểm tra trạng thái job và file checkpoint:**
   ```bash
   cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_for_text_sum-main
   ls -la /workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/dflash_paper_4gpu
   ```
   *Yêu cầu kiểm tra:* Phải có `checkpoint_final.pt`, `checkpoint_best_eval.pt` và file `eval_metrics.json`.
2. **Xem loss cuối cùng và lịch sử hội tụ:**
   ```bash
   tail -n 20 /workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/dflash_paper_4gpu/metrics.jsonl
   cat /workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/dflash_paper_4gpu/eval_metrics.json
   ```

---

### Bước 2: Khởi chạy tiếp Job 2 — MR-DFlash (HCA + CSA) trên 4 card

Sau khi DFlash Paper hoàn tất, toàn bộ 4 card GPU sẽ rảnh. Tiến hành khởi chạy ngay **MR-DFlash** với cùng dữ liệu, cùng batch 8 trên 4 GPU:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_for_text_sum-main

export CUDA_VISIBLE_DEVICES=0,1,2,3
export FI_OFFLINE=1
export PYTHONPATH=src

torchrun --standalone --nproc_per_node=4 -m MR_DFlash.run_train \
  --config src/MR_DFlash/configs/train_qwen3_4b_mr_dflash_b200_2gpu_170gb.yaml \
  --output-dir /workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_4gpu \
  --device cuda \
  --batch-size 8 \
  --accumulation-steps 1
```
*(Job này cũng sẽ chạy trong khoảng **~3.5 tiếng**).*

---

### Bước 3: Benchmark đối đầu (A/B Testing & Evaluation)
Sau khi có cả 2 checkpoint:
1. Checkpoint 1: DFlash Paper 5L (`outputs/dflash_paper_4gpu/checkpoint_best_eval.pt`)
2. Checkpoint 2: MR-DFlash 2S (`outputs/mr_dflash_4gpu/checkpoint_best_eval.pt`)

Chạy script đánh giá suy luận (Inference Evaluation) trên tập benchmark LongBench canonical để so sánh:
- **Mean Acceptance Length ($\tau$)**
- **Draft Exact Match Rate**
- **Tốc độ sinh (Tokens/second & Latency E2E)**
- **Điểm ROUGE task-aware**
