# MR-DFlash Correctness and Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sửa các invariant P0 của MR-DFlash và đưa draft stage về DFlash joint-attention semantics trước khi train lớn.

**Architecture:** Giữ feature contract và DFlash loss/trainer spine. Memory sẽ có complete compressed groups cùng hai pending streams độc lập; joint attention sẽ nhận HCA hoặc CSA context cùng draft KV trong một softmax, với RoPE và mask dùng chung giữa train/inference. CSA có dense warm-up/score-bias path để Indexer nhận gradient, còn inference dùng hard Top-k.

**Tech Stack:** Python 3.12, PyTorch/SDPA, Hugging Face tiny Qwen3 tests, YAML/dataclass; không thêm dependency runtime.

**Spec:** `docs/superpowers/specs/2026-09-06-mr-dflash-design.md`

## Global Constraints

- Chỉ sửa `src/MR_DFlash`, tests và docs liên quan; không sửa `externals/dflash`.
- Không chạy GPU job trong lượt này; T4 hiện tại chỉ dùng CPU vì runtime cu130 không tương thích driver 12.4.
- Giữ `input_ids`, `hidden_states`, `loss_mask` và DFlash hard-label objective.
- Khi không có `sliding_window`, dùng full same-block mask giống DFlash/SpecForge; train và inference dùng cùng policy.
- Mỗi production change có failing test trước implementation; không chạy train lớn trước khi P0 tests xanh.

---

### Task 1: Memory state, complete groups và anchor-local views

**Files:**
- Modify: `src/MR_DFlash/memory.py`
- Modify: `src/MR_DFlash/training.py`
- Test: `src/MR_DFlash/tests/test_mr_memory.py`
- Test: `src/MR_DFlash/tests/test_b200_training_design.py`

**Interfaces:**
- `MRMemoryState` có `pending_hca_positions` và `pending_csa_positions` độc lập.
- `MRTargetMemory.build(features, query_positions=None) -> MRMemoryState` giữ partial tail.
- `MRTargetMemory.append(state, features, positions=None) -> MRMemoryState` bảo toàn parity.
- Training truyền anchor positions vào `build()` để dựng local context theo `[anchor-window, anchor)`.

- [x] Viết test build/append parity với ratio HCA khác CSA và partial tails.
- [x] Chạy test để xác nhận failure vì state hiện dùng một pending position và `ceil` compress.
- [x] Sửa state và `_append_stream`; `build()` chỉ pool `floor(length/ratio)*ratio`, phần dư vào pending.
- [x] Thêm query-relative local raw views cho training; giữ tail view 3D cho inference incremental.
- [x] Chạy memory/training tests và xác nhận complete positions, pending positions và local windows đúng.

### Task 2: Shared mask và đúng scaling

**Files:**
- Modify: `src/MR_DFlash/training.py`
- Modify: `src/MR_DFlash/inference.py`
- Modify: `src/MR_DFlash/model.py`
- Modify: `src/MR_DFlash/mr_model.py`
- Test: `src/MR_DFlash/tests/test_mr_model_cpu.py`
- Test: `src/MR_DFlash/tests/test_mr_inference_cpu.py`

**Interfaces:**
- Một helper biểu diễn draft block visibility được dùng cho SDPA train và reference inference.
- `build_dflash_additive_mask(..., sliding_window=None)` và inference cùng full same-block policy.

- [x] Viết test so sánh train/inference mask cho full same-block và sliding mode.
- [x] Viết test attention scale với một fixture nhỏ, chứng minh SDPA không bị nhân scale hai lần.
- [x] Sửa mask inference và các comment/semantics tương ứng.
- [x] Bỏ `q * self.scaling` trước SDPA, truyền scale đúng một lần; sửa cả DFlash self-contained và MR joint attention.
- [x] Chạy model/inference tests và suite DFlash-related để xác nhận không regress.

### Task 3: DFlash joint attention, CSA composition và RoPE

**Files:**
- Modify: `src/MR_DFlash/mr_model.py`
- Modify: `src/MR_DFlash/memory.py`
- Test: `src/MR_DFlash/tests/test_mr_model_cpu.py`
- Test: `src/MR_DFlash/tests/test_mr_memory.py`

**Interfaces:**
- `MRDFlashJointAttention.forward(query, context, query_positions, context_positions, attention_mask, context_bias=None) -> hidden`.
- Context nhận `[B,K,H]` hoặc per-query `[B,Q,K,H]`; draft KV luôn được nối trong cùng attention.
- CSA tạo context `[local; selected]` và một combined mask/bias trước một softmax.

- [x] Viết test đảm bảo output joint attention thay đổi khi target context thay đổi và không gọi hai branch attention độc lập.
- [x] Viết test CSA local+selected có một combined context và score bias shape đúng.
- [x] Viết test RoPE dùng compressed group-end positions và speculative absolute positions.
- [x] Thay `MRBlockAttention + MRTargetAttention` bằng joint attention, giữ residual/FFN và route HCA/CSA.
- [x] Chạy model tests, kiểm tra output hữu hạn và gradient hữu hạn.

### Task 4: Stage/init schema và Indexer training path

**Files:**
- Modify: `src/MR_DFlash/config.py`
- Modify: `src/MR_DFlash/mr_model.py`
- Modify: `src/MR_DFlash/training.py`
- Modify: `src/MR_DFlash/run_train.py`
- Modify: `src/MR_DFlash/configs/qwen3_4b_mr_dflash.yaml`
- Modify: `src/MR_DFlash/configs/qwen3_8b_mr_dflash.yaml`
- Test: `src/MR_DFlash/tests/test_b200_training_design.py`
- Test: `src/MR_DFlash/tests/test_mr_train_smoke.py`

**Interfaces:**
- `mr_stage_init_layer_ids` có đúng `mr_num_stages` phần tử.
- `MRDraftSpec` route xen kẽ `HCA, CSA, HCA, ...`.
- `CSAIndexer.score()` trả dense scores; `select()` giữ hard Top-k.
- `indexer_train_mode=schedule|dense|topk`; dense dùng scores làm attention
  bias để q/k có gradient, schedule tự chuyển sang top-k sau warm-up.

- [x] Viết test config MR init không còn truyền một ID cho hai stage.
- [x] Viết test `indexer.q_proj/k_proj` nhận gradient hữu hạn ở dense mode và top-k score-bias mode.
- [x] Implement stage init schema độc lập với DFlash `draft_num_hidden_layers`.
- [x] Implement Indexer score/dense path và truyền mode từ training wrapper.
- [x] Chạy train smoke và kiểm tra optimizer bao phủ adapter/compressor/indexer/stage.

### Task 5: Checkpoint provenance và inference reconstruction

**Files:**
- Modify: `src/MR_DFlash/checkpoint.py`
- Modify: `src/MR_DFlash/trainer.py`
- Modify: `src/MR_DFlash/inference.py`
- Modify: `src/MR_DFlash/data.py`
- Test: `src/MR_DFlash/tests/test_mr_inference_cpu.py`
- Test: `src/MR_DFlash/tests/test_b200_training_design.py`

**Interfaces:**
- Weights-only checkpoint chứa `config_yaml` hoặc resolved model metadata.
- Inference ưu tiên reconstruct model từ checkpoint config; CLI chỉ override rõ ràng.
- Manifest validate schema, feature width, layer IDs và target provenance khi có expected target path.

- [x] Viết test save/load checkpoint rồi dựng lại đúng 5 feature layers và MR ratios.
- [x] Viết test inference không mismatch input width khi không truyền `--target-layer-ids`.
- [x] Lưu config vào full và weights-only checkpoint; thêm parser reconstruction.
- [x] Thêm `--local-files-only` và offline env handling cho inference CLI.
- [x] Chạy inference smoke với checkpoint reload.

### Task 6: Verifier edge cases, docs và full validation

**Files:**
- Modify: `src/MR_DFlash/inference.py`
- Modify: `src/MR_DFlash/README.md`
- Modify: `docs/mr_dflash.md`
- Modify: `docs/mr_dflash_gpu_experiments.md`
- Test: `src/MR_DFlash/tests/test_mr_inference_cpu.py`

- [x] Viết test EOS xuất hiện giữa accepted block và test bonus token bị giới hạn bởi max_new_tokens.
- [x] Implement stop-at-first-EOS và bonus-token bookkeeping không làm sai greedy output.
- [x] Cập nhật docs: approximation vs DeepSeek-faithful parts, indexer warm-up, stage config và B200 command.
- [x] Chạy `git diff --check`, `py_compile`, toàn bộ `src/MR_DFlash/tests` và real Qwen3-4B smoke CPU.
- [x] Ghi rõ GPU chưa chạy và không claim speedup/quality.

## Follow-up review — scalability and architecture fidelity (2026-09-07)

Phần này thực hiện các nhận xét mới sau khi đọc lại implementation tại
`a182c46`. Các điểm DeepSeek-specific chưa đủ để gọi là faithful production
module được giữ trong phạm vi V1 và ghi thành ablation/deferred work.

### Task 7: Block-batch training và projected KV

**Files:** `src/MR_DFlash/training.py`, `src/MR_DFlash/mr_model.py`,
`src/MR_DFlash/memory.py`, `src/MR_DFlash/tests/test_mr_review_followups.py`

- [x] Thêm `build_dflash_block_additive_mask()` trả `[B*N,1,K,K]` mà không
  materialize mask/logits `[B,1,N*K,N*K]`.
- [x] Reshape training noise/positions thành `[B*N,K,*]`; flatten memory
  query-relative theo block và giữ tương thích với caller cũ nhiều block.
- [x] Chiếu shared context một lần trong `project_context()`; Top-k gather
  projected K/V thay vì gather hidden context rồi chiếu theo query.
- [x] Test block batch, mask diagonal tương đương và local view từng anchor.

### Task 8: Adapter, Indexer và checkpoint transfer

**Files:** `src/MR_DFlash/memory.py`, `src/MR_DFlash/checkpoint.py`,
`src/MR_DFlash/mr_model.py`, `src/MR_DFlash/tests/test_mr_review_followups.py`

- [x] Thêm RMSNorm riêng cho HCA/CSA adapter.
- [x] Đổi Indexer thành head-wise `ReLU(dot(q,k))`; giữ score-bias mặc định
  như bridge differentiable và ghi rõ không phải ranking loss/Lightning
  production.
- [x] Thêm converter DFlash → MR-DFlash: map attention/MLP/norm vào mọi stage,
  map `fc`/`hidden_norm` vào hai adapter, để compressor/indexer khởi tạo mới.
- [x] Native MR checkpoint load strict; thiếu/unexpected key phải fail rõ ràng.
- [x] Thêm test công thức score, converter và strict loading.

### Task 9: Documentation and validation

**Files:** `docs/mr_dflash.md`, `docs/mr_dflash_gpu_experiments.md`,
`docs/superpowers/specs/2026-09-06-mr-dflash-design.md`

- [x] Ghi complexity/layout mới, projected KV, Q/K Norm→RoPE, Indexer
  surrogate và ablation `indexer_num_heads={1,4,8}`.
- [x] Cập nhật B200 runbook; không tự chạy GPU khi chưa có allocation riêng.
- [x] Chạy review tests và toàn bộ `src/MR_DFlash/tests`; giữ real Qwen smoke
  cho validation cuối cùng trước claim pilot.
