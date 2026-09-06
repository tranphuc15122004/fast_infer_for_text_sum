# MR-DFlash Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng model, inference engine và training adapter MR-DFlash trong `src/MR_DFlash`, có CPU smoke đầy đủ và kế hoạch GPU deferred.

**Architecture:** Giữ DFlash model/trainer/loss hiện có làm baseline. Thêm module memory HCA/CSA dùng chung cho train và inference, một MR draft model block-parallel, wrapper train tương thích feature contract cũ, và HF speculative engine reference.

**Tech Stack:** Python 3.12, PyTorch, transformers tiny local config trong smoke, YAML/dataclass hiện có; không thêm dependency runtime.

**Spec:** `docs/superpowers/specs/2026-09-06-mr-dflash-design.md`

## Global Constraints

- Chỉ sửa `src/MR_DFlash` và tài liệu/test liên quan; không sửa `externals/dflash`.
- CPU dev/debug là bắt buộc trên máy hiện tại; không khởi chạy hoặc chiếm GPU.
- Giữ feature contract `input_ids`, `hidden_states`, `loss_mask` và semantics DFlash loss.
- Mọi production code phải có test đỏ trước implementation tương ứng.
- Các claim hoàn tất chỉ đưa ra sau khi chạy test fresh và đọc exit code.

---

### Task 1: HCA/CSA memory primitives

**Files:**
- Create: `src/MR_DFlash/memory.py`
- Test: `src/MR_DFlash/tests/test_mr_memory.py`

**Interfaces:**
- `MRMemoryState`: state immutable-at-boundary gồm `hca`, `csa`, `positions`, `local`.
- `MRTargetMemory.build(features) -> MRMemoryState`.
- `CSAIndexer.select(query, csa_memory, top_k) -> (indices, scores)`.
- `MRTargetMemory.append(state, features, positions) -> MRMemoryState`.

- [x] Viết test compression shape, tail handling, Top-k clamp, finite scores và append-only state.
- [x] Chạy `pytest src/MR_DFlash/tests/test_mr_memory.py -q`; xác nhận fail vì module chưa tồn tại.
- [x] Implement adapter, weighted pooling, HCA local merge, CSA indexer và state append bằng torch thuần.
- [x] Chạy lại test; kiểm tra gradient tới adapter/compressor/indexer.

### Task 2: MR draft model và khởi tạo từ target

**Files:**
- Create: `src/MR_DFlash/mr_model.py`
- Modify: `src/MR_DFlash/model.py`
- Test: `src/MR_DFlash/tests/test_mr_model_cpu.py`

**Interfaces:**
- `MRDraftSpec.from_dflash(spec, ...) -> MRDraftSpec`.
- `MRDFlashDraftModel.forward(noise_embedding, memory, position_ids, attention_mask) -> hidden`.
- `MRDFlashDraftModel.init_from_target(target_model) -> list[str]`.

- [x] Viết test tiny Qwen3 initialized draft: copied keys > 0, output `[B,N*block,H]`, finite output và nonzero gradient.
- [x] Chạy test để xác nhận fail.
- [x] Implement block-causal draft attention, HCA target attention, CSA selected target attention, FFN stages và RoPE-compatible positions.
- [x] Implement `MRDraftSpec` validation/defaults; copy common target layer keys và stable adapter initialization.
- [x] Chạy model test và test DFlash model cũ để đảm bảo không regress.

### Task 3: Config và training adapter

**Files:**
- Modify: `src/MR_DFlash/config.py`
- Modify: `src/MR_DFlash/training.py`
- Modify: `src/MR_DFlash/run_train.py`
- Modify: `src/MR_DFlash/trainer.py`
- Test: `src/MR_DFlash/tests/test_mr_train_smoke.py`

**Interfaces:**
- `ModelConfig.architecture` accepts `dflash|mr_dflash` plus MR fields.
- `OnlineMRDFlashModel` giữ signature `forward(input_ids, hidden_states, loss_mask)`.
- `MRDFlashTrainStrategy` giữ required features và checkpoint prefix `draft_model.`.
- `build_online_model` chọn model theo `architecture`.

- [x] Viết train smoke tiny Qwen3, synthetic offline features, 2 optimizer steps, finite loss, checkpoint và reload.
- [x] Chạy test để xác nhận fail.
- [x] Refactor phần objective DFlash dùng chung hoặc subclass an toàn; MR wrapper chỉ thay draft forward.
- [x] Thêm config defaults tương đương DFlash và validate ratio/top-k/window.
- [x] Nối initialization, warm-start và checkpoint filter cho toàn bộ MR trainable parameters.
- [x] Chạy MR train smoke rồi chạy lại `src/MR_DFlash/tests/test_smoke_cpu.py`.

### Task 4: Reference speculative inference

**Files:**
- Create: `src/MR_DFlash/inference.py`
- Modify: `src/MR_DFlash/capture.py`
- Test: `src/MR_DFlash/tests/test_mr_inference_cpu.py`

**Interfaces:**
- `MRDFlashInferenceEngine.prefill(input_ids) -> PrefillOutput`.
- `MRDFlashInferenceEngine.draft_block(state, input_ids, block_size) -> DraftOutput`.
- `MRDFlashInferenceEngine.verify(input_ids, proposed_ids, state) -> VerifyOutput`.
- `MRDFlashInferenceEngine.generate(input_ids, max_new_tokens) -> GenerationOutput`.

- [x] Viết test prefill/draft/verify với target tiny local; kiểm tra proposed shape, finite logits và rejected token không làm tăng memory length.
- [x] Chạy test để xác nhận fail.
- [x] Implement target hidden extraction, target logits verification và accepted-only incremental state update.
- [x] Thêm CLI `python -m MR_DFlash.inference` với checkpoint/model/config rõ ràng; không tự chọn GPU ngoài device truyền vào.
- [x] Chạy CPU inference test và CLI help/argument smoke.

### Task 5: Docs và GPU deferred experiment

**Files:**
- Modify: `src/MR_DFlash/README.md`
- Modify: `docs/mr_dflash.md`
- Create: `docs/mr_dflash_gpu_experiments.md`
- Modify: `src/MR_DFlash/configs/qwen3_8b_dflash_offline.yaml`

- [x] Ghi mapping DFlash kế thừa/MR code mới, API train/inference và config tương đương DFlash.
- [x] Ghi lệnh GPU placeholder yêu cầu người dùng cấp `CUDA_VISIBLE_DEVICES`, không chạy trong lượt này.
- [x] Ghi acceptance/latency/ROUGE protocol, seed, output schema và điều kiện không trùng job.
- [x] Chạy `python -m MR_DFlash.inference --help` và toàn bộ CPU test suite liên quan.
