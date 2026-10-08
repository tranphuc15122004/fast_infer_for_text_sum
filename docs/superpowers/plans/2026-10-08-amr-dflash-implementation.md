# Kế hoạch triển khai AMR-DFlash

> **Trạng thái 08/10/2026:** đã triển khai một V0 rút gọn cho T1–T11 và có CPU contract tests. Các checkbox dưới đây là acceptance gate đầy đủ; chưa đánh dấu chỉ vì code path tồn tại. T12 và mọi kiểm chứng bằng model thật/B200 còn pending.

**Mục tiêu:** xây AMR memory interface trên pretrained DFlash-5L, train theo acceptance preferences, chạy full integrated pilot với full-context target verifier và measured cost.

**Kiến trúc:** core riêng `src/AMR_DFlash`, giữ nguyên weights/depth/block. Selection và incremental slots đi qua memory adapter; dense và AMR dùng shared evaluator/verifier. Reference executor phục vụ labels/correctness; cached executor phục vụ latency.

**Stack:** Python 3.12, PyTorch, Transformers, PyYAML và các dependency đã có.

**Đặc tả:** [thiết kế](../specs/2026-10-08-amr-dflash-design.md), [data/train](../../../src/ARMdflash/AMR_DFlash_Implementation/data_training_contract.md), [protocol](../../../src/ARMdflash/AMR_DFlash_Implementation/experiment_protocol.md).

**Evaluation:** mỗi 50 optimizer steps và cuối phase, toàn validation manifest; một evaluator cho checkpoint/in-memory và fixed-state/rollout. CPU synthetic override cadence=1; smoke không là scientific evidence.

## Kết quả implementation V0

Đã có `src/AMR_DFlash/` cho selector/compressor, DFlash adapter, sparse memory, cached greedy verifier, candidate generation/labels/preferences, selector/compressor training, fingerprinted checkpoints và GPU-hour ledger. CLI tại `scripts/amr_dflash/cli.py` nối `preflight`, `capture`, `candidates`, `label`, `train-selector`, `train-compressor`, `infer`; launcher được nối vào `scripts/run.sh amr_dflash`.

CPU synthetic tests đã xác nhận dense forward parity và token parity với target greedy cho dense/hybrid/compressor modes, gồm rejection/EOS/output cap; fixed-state evaluator có tiny-model test. V0 chưa tự chạy full validation theo cadence, optimizer/RNG resume, measured crossover gate, genuine GPU performance results hay B200 model run. Vì vậy các checklist bên dưới vẫn là acceptance gates còn mở, không phải todo cho các file chưa được tạo.

Một số đường dẫn test/entrypoint trong checklist chi tiết bên dưới là mục tiêu ban đầu của thiết kế. Lệnh CPU hiện chạy được là `.venv/bin/python -m pytest -q tests/test_amr_dflash_contracts.py tests/test_amr_dflash_launcher.py`; launcher B200 dùng `bash scripts/run.sh amr_dflash ...`.

## Ràng buộc toàn cục

- Tài liệu tiếng Việt; Python **3.12**; server dùng `python3` hệ thống.
- Local chỉ CPU qua `.venv/bin/python`; không sửa GPU/driver local.
- Không thêm dependency mới, installer online hoặc venv từng baseline.
- Snapshot local, offline flags, `local_files_only=True`.
- DFlash **5 layer**, **16 live positions = 1 anchor + 15 proposals**; feature metadata từ checkpoint.
- Target/backbone frozen; không giảm depth hoặc đổi block trong pilot.
- Raw keys giữ original positions/RoPE; live block không prune.
- V0 greedy/batch 1; sampling và batch>1 phải fail nếu chưa có executor.
- Generation qua `JsonlWriter` + summary, ROUGE qua helper chung.
- Không sửa `externals/dflash`; không đổi default dense baseline.
- Artifact/tensors/checkpoints không commit.
- Dense mask/reference collector timing không chứng minh acceleration.

## 1. Hạ tầng và dependency

Đọc `externals/dflash/dflash/model.py`, `scripts/infer_dflash.py`, `scripts/common/io_util.py`, `scripts/common/rouge.py`, `scripts/common/config.sh` và `scripts/common/runtime.sh` trước task liên quan.

Tham khảo feature-store atomic/lazy patterns trong `src/Finetuning/features.py` và `src/MR_DFlash/offline_features.py`. Không import MR two-stage model làm pretrained backbone.

~~~mermaid
flowchart LR
    T1["T1 contracts/config/metrics"] --> T2["T2 backbone"]
    T2 --> T3["T3 sparse reference"]
    T3 --> T4["T4 state store"]
    T4 --> T5["T5 labels"]
    T5 --> T6["T6 selector"]
    T2 --> T7["T7 compressor"]
    T6 --> T8["T8 evaluator/loss"]
    T7 --> T8
    T8 --> T9["T9 train integration"]
    T9 --> T10["T10 cached execution"]
    T10 --> T11["T11 launcher"]
    T11 --> T12["T12 pilot/ablation"]
~~~

T9 là task integration training duy nhất. T12 là thực nghiệm, không thêm training pipeline thứ hai.

## 2. Kiểm thử khi code đã có

Từ repo root, local:

~~~bash
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/externals/dflash"
.venv/bin/python -m pytest src/AMR_DFlash/tests -q
~~~

Mỗi task viết semantic tests, chạy trước implementation để thấy failure đúng behavior còn thiếu, rồi chạy lại. Các paths/lệnh AMR trong kế hoạch **chưa tồn tại** cho đến task tương ứng.

Fixture `src/AMR_DFlash/tests/fixtures.py`: tiny Qwen3 target 8 layer, hidden 64, intermediate 128, vocab 128, 4 Q heads/2 KV heads; DFlash 5 layer, feature IDs `[1,2,3,4,5]`, block 16. Seed 17, CPU FP32, eval mode; tạo từ config, không download. Core không import test modules.

## Task T1: Config, contracts và accounting

**Tạo:** `src/AMR_DFlash/__init__.py`, `config.py`, `contracts.py`, `metrics.py`, `tests/test_contracts.py`, `tests/test_metrics.py`.

**Input:** tensor/state spec.
**Output:** `AMRConfig`, `DraftState`, `MemoryBank`, `WorkingMemory`, `DraftOutput`, `VerifyResult`; `accepted_prefix_length`, `clip_emitted_ids`, `aggregate_decode`.

~~~python
def accepted_prefix_length(proposals, target_choices):
    equal = proposals.eq(target_choices)
    return int(equal.long().cumprod(dim=-1).sum().item())

def aggregate_decode(committed_counts, decode_seconds):
    seconds = sum(decode_seconds)
    return sum(committed_counts) / seconds if seconds > 0 else None
~~~

- [ ] Test đúng 3 proposals đầu, sai vị trí 4, đúng phần sau → A=3.
- [ ] Test `aggregate_decode([8,4], [2,4]) == 2`; không reciprocal mean TPOT.
- [ ] Test EOS/cap clipping, unique selected positions, future rejection, raw+slot budgets và invalid config.
- [ ] Implement fail-fast contracts; explicit unsupported sampling/batch errors.
- [ ] Chạy `.venv/bin/python -m pytest src/AMR_DFlash/tests/test_contracts.py src/AMR_DFlash/tests/test_metrics.py -q`.

**Nghiệm thu:** emitter sum G bằng output tokens; A raw/committed có semantics rõ.

## Task T2: Load backbone và dense identity

**Tạo:** `backbone.py`, `model.py`, `tests/fixtures.py`, `tests/test_backbone.py`.

**Input:** AMRConfig, local target/draft snapshots.
**Output:** `AMRDFlashBackbone`, `AMRModel`; `project_context`, `original_forward`, `draft_forward`. Giữ reference tới pretrained modules; memory attach riêng.

~~~python
projected = draft.hidden_norm(draft.fc(state.target_features))
live_ids = torch.full((1, 16), mask_token_id, dtype=torch.long)
live_ids[:, 0] = state.anchor_id
live_positions = state.anchor_position + torch.arange(16).unsqueeze(0)
# original_forward dùng target embedding/head, lấy proposal rows 1:16.
~~~

- [ ] Tạo tiny fixtures, strict load/save test; thiếu pretrained key phải fail.
- [ ] Freeze target/draft; validate depth/block/feature fingerprint; không converter MR two-stage.
- [ ] Implement frozen projection/dense forward; không fc/norm hai lần.
- [ ] Test bypass logits/argmax khớp original và weights không đổi khi attach memory.
- [ ] Chạy `.venv/bin/python -m pytest src/AMR_DFlash/tests/test_backbone.py -q`.

**Nghiệm thu:** tiny dense identity; kiểm tra checkpoint thật trên server khi thực thi GPU.

## Task T3: Sparse raw interface và reference verifier

**Tạo:** `memory.py`, `verifier.py`, `inference.py`, `tests/test_raw_memory.py`, `tests/test_reference_verifier.py`.

**Input:** projected context, original positions, target state.
**Output:** `build_bank`, `build_working_memory`, `ReferenceGreedyVerifier`, `verify_greedy` và emitter. `TargetCacheState` định nghĩa trong verifier.

~~~python
k_raw = layer.k_norm(layer.k_proj(raw_features).view(1, -1, kv_heads, head_dim))
v_raw = layer.v_proj(raw_features).view(1, -1, kv_heads, head_dim)
# Rotate raw K tại raw_positions; Q/live K tại live_positions riêng.
# Attention raw + toàn live block; giữ pretrained o_proj/residual/MLP/norm.
~~~

- [ ] Test all-context raw path khớp original dense khi slots off.
- [ ] Test positions `[0,3,7]` khớp dense-masked diagnostic cùng visibility trên năm layer; fixture phát hiện compact renumbering.
- [ ] Implement reference target greedy verify và context-before-anchor normalization.
- [ ] Test reject đầu/giữa/full acceptance, EOS anchor/proposal/correction, remaining=1 và target AR parity.
- [ ] Test rejected features không append; anchor đã emit không đếm lại.
- [ ] Chạy `.venv/bin/python -m pytest src/AMR_DFlash/tests/test_raw_memory.py src/AMR_DFlash/tests/test_reference_verifier.py -q`.

**Nghiệm thu:** sparse semantics đúng; reference timing gắn instrumented.

## Task T4: State store, split và capture

**Tạo:** `data.py`, `scripts/amr_dflash/capture_states.py`, `tests/test_data.py`.

**Input:** document manifest, target trajectories, checkpoint fingerprints.
**Output:** `StateStore`, `capture_states`, document feature shards và states.jsonl.

~~~python
prefix_features = document_features[:, :state.context_length]
prefix_positions = document_positions[:, :state.context_length]
assert prefix_positions.max().item() < state.anchor_position
~~~

- [ ] Test ID/content hash split overlap rejection.
- [ ] Test future suffix không ảnh hưởng replay; summary không tăng dataset length.
- [ ] Implement document sharding/lazy slices, atomic COMPLETE và fingerprint resume.
- [ ] Capture pre-anchor state với emitter/cursor/remaining; phase log và failure counts.
- [ ] Chạy `.venv/bin/python -m pytest src/AMR_DFlash/tests/test_data.py -q`.

**Nghiệm thu:** replay deterministic; không nhân bản full features theo candidates.

## Task T5: Candidate search và preferences

**Tạo:** `candidates.py`, `preferences.py`, `scripts/amr_dflash/label_candidates.py`, `tests/test_preferences.py`.

**Input:** immutable states, reference verifier, seeds/budgets.
**Output:** `generate_candidates`, `label_candidates`, `build_preferences`.

~~~python
gap = positive.accepted_proposals_committed - negative.accepted_proposals_committed
if gap >= 1 and not positive.censored and not negative.censored:
    pair = (positive.candidate_id, negative.candidate_id)
~~~

- [ ] Test cardinality/guards/seeds, oracle labels và dedup.
- [ ] Implement position/random candidates và 32-token swaps; tối đa 12 candidates/8 pairs mỗi state.
- [ ] Verify từ target state sạch mỗi candidate; record A/G/survival/ties/censoring/failures.
- [ ] Test iteration order không đổi labels; fixture bắt cache contamination.
- [ ] Empty preference dataset dừng với headroom report.
- [ ] Chạy `.venv/bin/python -m pytest src/AMR_DFlash/tests/test_preferences.py -q`.

**Nghiệm thu:** labels có provenance; current-round teacher không vào inference inputs.

## Task T6: State-conditioned selector

**Tạo:** `selector.py`, `losses.py` phần preference, `tests/test_selector.py`.

**Input:** pre-draft query, compact index, preferences.
**Output:** `AcceptanceSelector`, `select_context`, `preference_loss`, hard Top-K support.

~~~python
scores = (query.unsqueeze(1) * index).sum(-1) / (index.shape[-1] ** 0.5)
remaining = max(raw_budget - len(guard_positions), 0)
# Top-K valid non-guard positions, union guards, sort absolute positions.
~~~

- [ ] Implement index/query dim 64, distance/prompt-output biases và ranking loss.
- [ ] Test future labels/target choices không đổi query inputs.
- [ ] Test guards trong budget, padding excluded và permutation-consistent positions.
- [ ] Fit toy preferences: hai query cần hai vùng khác nhau; đánh giá hard selected-set utility.
- [ ] Chạy `.venv/bin/python -m pytest src/AMR_DFlash/tests/test_selector.py -q`.

**Nghiệm thu:** preference gradient tới scorer; plain gather không bị gọi là differentiable index.

## Task T7: Incremental compressor và slots

**Tạo:** `compressor.py`, `tests/test_compressor.py`; mở slot branch trong backbone/memory.

**Input:** projected context, positions, capacity/M.
**Output:** `StreamingBucketCompressor`, `SlotAdapter`, accumulator/slot positions/valid mask.

~~~python
m_new = torch.maximum(m_old, a_new)
old_scale = torch.exp(m_old - m_new)
new_scale = torch.exp(a_new - m_new)
numerator_new = numerator_old * old_scale + value_new * new_scale
denominator_new = denominator_old * old_scale + new_scale
slot = numerator_new / denominator_new.clamp_min(1e-8)
~~~

- [ ] Test full build == token/chunk append FP32 tolerance 1e-5.
- [ ] Implement learned pooling/residual rank 16, capacity errors và valid midpoint.
- [ ] Test no unseen bucket position, no rejected-token accumulator change.
- [ ] Implement slot adapters/k_norm/RoPE/log-sigmoid gate -4; disabled slots bỏ thật.
- [ ] Test nonzero memory gradient qua frozen draft; target/backbone grads None.
- [ ] Chạy `.venv/bin/python -m pytest src/AMR_DFlash/tests/test_compressor.py -q`.

**Nghiệm thu:** streaming equivalence; invalidate state khi weights đổi; chưa claim complementarity.

## Task T8: Shared evaluator và alignment loss

**Tạo:** `evaluation.py`, `checkpoint.py`, `scripts/amr_dflash/evaluate.py`, `tests/test_evaluation.py`; thêm alignment loss ở `losses.py`.

**Input:** in-memory AMRModel hoặc restored checkpoint, validation store/documents.
**Output:** `EvalReport` và `evaluate_model(model, states, mode)`, checkpoint CLI gọi cùng core.

~~~python
draft = model.draft_forward(state, working_memory)
candidate_ids = draft.logits.detach().argmax(-1)
# Target rows đúng position/candidate prefix, detached.
loss = alignment_loss(draft.logits, aligned_target_logits, proposal_valid)
~~~

- [ ] Implement fixed_state/rollout, full validation, logs/progress/error status.
- [ ] Test target row mapping: shift một token fail fixture; giữ block semantics.
- [ ] Test checkpoint/in-memory metrics/output đồng nhất; total tokens/time aggregation.
- [ ] Test empty validation, partial/missing checkpoint, fingerprint mismatch, NaN.
- [ ] Per-state/per-document JSONL + summary và ROUGE khi có reference.
- [ ] Chạy `.venv/bin/python -m pytest src/AMR_DFlash/tests/test_evaluation.py -q`.

**Nghiệm thu:** shared evaluator; surrogate/actual acceptance báo riêng.

## Task T9: Pipeline huấn luyện hoàn chỉnh [INTEGRATION]

**Tạo:** `training.py`, `run_train.py`, `configs/pilot.yaml`, `configs/synthetic.yaml`, `tests/test_training_integration.py`.

**Input:** T1–T8 và data/label contract.
**Output:** phase-specific training, checkpoint/resume và step-based full evaluation.

~~~python
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(trainable_parameters, config.max_grad_norm)
optimizer.step()
# Theo optimizer step cadence: evaluate shared core trên full validation.
~~~

- [ ] CPU data→selection/slots→draft→loss→backward→step fixture, 2 steps/phase.
- [ ] Phase selector/compressor có optimizer groups đúng; assert frozen params excluded.
- [ ] Implement seeds/accumulation/LR/grad norm/progress/human log/JSONL/GPU-hours.
- [ ] Atomic COMPLETE checkpoint, RNG/sampler/optimizer restore; resume khớp uninterrupted CPU run.
- [ ] Full eval mỗi 50 steps/cuối phase, checkpoint/in-memory modes và failure handling.
- [ ] Log memory/step throughput; MFU null+reason nếu chưa có FLOPs denominator.
- [ ] Chạy `.venv/bin/python -m pytest src/AMR_DFlash/tests/test_training_integration.py -q`.
- [ ] Chạy `.venv/bin/python -m AMR_DFlash.run_train --config src/AMR_DFlash/configs/synthetic.yaml --phase selector --device cpu` và lặp `--phase compressor`.
- [ ] Review static/gradient/row mapping; runtime validation server ngắn ≤5 phút trước mở pilot GPU.

**Nghiệm thu:** train/eval/freeze/resume evidence; CPU loss giảm không chứng minh long-context gain.

## Task T10: Cached target và physical execution

**Tạo:** `tests/test_cached_inference.py`, `scripts/amr_dflash/profile.py`; sửa verifier/memory/inference trong AMR package.

**Input:** reference parity, trained memory, target KV/full positions.
**Output:** `CachedGreedyVerifier`, shared dense/AMR cached executor, stage profile.

~~~python
verified = target(block_ids, past_key_values=target_cache, use_cache=True)
# Crop processed accepted prefix, giữ pending anchor riêng.
# Chỉ persist target features của processed committed positions.
~~~

- [ ] Cached/reference/target AR parity qua nhiều rejection rounds trên tiny target.
- [ ] Cache crop/rollback, pending anchor, EOS/cap và emitter invariants.
- [ ] Genuine gathered K/V: attention đọc raw+valid slots+16, không N-key mask.
- [ ] Full raw-KV bank storage option; bytes index/features/slots/temporary/logical cursor.
- [ ] Auto gate trong `inference.py` đọc crossover artifact khóa theo hardware/backend/dtype/batch/checkpoint fingerprint. Chưa có calibration hoặc ngoài vùng đã đo → dense với reason; artifact sai fingerprint → fail.
- [ ] Test force dense/amr không bị gate đổi; short dense bypass không build AMR index/slots; lazy switching tính đủ build time.
- [ ] Spy test sau prefill không recompute growing target prefix; dense/AMR cùng verifier.
- [ ] CUDA events + CPU/wall time, bootstrap/refresh/gather/update; tránh sync mọi kernel.
- [ ] Chạy `.venv/bin/python -m pytest src/AMR_DFlash/tests/test_cached_inference.py -q`.

**Nghiệm thu:** physical execution sẵn sàng profile; full resident bank được ghi rõ.

## Task T11: Launcher và schema chung

**Tạo:** `scripts/infer_amr_dflash.py`, `scripts/runners/run_amr_dflash.sh`, `docs/baselines/amr_dflash.md`, `tests/test_amr_dflash_launcher.py`.

**Sửa:** `scripts/run.sh` dispatch; `scripts/common/config.sh` mapping/env defaults; `docs/model_baseline_matrix.md` experimental entry khi có code.

**Input:** cached executor/evaluator, master-env.
**Output:** `bash scripts/run.sh amr_dflash`, CPU synthetic smoke mặc định; full explicit.

~~~bash
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config amr_dflash
source "$ROOT/scripts/common/runtime.sh"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/externals/dflash"
exec "$FAST_INFER_PYTHON" "$ROOT/scripts/infer_amr_dflash.py" --config "$AMR_CONFIG" "$@"
~~~

- [ ] Test dispatcher/master DATA_INPUT/RUN_SAMPLES mapping, Python 3.12 và offline errors.
- [ ] YAML qua env `AMR_CONFIG`; target/data/output/device từ master; không sửa master.path.
- [ ] Base/spec schema, accept-length semantics, A/G, stages, summary và ROUGE.
- [ ] `--smoke` tiny CPU, `--mode dense|amr|auto`, unsupported batch/sampling errors.
- [ ] Regression default DFlash launcher.
- [ ] Chạy `.venv/bin/python -m pytest tests/test_amr_dflash_launcher.py tests/test_dflash_wrapper_contract.py tests/test_master_config_contract.py -q`.

**Nghiệm thu:** experimental baseline đúng master/runtime; chưa gọi paper-ready.

## Task T12: Pilot, ablation và gói kết quả

**Tạo:** `scripts/amr_dflash/run_matrix.py`, `scripts/amr_dflash/collect_results.py`, `src/AMR_DFlash/tests/test_experiment_matrix.py` và runbook trong `docs/experiments/`.

**Input:** locked splits/config/gates, T1–T11 và measured compute cap.
**Output:** deployable/oracle matrix, paired document results, decision.json/report.

~~~python
document_ratio = dense_decode_seconds / amr_decode_seconds
# Cluster bootstrap theo document, primary throughput sum(tokens)/sum(time).
~~~

- [ ] CPU test oracle/reference exclusion, matched raw+slot budgets, failed samples.
- [ ] P0 sanity 3 dev docs; capture/label mới theo split đã khóa.
- [ ] Full AMR train trong cap; force dense/AMR eval rồi calibrate gate.
- [ ] Ablation selection/compression/attention imitation/reuse và fine-tune cùng compute.
- [ ] Khóa variant trước holdout; parity/paired timing/bootstrap 2000 document resamples.
- [ ] Report prefill/decode/E2E/A/G/survival/resident memory/GPU-hours/limitations.
- [ ] Chạy `.venv/bin/python -m pytest src/AMR_DFlash/tests/test_experiment_matrix.py -q` trước server matrix.
- [ ] Điền [checklist](../../../src/ARMdflash/AMR_DFlash_Implementation/acceptance_checklist.md) bằng artifact, quyết định go/simplify/no-go.

**Nghiệm thu:** scientific claim có measured holdout evidence; no-gain là outcome hợp lệ.

## 3. Đóng task và truy vết proposal

Ghi files thay đổi, commands/exit codes, artifact và limitations. Nếu commit, stage đúng files của task; không `git add .` hoặc commit working-tree changes khác của người dùng.

| Yêu cầu | Task |
|---|---|
| Pretrained/frozen 5L, block 16 | T1–T3 |
| Position-safe local/dynamic selection | T3,T6 |
| Incremental complementary slots | T7 |
| Actual acceptance preferences | T4–T6 |
| Surrogate/gradient/train integration | T7–T9 |
| Shared eval/checkpoint/resume | T8–T9 |
| Cached full-context verification | T3,T10 |
| Cost gate/short-context protection | T10–T12 |
| Matched budget/compute/holdout | T12 và protocol |
| Offline runtime/schema/ROUGE | T1,T4,T11 |

T9 complete kỹ thuật không đồng nghĩa T12 scientific success. GPU chưa chạy phải ghi chưa chạy.
