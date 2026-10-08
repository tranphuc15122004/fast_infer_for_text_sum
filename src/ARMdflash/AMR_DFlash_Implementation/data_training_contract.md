# Hợp đồng dữ liệu và huấn luyện AMR-DFlash

Ngày: **08/10/2026**. Trạng thái: đặc tả, các API/lệnh AMR dưới đây **chưa được triển khai**.

Đọc cùng [đặc tả kỹ thuật](../../../docs/superpowers/specs/2026-10-08-amr-dflash-design.md) và [protocol](experiment_protocol.md). Proposal gốc xác định research intent; tài liệu này quy định artifact và cách hiện thực loss.

## 1. Pipeline và artifact

~~~text
Document manifest đã khóa split
  → target greedy trajectory + state snapshots
  → diverse candidate sets tại cùng state
  → frozen DFlash + full target verification
  → acceptance preferences
  → train selector
  → train complementary compressor/adapters
  → shared evaluation: fixed-state và rollout
~~~

Artifact dưới `outputs/amr_dflash/<run_id>/`:

~~~text
manifest.json
splits.json
documents.jsonl
features/manifest.json + shards/*.pt
states.jsonl
candidates.jsonl
preferences.jsonl
train_log.jsonl
evaluation/<checkpoint_id>/fixed_state.jsonl
evaluation/<checkpoint_id>/rollout.jsonl
profiling/
decision.json
~~~

Checkpoint dưới `checkpoints/amr_dflash/<run_id>/<phase>-step<step>/`. Manifest ghi cả hai root; không copy tensors vào Git.

JSONL index, trace, evaluation và training log dùng `JsonlWriter`, summary cuối có `record_type="summary"`. Dataset loader bỏ summary bằng record_type; không đếm summary như sample. Internal state/preference rows có schema riêng; base benchmark schema chỉ bắt buộc trên generation records.

## 2. Fingerprint bắt buộc

`manifest.json` chứa:

- `schema_version="amr-v0"`, run ID, code commit và working-tree diff hash nếu có.
- Target/draft/tokenizer resolved local paths, revision hoặc hash của weights/config/tokenizer.
- DFlash depth, block size, feature layer IDs, hidden-state offset convention, mask/EOS/PAD IDs.
- Prompt template chính xác và hash, chat-template arguments, source truncation policy, tokenizer options.
- Dtype/backend, temperature, hardware/runtime versions, input/output caps, random seed.
- Document IDs và content hashes theo split; `split_manifest_sha256`.
- Storage policy, candidate generator version, raw/slot budgets, slot position policy, label semantics.
- `label_generation_gpu_hours`, `training_gpu_hours` và `evaluation_gpu_hours` đo được; giá trị chưa đo là null.

Model weights và feature store không khớp fingerprint phải fail trước train. Không dùng tên model hoặc path giống nhau làm bằng chứng weights giống nhau.

## 3. Capture state và feature store

`DraftState` dùng convention context-before-anchor trong đặc tả. Snapshot gồm prefix đã xử lý, pending anchor, original positions, output emitter state và remaining budget.

Capture trên greedy target trajectory; lấy nhiều mốc đầu/giữa/cuối. Một document nhiều states vẫn chỉ là **một đơn vị split/thống kê**.

Target features của một trajectory được lưu theo document/shard một lần; states tham chiếu prefix slice. Không nhân bản tensor 16K × d_H cho mỗi candidate hoặc state. Với candidate branch, persist chỉ features của prefix thực sự accepted; rejected suffix không trở thành feature store.

Ví dụ index state tổng hợp:

~~~json
{
  "record_type": "state",
  "schema_version": "amr-v0",
  "state_id": "synthetic-doc-001-round-003",
  "document_id": "synthetic-doc-001",
  "split": "train",
  "feature_shard": "features/shards/00000.pt",
  "feature_key": "synthetic-doc-001",
  "context_length": 16384,
  "prompt_length": 16384,
  "anchor_position": 16384,
  "anchor_id": 101,
  "remaining_output_budget": 256,
  "state_fingerprint": "synthetic-fixture-v0"
}
~~~

Đây là fixture schema, không phải artifact chạy thật. Loader kiểm tra length, feature dim, positions, anchor offset, prompt/output scope và remaining budget dương. Dùng mmap/lazy shard load khi phù hợp; một worker không materialize toàn feature store.

`apply_chat_template(..., return_tensors="pt", return_dict=False)` là contract tensor ở Transformers 5.x. Lưu tokenized prompt cuối cùng và so hash khi replay.

## 4. Candidate search

Input: một immutable `DraftState`, raw budget, guard policy và seed.

V0 thử recent, head, middle, random-window, random-scattered, previous-support và chunk-swap. DFlash-current attention và parent-target attention là candidate offline nếu artifact hợp lệ; thiếu signal phải ghi `unavailable`, không thay bằng zeros hoặc recent rồi giữ tên method.

Candidates của cùng state:

- Có cùng raw budget, cùng guards, unique positions, không chứa live positions/padding/future.
- Chọn đủ min(budget,N) entries; partial chunk ở boundary phải accounting bằng số raw token thực.
- Deduplicate theo hash sorted positions; cùng tập nhưng khác heuristic không tăng effective candidate count.
- Nếu N ≤ budget và mọi candidate trùng full, state không cung cấp preference headroom; log là degenerate.
- Candidate attention oracle có `oracle=true` và `deployable=false`.

Chunk-swap V0 dùng chunk size 32, chọn một chunk trong support và một chunk ngoài support, giữ cardinality sau swap. Với candidate winner, thử tối đa 8 swaps/state; record cả negative/tie outcomes. Utility là set-level, không suy utility độc lập của từng token.

Generator cho phép tối đa 12 distinct candidates/state ở pilot. Đây là trần chi phí khởi đầu, không đảm bảo tìm được acceptance optimum.

## 5. Verification labels

Mỗi candidate dùng cùng weights, state, block positions, dtype và full target history. Chạy reference target forward để tạo labels; reset/restore target prefix giữa candidate, tránh cache bị candidate trước làm bẩn.

Record gồm:

~~~json
{
  "record_type": "candidate",
  "state_id": "synthetic-doc-001-round-003",
  "candidate_id": "recent-4096",
  "method": "recent",
  "positions_ref": "candidate_positions/00000.pt",
  "raw_budget": 4096,
  "oracle": false,
  "deployable": true,
  "accepted_proposals_raw": 3,
  "accepted_proposals_committed": 3,
  "committed_tokens": 4,
  "survival": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "status": "success",
  "timing_scope": "instrumented_reference",
  "seed": 17
}
~~~

G trong fixture là emitter outcome cụ thể, không là công thức suy từ A. Save proposal IDs/target verifier choices trong tensor trace để audit first rejection.

V0 greedy: reward chính là `accepted_proposals_committed` với cùng remaining budget; chỉ train preference từ states chưa terminal/censored nghiêm trọng. Raw A cũng lưu để chẩn đoán. Nếu EOS/output cap khiến reward không so được, gắn `censored=true` và loại khỏi preference fit, nhưng giữ cho correctness/evaluation.

Không rank theo instrumented latency của attention oracle. Latency dùng trong gate phải đến từ production-compatible measurements riêng.

Stochastic labels là extension: cần repeated samples và verifier/correction đúng distribution; không bật bằng cách đặt temperature >0 trên equality verifier.

## 6. Preference dataset và selector loss

`C+` tốt hơn `C-` khi accepted committed proposals khác ít nhất 1 ở cùng state/budget. Tie không đưa vào binary preference loss; log tie rate và fraction có headroom.

~~~json
{
  "record_type": "preference",
  "state_id": "synthetic-doc-001-round-003",
  "positive_candidate_id": "swap-007",
  "negative_candidate_id": "recent-4096",
  "reward_positive": 6,
  "reward_negative": 3,
  "weight": 1.0,
  "split": "train"
}
~~~

Mỗi state đóng góp tối đa 8 pairs. Sampler phân tầng document/state và normalize total pair weight mỗi state, tránh document có nhiều pairs áp đảo.

~~~python
def preference_loss(positive_scores, negative_scores, weights):
    # F(z,C) là mean valid token scores của candidate set.
    terms = torch.nn.functional.softplus(negative_scores - positive_scores)
    return (terms * weights).sum() / weights.sum().clamp_min(1e-8)
~~~

Preference train nhận input chỉ có sẵn trước draft. Các reward/proposals/target choices chỉ là label, không được concatenate vào query features.

Phase I freeze compressor và backbone; optimizer selector riêng. Diagnostic: pairwise accuracy, tie/drop rates, reward gap trên validation, actual A/G của hard Top-K policy. Pairwise accuracy cao chưa chứng minh Top-K set cuối tốt; bắt buộc đánh giá selected set thật.

## 7. Compressor alignment: giữ đúng block semantics

Phase II freeze selector, train compressor + slot adapters + slot gate. Raw support được chọn từ state; global slots build từ prefix features bằng **weights hiện tại**.

DFlash xuất 15 proposal logits trong một forward từ anchor + masks. Nó không tự conditioning tuần tự trên previous proposals như AR model. Không viết `q_D(y_j | candidate[:j-1])` nếu forward không thực sự nhận history đó.

V0 chọn **candidate-prefix alignment surrogate**:

1. Forward AMR DFlash một lần để có `q_j` và greedy candidates `y_j = argmax(q_j.detach())`.
2. Full target forward trên context + anchor + candidate block để có `p_T,j`, mỗi row đúng position và causal candidate prefix.
3. Tính KL từ target rows detach sang các block logits tương ứng; mặc định `w_j = (1/j) / sum(1/k)` với j=1..15 để ưu tiên early prefix.
4. Đánh giá actual accepted prefix bằng verifier. Candidate argmax/history không backprop; target không nhận gradient.

~~~python
def alignment_loss(draft_logits, target_logits, proposal_valid):
    # Hai tensor [B,15,V], map đúng proposal positions của verifier.
    log_q = torch.log_softmax(draft_logits.float(), dim=-1)
    p_t = torch.softmax(target_logits.detach().float(), dim=-1)
    per_position = torch.nn.functional.kl_div(
        log_q, p_t, reduction="none"
    ).sum(dim=-1)
    early = 1.0 / torch.arange(1, 16, device=log_q.device).float()
    weight = proposal_valid.float() * early
    return (per_position * weight).sum() / weight.sum().clamp_min(1e-8)
~~~

`proposal_valid` mask bỏ padding/positions ngoài valid horizon. Candidate-prefix target rows sau first rejection là counterfactual rows; đây là surrogate cho block drafting, **không phải KL giữa hai AR distributions cùng conditioning** và không là theorem bảo đảm acceptance. Ablate teacher-trajectory alignment nếu cần; luôn ghi `alignment_history` vào checkpoint/manifest.

Test bắt buộc: shift proposal/logit rows một position phải bị phát hiện bằng fixture; zero padding không ảnh hưởng loss; gradient nonzero vào compressor/adapters nhưng backbone/target grad là None.

Không dùng `torch.inference_mode` cho features sẽ tham gia backward mà chưa chuyển thành ordinary detached tensors; không `no_grad` toàn bộ frozen DFlash forward khi học input memory.

## 8. Config pilot dự kiến

YAML đặt ở `src/AMR_DFlash/configs/pilot.yaml` khi code được tạo. Tên env được resolve từ master trước run; thiếu env/path sẽ fail, không tải online.

~~~yaml
schema_version: amr-v0
model:
  target_model_env: TARGET_MODEL
  draft_model_env: DRAFT_MODEL
  local_files_only: true
  draft_layers: 5
  block_size: 16
  feature_layers_source: checkpoint
  dtype: bfloat16
inference:
  temperature: 0.0
  batch_size: 1
  mode: amr
  executor: cached
memory:
  raw_budget: 4096
  num_slots: 128
  local_window: 128
  sink_tokens: 0
  index_dim: 64
  selector_refresh_rounds: 1
  enable_selector: true
  enable_compressor: true
  slot_position_policy: observed_span_midpoint
training:
  seed: 17
  selector_steps: 200
  compressor_steps: 200
  selector_lr: 0.0001
  compressor_lr: 0.0001
  weight_decay: 0.01
  max_grad_norm: 1.0
  state_batch_size: 1
  gradient_accumulation_steps: 8
  evaluate_every_steps: 50
  checkpoint_every_steps: 50
  validation_scope: full
  alignment_history: candidate_prefix
  max_gpu_hours: 24
data:
  manifest_env: AMR_DATA_MANIFEST
output:
  run_id_env: AMR_RUN_ID
  checkpoint_env: AMR_CHECKPOINT
~~~

Các LR/steps/budget là điểm khởi đầu để calibration, không hứa hội tụ trong 200 step hoặc 24 GPU-giờ. Trần đề xuất gồm capture/label/train/eval; allocator phải trừ compute đã dùng trước khi mở pha tiếp theo. Khi chạy CPU synthetic, override dtype float32 và dùng tiny model; không load checkpoint 4B vào smoke.

## 9. Evaluation, checkpoint và resume

Evaluator chung `evaluate_model` phục vụ cả in-memory lúc train và checkpoint CLI. Mỗi 50 optimizer steps và cuối pha, đánh giá **toàn validation manifest đã khóa**, báo fixed-state A/G/survival và full greedy rollout. Có thể dùng manifest validation nhỏ ở pilot, nhưng không tự giảm scope khi thiếu VRAM.

Log phase-start, progress, phase-end, số document/state thành công/thất bại, loss/A/G, timing và memory. Metrics không hữu hạn, dataloader rỗng hoặc restore mismatch là lỗi có status; không ghi checkpoint "best" từ evaluation lỗi.

Checkpoint chứa:

- `amr_state.pt`: selector, compressor/adapters/gates, config và backbone fingerprint; không nhúng lại frozen target/draft weights.
- `trainer_state.pt`: optimizer, scheduler, RNG Python/NumPy/Torch/CUDA, optimizer step, phase và sampler cursor.
- `manifest.json`: split/data/code/config fingerprints, budget/position/loss semantics.
- `COMPLETE` viết cuối sau atomic publish.

Resume chỉ nhận checkpoint COMPLETE với đúng fingerprints. Warm-start weights-only là flag riêng, không khôi phục optimizer/sampler như resume. Slot accumulators/index caches phải rebuild khi checkpoint memory weights thay đổi.

Training log mỗi optimizer step: loss thành phần, grad norm, LR, number of states/pairs, peak VRAM, elapsed GPU-giờ và evaluation status. MFU chỉ báo khi có FLOPs estimate/hardware denominator được ghi; nếu chưa đo được, dùng null + reason thay vì bịa phần trăm.

## 10. Các lỗi phải dừng sớm

| Lỗi | Hành vi |
|---|---|
| Split overlap/content duplicate | Fail dataset publication |
| State fingerprint khác checkpoint | Fail load |
| Current/future target outcomes xuất hiện trong selector inputs | Fail schema/leakage audit |
| Rejected feature trong committed bank | Fail inference invariant |
| Không có preference pair hợp lệ | Dừng fit, ghi no-headroom/tie distribution |
| NaN/Inf hoặc gradient vào frozen weights | Dừng step, ghi diagnostic, không publish checkpoint |
| Eval restore hoặc aggregation lỗi | Status failed, không cập nhật best checkpoint |
| Hết compute cap | Save resumable checkpoint và quyết định dừng; không tự scale |
