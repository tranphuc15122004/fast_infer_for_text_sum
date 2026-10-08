# Hợp đồng dữ liệu và huấn luyện AMR-DFlash

Ngày: **08/10/2026**. Trạng thái: **pipeline V0 đã có code và CPU contract tests; model thật/B200 chưa chạy**. Đây là mô tả artifact hiện hành; các acceptance gate còn mở được ghi riêng bên dưới.

Đọc cùng [đặc tả kỹ thuật](../../../docs/superpowers/specs/2026-10-08-amr-dflash-design.md) và [protocol](experiment_protocol.md). Proposal gốc xác định research intent; tài liệu này quy định artifact và cách hiện thực loss.

## 1. Pipeline và artifact

~~~text
Document manifest đã khóa split
  → target greedy trajectory + state snapshots
  → diverse candidate sets tại cùng state
  → frozen DFlash + full target verification
  → acceptance preferences
  → train selector
  → train global-slot compressor
  → rollout inference/evaluation qua schema benchmark chung
~~~

Artifact dưới `outputs/amr_dflash/<run_id>/`:

~~~text
manifest.json
documents.jsonl
states.jsonl
features/<document-id>.pt
candidate_positions/<state-id>.pt
candidates.jsonl
candidate_labels.jsonl
preferences.jsonl
teacher/<state-id>.pt
train_selector.jsonl
train_compressor.jsonl
resource_ledger.json
evaluation/<mode>.jsonl
~~~

Mỗi `features/<document-id>.pt` chứa trajectory token IDs, projected target-feature bank và các state được capture. `states.jsonl` tham chiếu bundle cùng `state_index`; mặc định capture đều tối đa 6 state/document. Candidate positions và teacher logits nằm ở file tensor riêng. Checkpoint atomic dưới `checkpoints/amr_dflash/<run_id>/<phase>.pt`; hiện có optimizer snapshot để provenance, chưa có optimizer/RNG/sampler resume. Không copy tensors vào Git.

Tên file tensor hiện dùng prefix đọc được cộng SHA-256 của **toàn bộ ID gốc**,
không cắt mất hậu tố state. Các path minh họa trong tài liệu là tên rút gọn.

Benchmark rollout và fixed-state evaluation dùng `JsonlWriter` với summary `record_type="summary"`. Pipeline indexes/labels dùng atomic `write_jsonl`; các file có summary cũng đánh dấu `record_type="summary"`, còn train logs kết thúc bằng `phase_summary`. Reader bỏ summary records trước khi đếm sample. Internal state/preference rows có schema riêng; base benchmark schema bắt buộc trên generation records.

## 2. Fingerprint bắt buộc

`manifest.json` hiện ghi:

- `schema_version`, hash của input JSONL, resolved path và fingerprint SHA-256 đầy đủ cho target/draft snapshot (weights, config, tokenizer files).
- Target feature-layer IDs, DFlash block size, dtype/backend, greedy temperature, input/output caps, seed.
- Split policy, memory budgets, local/query windows, alignment history và resource-ledger path/capture GPU-hours.
- Mỗi document có input-prompt hash và split; checkpoint có target/draft fingerprints, layer IDs, block size, memory config và index dimension.

`capture_contract` version 2 khóa input hash, model bytes, dtype/backend,
memory/index config, caps token/state, sampling policy, seed, split fractions,
prompt/truncation policy và alignment history. `capture_contract_sha256` đi
theo documents/states/candidates/labels/preferences/teacher/checkpoint.
Feature bundle có `feature_contract` riêng gồm target/draft SHA-256, layers,
hidden offset, dtype và backend; thay weights projection drafter cũng bắt
buộc capture lại. Model snapshot fingerprint tính cả weight index JSON.

Candidate positions và teacher logits có file digest trong label; preferences
khóa hash của `candidate_labels.jsonl`. Các pha kiểm tra state/document/split
và hợp đồng artifact trước khi dùng dữ liệu. Teacher còn khóa state/candidate,
feature contract và proposal history. Model/checkpoint được chuyển mount nếu
file hashes/config giữ nguyên; `resolved_path` chỉ là provenance.

ID document phải duy nhất trên toàn input. Source ID (`source_document_id`,
`document_id`, `source_id`) hoặc nội dung source trùng giữa train/validation/
holdout bị từ chối. Validation chạy trước cắt `max_samples`; split được khóa
theo source ID khi có nhiều prompt của cùng source.

Seed Python/Torch/CUDA được đặt trước khi load/khởi tạo runtime, gồm selector
và compressor. Checkpoint ghi `training_seed` và trạng thái deterministic
algorithms; chưa có cam kết bitwise giữa các CUDA kernel/stack khác nhau.

Manifest V0 chưa ghi git diff/code hash, đầy đủ chat-template arguments, EOS/PAD IDs hoặc content hash độc lập cho raw/reference fields; split map nằm trong `documents.jsonl`, chưa có `splits.json` riêng. Checkpoint loader fail nếu model/config/memory fingerprint không khớp. Không dùng tên model hoặc path giống nhau làm bằng chứng weights giống nhau.

## 3. Capture state và feature store

`DraftState` dùng convention context-before-anchor trong đặc tả. State index V0 gồm prefix length, pending anchor ID, split/document, prompt hash, bundle/index và remaining budget. Feature bundle giữ full target-greedy trajectory; query state dùng anchor embedding, feature mới nhất và trung bình của tối đa 16 feature gần nhất.

Capture trên greedy target trajectory; mặc định chọn tối đa 6 state cách đều, gồm hai đầu trajectory. Có thể đổi bằng `--max-states-per-document`. Một document nhiều states vẫn chỉ là **một đơn vị split/thống kê**.

Projected target features của một trajectory được lưu một lần theo document; states tham chiếu prefix slice trong bundle. Không nhân bản tensor cho mỗi candidate/state. Inference chỉ append features của anchor và accepted proposals đã được target xử lý; correction pending và rejected suffix chưa vào persistent bank.

Ví dụ index state tổng hợp:

~~~json
{
  "record_type": "state",
  "schema_version": "amr-v0",
  "state_id": "synthetic-doc-001-round-003",
  "document_id": "synthetic-doc-001",
  "split": "train",
  "bundle": "features/synthetic-doc-001-a1b2c3d4.pt",
  "state_index": 3,
  "context_length": 16384,
  "input_sha256": "<sha256-of-rendered-prompt>",
  "anchor_id": 101,
  "remaining_output_budget": 256,
  "feature_hidden_size": 12800,
  "selection_was_bypassed": false
}
~~~

Đây là ví dụ schema V0. Capture hiện ghi JSONL summary ở cuối index; loader pipeline bỏ summary. Tensor bundle dùng `torch.load` theo document, chưa dùng mmap/LRU và module train giữ bundle cache trong RAM; pilot cần theo dõi RAM khi tăng corpus.

`apply_chat_template(..., return_tensors="pt", return_dict=False)` là contract tensor ở Transformers 5.x. Lưu tokenized prompt cuối cùng và so hash khi replay.

## 4. Candidate search

Input: một immutable `DraftState`, raw budget, guard policy và seed.

V0 tạo recent, head, middle, random-window, random-scattered và chunk-swap candidates. Previous-support, DFlash-current attention và parent-target attention chưa được nối vào generator; không gắn tên các heuristic này cho candidate khác.

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

Record V0 gồm (candidate positions được lưu ở `candidate_positions/*.pt`; label lưu ở `candidate_labels.jsonl`):

~~~json
{
  "record_type": "candidate_label",
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
  "censored": false,
  "first_mismatch": 3,
  "correction_token_id": 42,
  "proposal_ids": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
}
~~~

`committed_tokens` trong label hiện là `A+1` cho một state không bị cắt; label đánh dấu censor khi EOS/correction hoặc output cap làm so sánh không tương đương. Candidate proposal IDs được lưu để audit first rejection. Candidate order được tuần tự hóa và target DynamicCache được crop về đúng prefix sau mỗi candidate.

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

Phase II freeze selector, train global-slot compressor + slot gate. Raw support của mỗi state/candidate được cố định từ phase labeling; global slots build từ prefix features bằng **weights hiện tại**. V0 chưa có per-layer slot adapters.

DFlash xuất 15 proposal logits trong một forward từ anchor + masks. Nó không tự conditioning tuần tự trên previous proposals như AR model. Không viết `q_D(y_j | candidate[:j-1])` nếu forward không thực sự nhận history đó.

V0 dùng **candidate-prefix alignment surrogate**. Label phase lưu target logits của candidate không bị censor có acceptance cao nhất cho state; train phase giữ nguyên candidate proposal IDs/support và dùng logits đã lưu. Teacher logits không được tái tính theo weights mới mỗi bước.

1. Label phase chạy DFlash frozen một lần/candidate để tạo proposals, sau đó target verifier có cache đầy đủ để lưu các target rows tương ứng.
2. Train phase chạy DFlash với cùng candidate raw support + slots hiện tại; các draft proposal logits được so với target logits đã lưu.
3. Tính KL từ target rows detach; trọng số `w_j ∝ 1/j`, j=1..15. Candidate proposal IDs không backprop.
4. Rollout inference đánh giá actual acceptance riêng; training loss không bảo đảm acceptance tăng.

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

V0 hiện dùng mask toàn true vì label phase luôn draft đủ 15 proposals; target logits phía sau first rejection là counterfactual rows. Đây là surrogate cho block drafting, **không phải KL giữa hai AR distributions cùng conditioning** và không là theorem bảo đảm acceptance. Manifest/checkpoint ghi `alignment_history=captured_verifier_candidates`.

Acceptance tests còn cần: shift proposal/logit rows một position phải bị phát hiện bằng fixture; zero padding không ảnh hưởng loss. CPU test hiện xác nhận gradient vào compressor/slot gate qua frozen DFlash và target/backbone grad None; per-layer slot adapters chưa có.

Không dùng `torch.inference_mode` cho features sẽ tham gia backward mà chưa chuyển thành ordinary detached tensors; không `no_grad` toàn bộ frozen DFlash forward khi học input memory.

## 8. Config pilot dự kiến

YAML hiện hành ở `src/AMR_DFlash/configs/pilot.yaml`. Tên env được resolve từ master trước run; thiếu env/path sẽ fail, không tải online.

~~~yaml
schema_version: amr-v0
model:
  target_model_env: TARGET_MODEL
  draft_model_env: DRAFT_MODEL
  draft_layers: 5
  block_size: 16
  attention_backend: sdpa
  dtype: bfloat16
inference:
  temperature: 0.0
  batch_size: 1
  mode: amr
  use_cost_gate: true
memory:
  raw_budget: 4096
  num_slots: 128
  local_window: 128
  query_window: 16
  index_dim: 64
  slot_gate_init: -2.0
  min_context_tokens: 4096
training:
  seed: 17
  max_states_per_document: 6
  max_candidates: 12
  selector_steps: 200
  compressor_steps: 200
  max_gpu_hours: 24
  alignment_history: captured_verifier_candidates
data:
  manifest_env: AMR_DATA_MANIFEST
  input_env: DATA_FILE
output:
  run_root_env: AMR_RUN_ROOT
  run_id_env: AMR_RUN_ID
~~~

Các LR/steps/budget là điểm khởi đầu để calibration, không hứa hội tụ trong 200 step hoặc 24 GPU-giờ. Resource ledger V0 trừ capture/label/train elapsed GPU allocation-hours; inference/evaluation benchmark chưa được tính vào cap. Khi chạy CPU synthetic, override dtype float32 và dùng tiny model; không load checkpoint 4B vào smoke.

## 9. Evaluation, checkpoint và resume

CLI V0 có `evaluate-fixed` để đo hard policy trên captured states giống nhau; command này chưa được gọi tự động sau mỗi 50 optimizer steps và chưa có calibration split/checkpoint resume đầy đủ. `train-selector` báo pairwise accuracy trên validation preferences; `train-compressor` log surrogate loss. CLI `infer` chạy greedy rollout trên input đã chỉ định và ghi schema/ROUGE. Fixed-state/rollout artifacts vẫn cần được tạo và audit riêng trước khi dùng làm scientific result.

V0 in progress mỗi 10 optimizer steps và ghi JSONL phase logs. Chưa có full validation schedule, per-step memory metrics, document/state failure ledger hoặc best-checkpoint logic. Non-finite loss/gradient và empty train pairs dừng phase bằng exception.

Checkpoint V0 chứa:

- `<phase>.pt`: selector/compressor/slot gate weights, metadata fingerprints, phase, run root và optimizer state khi có; không nhúng frozen target/draft weights.
- optimizer state và step trong cùng `.pt` với memory weights/metadata; atomic replace.
- Model/checkpoint fingerprints, phase, run root và loss alignment history.

Loader kiểm tra model/memory fingerprints và strict state dict. Hiện chưa có resume optimizer/RNG/sampler; dùng checkpoint selector làm warm-start cho compressor. Slot accumulators/index caches được dựng lại khi bắt đầu mỗi inference.

`capture --resume` yêu cầu đủ manifest/document/state indexes và capture
contract không đổi; có thể mở rộng `max_samples` trên cùng input file.
Manifest gốc giữ nguyên, ledger ghi tài nguyên cộng dồn. Đổi caps, seed,
split fractions, dtype/backend hoặc weights phải tạo run mới. Feature/manifest
cũ thiếu hợp đồng version 2 bị từ chối với hướng dẫn recapture; cần tạo lại
candidate/label/preference/checkpoint từ capture mới.

Training JSONL ghi loss, grad norm, batch pairs/teacher row, elapsed wall time, GPU-hour ledger ở phase summary. LR, peak VRAM, full evaluation status và MFU chưa được log trong V0; không suy diễn các chỉ số này từ training loss.

## 10. Các lỗi cần nghiệm thu trước khi scale

| Lỗi | Hành vi |
|---|---|
| Split overlap/content duplicate | Chưa có detector V0; input split/content hashes phải audit trước khi chạy |
| State/model fingerprint mismatch | Checkpoint loader fail model/config/state mismatch |
| Future target outcomes trong selector inputs | Query V0 chỉ dùng anchor + projected committed context; cần leakage audit trên artifact |
| Rejected feature trong committed bank | Cache crop giữ anchor + accepted prefix; CPU parity test đã cover |
| Không có preference pair hợp lệ | Selector train dừng nếu preferences rỗng; chưa xuất no-headroom report đầy đủ |
| NaN/Inf hoặc gradient vào frozen weights | Non-finite loss/gradient dừng phase; CPU test xác nhận frozen target/draft grad None |
| Eval restore hoặc aggregation lỗi | Fixed-state CLI có summary; chưa có train-time best-checkpoint/evaluation flow |
| Hết compute cap | Phase ghi status/ledger nếu thoát có kiểm soát; chưa hỗ trợ optimizer/RNG/sampler resume |
