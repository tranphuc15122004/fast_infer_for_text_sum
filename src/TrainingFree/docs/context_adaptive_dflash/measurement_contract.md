# Hợp đồng artifact và đo lường

Ngày: **2026-10-08**. [Thiết kế](design.md), [protocol](experiment_protocol.md). Schema mới là contract để triển khai, không sửa schema trace RECAP-KV.

## 1. Run layout và provenance

```text
outputs/context_adaptive_dflash/<run_id>/
  manifest.json
  split_manifest.json
  locked_config.json
  calibration.json
  requests.jsonl
  rounds.jsonl
  outputs_token_ids.jsonl
  comparison.json
  report.md
  logs/
```

`run_id` duy nhất; writer hiện có append, nên runner phải từ chối output đã tồn tại trừ resume hợp lệ. Resume chỉ tiếp tục request chưa hoàn tất, kiểm tra config/model/data hashes, không append summary vào giữa stream. Partial artifacts/error rows không được báo như completed run.

Manifest bắt buộc: `schema_version=cadflash.manifest.v1`, run/time/code commit và dirty state; target/draft/tokenizer config/weight fingerprint, prompt template hash, data file SHA256, source group IDs/split hash, runtime versions, GPU name/capability, actual attention backend, dtype, kernel/graph settings, seed, sampling/stopping, selector/layer/refresh, budgets/gammas/length mode, timing/online-update mode, calibration signature/hash, model-load and warmup scope. Không dump HF token hoặc toàn bộ master-env vào manifest.

Hash model assets một lần trước request timing. Timestamp/model load không thuộc run compatibility signature; checkpoint, runtime, dataset/prompt/sampling và cost-model signature phải khớp để dùng calibration. Chi phí hashing/setup không được giả thành steady-state inference.

## 2. Calibration và locked configuration

`calibration.json` có `schema_version=cadflash.calibration.v1`, signature/hash và provenance của calibration source groups. Fields bắt buộc:

| Field | Kiểu và validation |
|---|---|
| `entropy_cutpoints` | 3 floats finite, nondecreasing, signal temperature/vocabulary đi cùng signature |
| `prefix_priors` | Key selector/B/gamma/refresh/state bucket; `survival` length gamma, values `[0,1]` nonincreasing, unique-state support count và backoff origin |
| `cost_priors` | Key runtime/selector/B/gamma/context bucket/refresh; positive total round estimate, components finite nonnegative, repetition count, dispersion và timing scope |
| `supported_actions` | Actions đã pass shape/correctness gate, không chỉ actions được checkpoint khai báo |
| `observations_hash` | Hash của counterfactual/calibration trace tạo priors |
| `collection_scope` | Diagnostic round profile hay production-compatible profile, collector mode, setup/warmup exclusions |

Greedy repetitions trên cùng prefix/action chỉ tính một outcome cho prefix prior; timing mean dùng các repetitions thật. Null/nonfinite timing không tạo supported cost action. Cost profiles từ diagnostic collectors đổi kernel không dùng cho primary wall controller. Profile round có biên synchronize được phép làm surrogate prior nếu cùng production kernel/collector và khai báo overhead/scope; dev rollout wall-clock phải kiểm tra benefit trước khi lock, không coi profile latency bằng request latency.

`locked_config.json` có `schema_version=cadflash.locked.v1`: model/runtime/data/split/calibration hashes, complete config, primary variants, best fixed pair, source-group counts, repetitions/seeds, output scopes và G0–G5 evidence paths/hashes. Test loader từ chối override đổi behavior; chỉ cho device/output/resume overrides không đổi experimental signature. Files này phải tạo từ measured artifacts; chưa có instance calibration/locked config trong bộ tài liệu.

## 3. Request records

Mọi request/error/summary đi qua [io_util.JsonlWriter](../../../../scripts/common/io_util.py). Success records chứa toàn bộ `BASE_SCHEMA_KEYS` và `SPEC_SCHEMA_KEYS` hiện có. Writer không tự validate, nên runner gọi schema validation trước `add`.

```json
{
  "schema_version": "cadflash.request.v1",
  "type": "sample",
  "status": "ok",
  "method": "context_adaptive_dflash",
  "variant": "joint",
  "sample_id": "gov_report_example",
  "dataset": "gov_report",
  "split": "dev",
  "model": "Qwen3-4B",
  "input_tokens": 10240,
  "retained_tokens": 10240,
  "output_tokens": 512,
  "batch_size": 1,
  "selector_latency_ms": 4.0,
  "ttft_ms": 120.0,
  "tpot_ms": 8.0,
  "e2e_ms": 4208.0,
  "throughput_tok_s": 121.673,
  "qps": 0.23764,
  "peak_memory_gb": 14.0,
  "avg_accept_length": 4.2,
  "acceptance_rate": 0.4,
  "draft_latency_ms": 600.0,
  "verification_latency_ms": 3200.0,
  "rejected_draft_ratio": 0.6
}
```

**Chỉ là ví dụ schema, không phải kết quả.** Values minh họa không dùng để dựng report/claim; runner tính metrics từ counters/timestamps thật. `retained_tokens` là target prompt retained, luôn bằng `input_tokens`; draft budget ghi riêng. `peak_memory_gb` dùng convention GiB = bytes / `2**30`, kèm `memory_unit=gib`.

Bổ sung bắt buộc: `prompt_hash`, `config_hash`, `model_signature`, `run_id`, `generation_temperature`, `output_cap`, `stopped_by`, `output_ids_hash`, `greedy_exact_match`, `correctness_status`, `round_count`, `draft_tokens_proposed`, `draft_tokens_accepted`, `mean_draft_context_tokens`, `mean_gamma`, `signal_latency_ms`, `controller_latency_ms`, `bank_update_latency_ms`, `draft_prefill_ms`, `fallback_rounds`, `dense_refresh_rounds`, `dense_bank_bytes`, `gather_bytes`, `physical_key_tokens`, `speedup_valid` và boundary counters ở integration contract.

Error record chứa identifying metadata, `status=error`, `error_type`, message và component; các metrics không đo được là `null`, không là 0. Error record vẫn có base/spec keys với null values, không tham gia average success. `unsupported`, `source_span_unavailable`, `uncalibrated` hoặc `missing_reference` phải phân biệt bằng status/reason.

Nếu có reference, dùng `rouge.add_rouge(record,text,reference)` và `rouge.aggregate_rouge(success_records)` trong summary. ROUGE là quality report; không thay exactness check. Không có reference thì bỏ ROUGE keys, không tự gán 0.

## 4. Round trace

Schema `cadflash.round.v1`, type `round`, qua writer riêng và kết thúc bằng trace summary. Bắt buộc:

| Nhóm | Fields |
|---|---|
| Identity | `run_id`, `sample_id`, `variant`, `repetition`, `round_index`, `status` |
| Causal state | `logical_length_before`, `processed_output_before`, `pending_anchor_position`, `parent_query_origin_round`, `parent_query_index`, `parent_entropy`, `entropy_signal_temperature`, `source_concentration`, `history_acceptance`, `ranking_origin_round`, `ranking_age` |
| Action | `requested_budget`, `selected_context_tokens_by_layer`, `physical_context_tokens_by_layer`, `gamma_requested`, `gamma_executed`, `block_size`, `length_mode`, `selector_id`, `refresh_required`, `action_reason` |
| Outcome | `accepted_candidates=L`, `processed_commits=1+L`, `logical_length_after`, `all_candidates_accepted`, `prefix_right_censored`, `eos_offset`, `boundary_round`, `trimmed_commits` |
| Cost | `signal_ms`, `controller_ms`, `selection_ms`, `bank_update_ms`, `draft_ms`, `verify_ms`, `round_gpu_span_ms`, `round_host_ms`, `gather_bytes`, `cost_observation_ready`, `predicted_cost_per_commit`, `fallback_reason`, `wasted_work_ms` |

Trace chỉ giữ scalars/IDs/hashes; full attention/logits/tensor snapshots nằm trong diagnostic run riêng. Source positions có thể lưu hash + chunk IDs để giảm IO. Không lưu score của future round làm current state.

## 5. Timing

Primary `e2e_ms`: CUDA synchronize ngay trước và sau toàn request generation; wall clock bao gồm target prefill, initial draft-bank projection, all drafting/verification, entropy/score extraction, gather, controller, refresh, fallback/wasted work và boundary finalization. Model load/JIT warmup và JSON serialization sau request nằm ngoài generation scope; benchmark công bố rõ phạm vi.

TTFT: từ request timer start tới first target-selected output token sẵn sàng; không chờ first draft. Nếu metadata/source mapping cần chuẩn bị trước generation, báo `request_prepare_ms` riêng và thêm `request_total_ms=prepare+e2e` cho deployment scope. Không chuyển online ranking sang preparation để giấu overhead.

`decode_ms=max(e2e_ms-ttft_ms,0)`. `tpot_ms=decode_ms/(output_tokens-1)` khi output>1; trường hợp <=1 là null. `throughput_tok_s=1000*output_tokens/e2e_ms`, `qps=1000/e2e_ms`. Decoder-only throughput phải có tên riêng; không gán nó vào E2E throughput.

Draft prefill/project bank nằm trong E2E; `draft_prefill_ms` là component, không trừ lần nữa khỏi decode. Component GPU time dùng CUDA events, CPU controller time dùng wall clock; có thể overlap nên **không ép sum component bằng E2E**. `round_gpu_span_ms` gồm gaps stream có thể do host dispatch; không cộng thêm CPU time vào span đó để cập nhật total cost. `J` dùng estimate total round cost hoặc component estimates có accounting rõ, không double count.

Mode `diagnostic`: được đồng bộ từng round để audit/cost profiling; mọi control dùng cùng instrumentation. Mode `wall`: primary matrix, chỉ đồng bộ biên request, events đọc khi ready; không gọi `synchronize` mỗi round chỉ để tính controller cost. Calibration profile ghi timing mode và controller lấy timing-compatible priors. `frozen_cost` là fallback hợp lệ khi không có async total-round observation; không gọi nó online latency adaptation.

## 6. Counter và memory

- Acceptance rate: `sum(L)/sum(gamma_executed)` trên draft rounds; AR gamma 0 không đóng góp denominator.
- Average accept length trong shared schema: mean `1+L` trên các speculative rounds, đúng convention DFlash; báo `mean_accepted_candidates` riêng.
- Prefix survival analysis dùng đúng shape/action và censoring, không gán suffix sau mismatch thành token-level rejects.
- Output count đối soát: `processed_commits + final_pending_emitted_count - trimmed_commits == output_tokens`. Prefill anchor đã được chọn nhưng thường được xử lý trong round đầu, không cộng lần hai.
- EOS/cap boundary rounds được ghi và tính E2E, nhưng tách khỏi stationary cost-model fit.
- Peak allocated/reserved CUDA bytes: reset trước request sau model load, đồng bộ trước đọc. Báo model-resident baseline, full target KV, full draft bank và scratch riêng nếu đo được; không trừ model weights khỏi `peak_memory_gb` shared metric.

## 7. Summary và so sánh

`requests.jsonl` kết thúc bằng `type=summary`, status `complete|partial|error`, số requested/success/error/unsupported, variant, signature, metric aggregates, ROUGE aggregates, output equivalence và gate decision. Trace writer có summary riêng. Không gộp các variant hoặc repetition khác nhau vào một speedup không có pairing.

Pair bằng `(source_group_id,prompt_hash,model_signature,sampling_signature,output_cap,repetition,timing_mode)`. Natural greedy output phải bằng nhau; mismatch/failed row invalidates comparison và được liệt kê, không âm thầm loại khỏi headline. Fixed-output scope cũng yêu cầu same token IDs, không chỉ cùng count.

Report per-dataset và macro mean đều dataset; primary statistic là geometric mean paired `baseline_e2e/variant_e2e`, kèm median, P95 latency và paired 95% bootstrap CI. Bootstrap 10.000 lần, seed 42, cluster theo source document giữ mọi repetition/length variant cùng cluster. Không bootstrap round như các samples độc lập. Báo CI geometric speedup và uncertainty của component/acceptance metrics riêng.

Sampling distribution audit là gate riêng; same-seed trajectory mismatch trong production RNG không tự invalidates distribution-equivalent scheme. Không dùng unmatched sampled output length làm headline latency comparison; fixed-token workload hoặc debug coupled sampler phải có scope riêng.
