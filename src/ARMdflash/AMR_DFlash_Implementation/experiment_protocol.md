# Giao thức thực nghiệm và quyết định AMR-DFlash

Ngày: **08/10/2026**. Đây là protocol cho thí nghiệm tiếp theo; V0 code và CPU synthetic tests đã có nhưng chưa có số AMR trên model/data thật.

Nguồn: [proposal](../AMR_DFlash_Research_Proposal_and_Paper_Story_2026-10-08.md), [đặc tả](../../../docs/superpowers/specs/2026-10-08-amr-dflash-design.md), [data/train](data_training_contract.md).

## 1. Câu hỏi và kết quả chính

| Giả thuyết | Phép kiểm tra quyết định |
|---|---|
| H1: context theo acceptance có utility khác attention Top-K | Same-state intervention, cùng budget và backbone |
| H2: selection theo state hơn static support | Learned vs fixed/recent/previous-attention trên rollout |
| H3: compression bổ trợ selection | Hybrid vs selection-only/compression-only ở matched-total-budget và matched cost |
| H4: acceptance gain bù overhead | Đo G/time, verifier calls và toàn stage cost |
| H5: bypass bảo vệ short context | Auto vs dense trên short-context holdout, tính build/switch cost |

Primary mechanism: accepted committed proposals A và survival `P(A≥j)`. Primary system outcome: `sum(G_decode)/sum(T_decode)`, cùng paired E2E latency. G_decode không bao gồm anchor đầu đã emit trong prefill; tổng G của mọi phase vẫn bằng output tokens. ROUGE/factuality là task sanity; chất lượng draft không tự làm target summary tốt hơn.

## 2. Chia dữ liệu trước khi chạy

Cohort 10 GovReport của D0–D4 là exploratory/dev. Dùng tối đa 3 document trong cohort đó cho P0 sanity, không dùng chúng làm untouched holdout.

Pilot mới đề xuất **60 GovReport document không trùng cohort cũ**:

- Train: 36 document.
- Validation: 12 document, evaluation toàn manifest mỗi 50 steps.
- Calibration: 12 document, chọn budget/gate/variant, không train.

Holdout mục tiêu: 30 GovReport + 30 Multi-News + 30 QMSum document mới, tách content hashes khỏi train/validation/calibration. Nếu thiếu dữ liệu/compute, ghi cohort nhỏ hơn trước khi mở holdout; không kết luận generalization ba task khi chỉ đo GovReport.

Same document IDs qua caps 3072/5120/8192/16384 khi có đủ độ dài. Cap ngắn làm nội dung thay đổi, nên thêm controlled distractor/position interventions để kiểm tra length/semantic confounding. Không tự khẳng định context length gây acceptance degradation.

Pilot capture mặc định tối đa 6 states/document, evenly spaced gồm hai endpoint; dùng `--max-states-per-document` để đổi cap. Trần 12 candidates/state và 8 pairs/state. Label phase đánh dấu EOS/output-cap censoring và bỏ censored pairs; calibration set và no-headroom report cần được tạo riêng.

Trước pilot, tạo input JSONL bất biến với `id` và explicit document-level `split` (`train`, `validation`, `holdout`), rồi ghi hash/run ID cùng artifact. V0 ghi split trong `documents.jsonl` và hash input JSONL trong `manifest.json`; chưa tạo/kiểm tra `splits.json`, chưa có split `calibration`, và chưa kiểm tra content overlap độc lập. Operator phải khóa split manifest bên ngoài trước label generation. Holdout không dùng chọn gate, slot count, LR hoặc early stopping.

## 3. P0 — Semantics và headroom sanity

Đầu vào: cùng Qwen3-4B/DFlash-b16 snapshot, target features đầy đủ; chỉ thay DFlash context.

Tại 16K, budget raw 1K/4K/8K:

| Variant | Mục đích |
|---|---|
| FULL | Baseline chất lượng với checkpoint gốc |
| RECENT / HEAD / MIDDLE | Position controls |
| RAND-WINDOW / RAND-SCATTERED | Random controls, 3 seeds |
| Current DFlash attention Top-K | Oracle attention offline, không là method tốc độ |
| Previous DFlash / parent-target Top-K | Heuristic khi signal sẵn có |
| Acceptance chunk-swap search | Ước lượng headroom trong search space hữu hạn |

Khóa state, target history, anchor, masks, RoPE, block và verifier. Báo A/G/logit agreement, first-rejection position và oracle cost.

Gate P0: all-context sparse forward khớp dense; greedy verifier khớp target AR; candidate state không làm bẩn cache; ít nhất có candidate pairs informative. Nếu không có headroom, ghi rõ search hữu hạn chưa bác bỏ mọi module học, nhưng không scale fit selector trên dataset toàn ties.

## 4. P1 — Full integrated pilot

Train selector từ actual preference labels; train global compressor/slot gate với frozen backbone. Chạy local+selected+slots và threshold bypass. V0 chưa có per-layer slot adapters hoặc calibrated crossover gate.

Đầu tiên force AMR/dense trên validation/calibration ở 8K/16K, output tối đa 256 token hoặc EOS. Short-context control 3K/5K. Giữ prompt/output protocol giống nhau; không ép sinh qua EOS hoặc cắt output mỗi method khác nhau để cân tốc độ.

Pilot initial config: raw 4096 + slots 128, local 128 trong raw budget. Gate lấy từ measured crossover; không giả định 8K luôn là ngưỡng tối ưu.

Mốc P1 thành công kỹ thuật: artifact đầy đủ, greedy parity đạt, gradient đúng, end-to-end rollout đến EOS/cap; **không đồng nghĩa scientific gain**.

## 5. P2 — Ablation sau integrated result

| Variant | Thay đổi so full AMR |
|---|---|
| Dense DFlash | Full raw context, AMR bypass |
| Heuristic recent/random/previous-attention | Selector không học |
| Learned selection-only | Slots off, dùng total budget tương đương |
| Compression-only | Dynamic distant selection off; giữ local + slots |
| Full AMR | Local + learned selected + learned slots |
| Attention imitation | Selector fit bằng attention labels thay acceptance preferences |
| No reuse / reuse 2 / reuse 4 | Thay tần suất rescore, giữ khả năng global retrieval |
| No gate / auto | Đo protection ở short context |
| Slot position ablation | Midpoint vs một convention đã khai báo, cùng checkpoint adaptation budget |
| Ordinary DFlash fine-tune | Cùng target, data và tổng adaptation GPU-giờ, gồm label generation |

Layer-specific selection chỉ thêm sau shared-support pilot. Report total entries mỗi layer và sum bytes; không gọi shared 4K tương đương 5 tập 4K disjoint về storage/gather cost.

**Matched-budget:** hybrid 4096 raw + 128 slots đối chiếu selection-only 4224 raw. Đồng thời giữ comparison 4096 raw để đo incremental gain của slots. V0 checkpoint fingerprint khóa toàn bộ memory config; muốn đổi budget phải capture/label/train một run riêng. Compression-only với cùng entry count có thể không khả thi/khác capacity; báo trade-off nhiều budget và matched measured cost, không ép equality giả.

## 6. P3 — Holdout và systems benchmark

Khóa checkpoint, gate, configs và decision criteria trên calibration rồi mới mở holdout. Lặp seeds training nếu compute cho phép; ít nhất ghi seed và run variability.

V0 claim chỉ batch 1/HF backend/hardware đã đo. Batch>1 cần task executor riêng, không chạy nhiều batch-1 process rồi gọi là batch-size ablation. Concurrency benchmark ghi rõ scheduler, process count và VRAM budget.

Model-transfer/32K là extension: chỉ báo nếu có checkpoint/context support và data thật; không ngoại suy 16K.

## 7. Timing không làm thay đổi fairness

- Dense và AMR cùng target cached verifier, backend/dtype, max output, tokenizer/prompt, hardware, warm-up policy.
- Instrumented reference và oracle có label timing riêng, không xuất hiện trong bảng deployable speedup.
- CUDA events cho device stages; wall clock đồng bộ ở sample boundary cho E2E. Không synchronize sau mỗi kernel trong production throughput run.
- Ghi CPU selection/scheduler time; CUDA overlap không cộng các stage event rồi gọi là wall clock.
- Warm-up 2 document ngoài cohort đo; measured paired order luân phiên dense→AMR và AMR→dense, 3 repeats/document ở pilot cost profile.
- Bootstrap bank/index/slots, first draft, refresh, gather, lazy switching, terminal processing đều tính vào sample time.
- Các stage phải có scope không chồng lặp; báo unattributed/overlap thay vì ép tổng event bằng E2E.

`T_e2e = T_prefill + T_decode + T_other_outside_decode`. Compression build ở prefill phải nằm trong TTFT/E2E; update giữa rounds nằm trong decode. TTFT đo tới lần emit token đầu, không mặc định bằng target prefill.

## 8. Metric contract

| Field | Định nghĩa |
|---|---|
| `accepted_proposals_raw` | Matched prefix trước EOS/cap clipping, 0..15 |
| `accepted_proposals_committed` | Accepted proposals thực sự emit |
| `committed_tokens` | Token mới emit, không đếm anchor lại |
| `survival_j` | Indicator A≥j, j=1..15; ghi censoring |
| `verification_calls` | Target block verification calls; phân biệt auxiliary target processing |
| `decode_committed_tok_s` | sum G của decode events / total decode seconds, báo initial/terminal policy |
| `decode_cost_per_token_ms` | Total decode ms / committed output tokens |
| `avg_accept_length` | Mean committed tokens theo convention baseline, kèm `accept_length_semantics` |
| `amr_mean_accepted_proposals` | Mean canonical A; không dùng alias mơ hồ |
| `target_exact_match` | Final token IDs bằng target full-context greedy |

Output length, first-token events, EOS và processed-cache cursor phải đối chiếu emitter trace. Báo cả per-document mean A/G và round-weighted mean, không trộn weighting giữa variants.

V0 generation record có các keys của `validate_schema(record, spec=True)`: method/dataset/model/token counts/batch/timing/throughput/QPS/peak memory và speculative fields; CLI fail nếu thiếu key. Record bổ sung raw/committed A totals, G totals/mean, `verification_calls` và `decode_committed_tok_s`; `retained_tokens` là mean raw+slot entries theo round, raw/slot counts nằm trong `state_trace`. `memory_build_latency_ms` (cũng map vào base field `selector_latency_ms`) gồm key-index/update, slot build và memory assembly, không phải MLP-only latency. Target exactness độc lập và resident-memory breakdown chưa là output field V0. Unknown timing dùng null, không zero giả.

Output ROUGE gọi helper chung. Summary ghi success/failed counts, total tokens/time, exactness, stage breakdown, draft/total memory và data/model fingerprints.

Hợp đồng timing hiện hành: `prefill_ms` đo target prefix forward với
`logits_to_keep=1`; verifier giữ toàn bộ logits của 16 vị trí. `ttft_ms` đo từ
đầu engine đến khi token đầu và initial projected bank sẵn sàng.
`decode_time_ms`/`decode_ms = e2e_ms - ttft_ms`; TPOT dùng decode time chia
`output_tokens - 1`, null nếu không có khoảng decode. Build latency gồm
bootstrap, selection/slots/mask và incremental key/slot/bank update;
projection có field riêng. CUDA stage timings dùng events, không thêm
synchronize riêng cho mỗi stage; E2E/TTFT đồng bộ tại ranh giới đo.

Rollout ghi sample/document/source ID, split, tokenized `prompt_hash`, input
manifest SHA-256, checkpoint SHA-256 và effective `run_config_hash`. Hash
run gồm mode, cost gate, budget, precision/backend và checkpoint;
`workload_hash` cho phép đối chiếu cùng workload giữa các mode. Pair theo
ID/prompt/workload và kiểm tra timing policy; không ghép chỉ theo số thứ tự
dòng. Summary throughput chuẩn dùng tổng committed decode tokens chia tổng
decode time; `mean_decode_committed_tok_s` chỉ là trung bình mô tả sample.

## 9. Thống kê và decision gate

Paired comparisons theo document; bootstrap 2000 resamples ở document level, giữ states/repeats của document trong cùng cluster. Báo 95% CI, p50/p95 và distribution first rejection. Hàng nghìn rounds không là hàng nghìn sample độc lập.

Gate pilot đề xuất, cần khóa trước holdout:

| Gate | Điều kiện |
|---|---|
| Correctness | Greedy final-token parity 100% trên measured cohort; không memory/cache invariant failure |
| Mechanism | Full AMR mean A tăng so dense trên long-context calibration; đo survival, không chỉ mean independent token accuracy |
| Systems pilot | Decode committed tok/s tăng ≥5% trên calibration 8K/16K, tính full overhead |
| Confirmatory result | Lower 95% CI của paired long-context throughput ratio >1; báo magnitude/E2E riêng |
| Short protection | Auto E2E không chậm quá 2% so dense trên short calibration; holdout CI và switching costs được báo |
| Complementarity | Hybrid hơn selection-only về Pareto acceptance/cost; nếu chưa rõ, không claim compression cần thiết |
| Adaptation economics | Báo capture+label+train+eval GPU-giờ, so ordinary fine-tune cùng ngân sách |

5%/2% là decision thresholds đề xuất, không là kết quả hoặc cam kết. Nếu CI rộng, kết luận chưa đủ bằng chứng; không đổi threshold sau xem holdout.

Scientific outcome có thể là selection-only thắng hybrid, heuristic đủ tốt hoặc no-go. Giữ negative results và quyết định trong `decision.json`; không giữ module chỉ để phù hợp tên AMR.

## 10. Compute và lệnh dự kiến

Trần pilot đề xuất **24 GPU-giờ** cho adaptation. Resource ledger V0 tính allocation-hours gần đúng theo elapsed wall time của capture, label và training phases trên `cuda`; rollout benchmark/inference không bị trừ khỏi cap này. Một run 4 GPU × 1 giờ dùng 4 GPU-giờ. Đo time/memory thực trước khi chia budget; không dự báo số giờ train từ trainable parameter count.

Các lệnh hiện dùng được sau khi master config trỏ tới model paths local, corpus đã khóa và Python 3.12 runtime sẵn sàng. Chạy từ repo root trên B200:

~~~bash
export AMR_RUN_ID=pilot_2026_10_08
export AMR_RUN_ROOT="$PWD/outputs/amr_dflash/$AMR_RUN_ID"
export AMR_DATA_MANIFEST=/path/to/locked_train_validation_holdout.jsonl

bash scripts/run.sh amr_dflash preflight --require-b200
bash scripts/run.sh amr_dflash capture --max-samples 60 --max-new-tokens 256 --max-states-per-document 6
bash scripts/run.sh amr_dflash candidates
bash scripts/run.sh amr_dflash label
bash scripts/run.sh amr_dflash train-selector --steps 200
export AMR_CHECKPOINT="$PWD/checkpoints/amr_dflash/$AMR_RUN_ID/selector.pt"
bash scripts/run.sh amr_dflash train-compressor --checkpoint-in "$AMR_CHECKPOINT" --steps 200
export AMR_CHECKPOINT="$PWD/checkpoints/amr_dflash/$AMR_RUN_ID/compressor.pt"
bash scripts/run.sh amr_dflash evaluate-fixed --split validation --mode dense --output "$AMR_RUN_ROOT/evaluation/fixed_dense.jsonl"
bash scripts/run.sh amr_dflash evaluate-fixed --split validation --mode selection --checkpoint "$AMR_CHECKPOINT" --output "$AMR_RUN_ROOT/evaluation/fixed_selection.jsonl"
bash scripts/run.sh amr_dflash evaluate-fixed --split validation --mode compressor --checkpoint "$AMR_CHECKPOINT" --output "$AMR_RUN_ROOT/evaluation/fixed_compressor.jsonl"
bash scripts/run.sh amr_dflash evaluate-fixed --split validation --mode amr --checkpoint "$AMR_CHECKPOINT" --output "$AMR_RUN_ROOT/evaluation/fixed_amr.jsonl"
bash scripts/run.sh amr_dflash infer --split holdout --mode dense --max-samples 30 --max-new-tokens 256 --output "$AMR_RUN_ROOT/evaluation/dense.jsonl"
bash scripts/run.sh amr_dflash infer --split holdout --mode selection --checkpoint "$AMR_CHECKPOINT" --max-samples 30 --max-new-tokens 256 --output "$AMR_RUN_ROOT/evaluation/selection.jsonl"
bash scripts/run.sh amr_dflash infer --split holdout --mode compressor --checkpoint "$AMR_CHECKPOINT" --max-samples 30 --max-new-tokens 256 --output "$AMR_RUN_ROOT/evaluation/compressor.jsonl"
bash scripts/run.sh amr_dflash infer --split holdout --mode amr --checkpoint "$AMR_CHECKPOINT" --max-samples 30 --max-new-tokens 256 --output "$AMR_RUN_ROOT/evaluation/amr.jsonl"
~~~

`infer --split` lọc toàn bộ manifest theo `split` trước khi áp dụng
`--max-samples`; giá trị mặc định `all` giữ hành vi tương thích. Record có
`split` rõ ràng được ưu tiên, còn record không có split dùng phép chia ổn định
theo document ID và fraction trong config. Giữ holdout chỉ cho lần đánh giá
confirmatory đã khóa; không dùng kết quả đó để chọn checkpoint hoặc ngưỡng.

`run.sh` load master shell-env theo `config/master.path`; `AMR_DATA_MANIFEST`, `AMR_RUN_ID`, model paths, device và Python được resolve từ đó/runtime. Checkpoint là file `.pt` có fingerprint strict, không phải directory `COMPLETE`. Selection/compressor/amr inference cần `--checkpoint`. Local chỉ dùng CPU synthetic; các lệnh trên chưa được chạy với model thật.

## 11. Gói bằng chứng để viết paper

`decision.json` liên kết run manifest, config/gate lock, checkpoints, exactness report, per-document paired results, CI script/version, stage profile, ablations và adaptation cost.

Chỉ sau đó mới điền claim C1 causal context usefulness, C2 selection/compression complementarity, C3 verification-driven learning và systems gains. Literature assertions trong proposal cần survey/citation verification riêng khi viết Related Work; protocol này không xác nhận novelty.
