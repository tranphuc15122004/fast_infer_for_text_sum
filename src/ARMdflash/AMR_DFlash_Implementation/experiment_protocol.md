# Giao thức thực nghiệm và quyết định AMR-DFlash

Ngày: **08/10/2026**. Đây là protocol đề xuất; không có số AMR thực nghiệm trong tài liệu.

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

Pilot capture tối đa 6 states/document (hai đầu/hai giữa/hai cuối, loại terminal-censored states cho preference). Trần 12 candidates/state và 8 pairs/state. Ghi số states/candidates **thực**, tie rate và no-headroom fraction.

Khóa `splits.json` trước label generation. Holdout không dùng chọn gate, slot count, LR hoặc early stopping.

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

Train selector từ actual preference labels; train compressor/adapters với frozen backbone. Chạy hệ thống đầy đủ local+selected+slots+gate.

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

**Matched-budget:** hybrid 4096 raw + 128 slots đối chiếu selection-only 4224 raw. Đồng thời giữ comparison 4096 raw để đo incremental gain của slots. Compression-only với cùng entry count có thể không khả thi/khác capacity; báo trade-off nhiều budget và matched measured cost, không ép equality giả.

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

Generation record mang toàn bộ keys của `validate_schema(record, spec=True)`: method/dataset/model/token counts/batch/timing/throughput/QPS/peak memory và speculative fields. `retained_tokens` là effective draft context entries; tách `raw_retained_tokens` và `compressed_slots`. Unknown timing dùng null + reason, không zero giả.

Output ROUGE gọi helper chung. Summary ghi success/failed counts, total tokens/time, exactness, stage breakdown, draft/total memory và data/model fingerprints.

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

Trần pilot đề xuất **24 GPU-giờ**, gồm capture, label generation, train và evaluation. Một run 4 GPU × 1 giờ dùng 4 GPU-giờ. Đo time/memory thực trước khi chia budget; không dự báo số giờ train từ trainable parameter count.

Các lệnh sau chỉ dùng **sau khi task tương ứng đã triển khai**. Production từ repo root với master đã load:

~~~bash
export FAST_INFER_PYTHON=python3
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/externals/dflash"

python3 scripts/amr_dflash/capture_states.py --config src/AMR_DFlash/configs/pilot.yaml
python3 scripts/amr_dflash/label_candidates.py --config src/AMR_DFlash/configs/pilot.yaml
python3 -m AMR_DFlash.run_train --config src/AMR_DFlash/configs/pilot.yaml --phase selector
python3 -m AMR_DFlash.run_train --config src/AMR_DFlash/configs/pilot.yaml --phase compressor
python3 scripts/amr_dflash/evaluate.py --config src/AMR_DFlash/configs/pilot.yaml --checkpoint "$AMR_CHECKPOINT" --split validation
bash scripts/run.sh amr_dflash
~~~

`AMR_DATA_MANIFEST`, `AMR_RUN_ID`, `AMR_CHECKPOINT`, target/draft paths, input/output và GPU assignment đến từ master shell-env. Checkpoint evaluation phải chỉ đến một directory COMPLETE cụ thể, không tự chọn “latest” mơ hồ. Không sửa `config/master.path` tự động. Local chỉ dùng CPU synthetic; không chạy các lệnh GPU như một bước kiểm chứng tài liệu.

## 11. Gói bằng chứng để viết paper

`decision.json` liên kết run manifest, config/gate lock, checkpoints, exactness report, per-document paired results, CI script/version, stage profile, ablations và adaptation cost.

Chỉ sau đó mới điền claim C1 causal context usefulness, C2 selection/compression complementarity, C3 verification-driven learning và systems gains. Literature assertions trong proposal cần survey/citation verification riêng khi viết Related Work; protocol này không xác nhận novelty.
