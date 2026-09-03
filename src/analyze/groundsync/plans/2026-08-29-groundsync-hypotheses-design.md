# GroundSync: thiết kế thí nghiệm kiểm chứng hypothesis

## Mục tiêu và phạm vi

Thực hiện một pipeline phân tích tái lập để kiểm chứng các hypothesis trong
proposal GroundSync, không giả định trước rằng phenomenon tồn tại. Toàn bộ
source code, raw artifact, biểu đồ và báo cáo của experiment nằm dưới
`src/analyze/groundsync/`.

Thiết kế không gọi attention là ground truth. Đại lượng chính được gọi là
`target-derived source-utilization proxy`; semantic grounding chỉ được dùng
như cách diễn giải sau khi có kiểm tra robustness.

## Model, dữ liệu và điều kiện cố định

- Target mặc định: `Qwen/Qwen3-1.7B`.
- Draft controlled mặc định: `Qwen/Qwen3-0.6B`, cùng tokenizer family.
- CLI cho phép thay target bằng `Qwen/Qwen3-4B` và thay draft bằng mọi snapshot
  tương thích đã mount trên server.
- `local_files_only=True`, greedy decoding, `do_sample=False`, tắt thinking,
  batch size 1, seed cố định.
- Dataset discovery: `govreport_representative.jsonl`,
  `multinews_representative.jsonl`, `xsum_representative.jsonl` và
  `cnn_dailymail_representative.jsonl`. GovReport là dataset chính; các dataset
  còn lại là kiểm tra regime/cross-dataset.
- Context lengths và output lengths được ghi theo token sau tokenizer; smoke
  dùng 1–2 mẫu/dataset, discovery dùng tối đa 100 mẫu/dataset, confirmatory
  dùng tập documents disjoint và không tune ngưỡng trên test.
- Attention analysis dùng `attn_implementation=eager` để lấy vector attention
  của query hiện tại. Không lưu ma trận `L x L`; chỉ lưu source-chunk vector
  sau mỗi output step.

## Đại lượng quan sát

Với source đã token hóa và chia thành các chunk 128 token (sensitivity 64/256),
ở output position `t` lưu:

- `g_raw[t, c]`: attention mass từ target query hiện tại vào source chunk `c`;
- `g_nosink[t, c]`: cùng đại lượng sau khi loại 4/8/16 source token đầu;
- `g_calibrated[t, c]`: raw mass đã chia cho positional prior học trên dev;
- target entropy, token id, output position, sentence-boundary proxy và hidden
  state summary nếu backend trả được;
- canonical target tokens/summary và metadata về model, tokenizer, commit,
  seed, context length, sample/document id.

Mọi `g` được normalize trên source chunks. `d_t` là Jensen–Shannon divergence
giữa `g_t` và `g_{t-1}`. Đây là continuous signal chính; threshold chỉ được
fit trên dev cho các phân tích segment/horizon.

## Hypothesis và phép kiểm chứng

### E0 — kiểm soát sink và positional bias

So sánh raw, no-sink và calibrated estimator. Tạo position-relocation fixture
khi có thể: cùng evidence span được đặt ở đầu/giữa/cuối. Báo cáo chênh lệch
mass, drift và độ ổn định của segment. Nếu tín hiệu biến mất khi bỏ sink hoặc
calibrate, raw-attention result không được dùng làm evidence độc lập.

### H1/E1 — source-state persistence

Claim cần test: adjacent output positions có source-utilization state giống nhau
hơn random positions sau controls.

Metrics:

- lag similarity `1 - JS(g_t, g_{t+l})` với `l = 1,2,4,8,16,32`;
- real trace so với shuffle/random-position null, cluster bootstrap theo
  document;
- meaningful segment length `L_G` từ threshold chọn trên dev;
- kết quả phải được báo riêng cho raw/no-sink/calibrated.

Engineering gate tham khảo: median `L_G >= 4` trên GovReport và ít nhất một
dataset khác, đồng thời adjacent similarity cao hơn null. Gate này không thay
thế confidence interval hay kiểm định chính thức.

### H2/E2 — transition và speculative rejection

Sinh canonical target trace trước. Với mỗi target prefix được chọn, draft
Qwen3-0.6B sinh tối đa `Kmax=8` token. First mismatch với canonical target
continuation chính là first rejection của greedy speculative verification; ghi
`accepted_len`, `first_reject_rel`, draft confidence và timing.

Fit hazard/survival summary theo relative draft position, với `d_{t+j}` là
predictor và bootstrap/cluster theo document. Claim ban đầu chỉ là
association/predictive hazard, không gọi là causal.

### H3/E3 — incremental signal ngoài token difficulty

So sánh hai predictor deterministic, split theo documents:

- baseline: target entropy, draft confidence, recent acceptance, output/position,
  sentence boundary và copyability;
- grounding: baseline cộng `g_t`, `d_t`, lag drift và estimator variant.

Báo cáo AUROC, AUPRC, log loss, Brier/calibration và hệ số drift trong model
đã kiểm soát. Negative controls: temporal shift `g[t+10/20/50]`, shuffle trong
request và position-only. Grounding chỉ được gọi incremental khi kết quả giữ
trên test document và không chỉ xuất hiện với raw estimator.

### H4/E4 — oracle horizon có computational utility

Định nghĩa oracle horizon là bước đầu tiên trong tương lai mà JS drift vượt
ngưỡng dev. So sánh fixed `k=2,4,8`, acceptance-history/entropy adaptive,
oracle grounding horizon và true-cost oracle.

Nếu có GPU, controlled runner đo draft/verification wall time cho từng block;
nếu chỉ có trace, báo riêng analytical cost proxy và không gọi đó là E2E speedup.
Metric chính là committed tokens/sec hoặc milliseconds/committed token, kèm
rejected verified tokens. Chỉ tiếp tục H5 nếu oracle tạo headroom rõ (mục tiêu
tham khảo >=8% so với best fixed/adaptive trên ít nhất hai regime).

### H5/E5 — online horizon predictability

Fit một predictor nhỏ, không dùng future `g`: logistic survival heads hoặc
regularized linear model trên `g_t`, drift history, entropy, draft confidence
và recent acceptance. Split theo documents; chọn threshold trên train/dev.

Báo cáo horizon calibration, NLL/Brier và utility của policy dự đoán. Metric
systems cuối là committed tokens/sec; metric phụ là phần oracle gain được phục
hồi. Nếu không có E2/E4 hoặc thiếu model/draft, H5 phải là `INCONCLUSIVE`.

## Pipeline và artifact

- `core.py`: chunking, normalize, JS divergence, lag/segment/horizon metrics,
  logistic/survival utilities và policy replay; không phụ thuộc model loading.
- `trace_target.py`: canonical target generation, incremental attention capture,
  entropy/hidden-state metadata và JSONL trace.
- `trace_speculative.py`: controlled target/draft greedy proposals, acceptance,
  confidence và optional measured cost.
- `run_experiment.py`: CLI orchestration, manifest, deterministic split và
  status/error records (`ok`, `unavailable`, `oom`, `error`).
- `report.py`: aggregate metrics, confidence/bootstrap summaries, plots và
  `hypothesis_report.md`.
- `tests/`: test thuần CPU cho core metrics, synthetic traces, schema và smoke
  CLI không tải model.
- `results/<run_id>/`: `manifest.json`, `target_traces.jsonl`,
  `speculative_traces.jsonl`, `metrics.json`, `metrics.csv`, `hypothesis_report.md`,
  PNG plots và `run.log`. Không ghi đè `src/analyze/full_infer/results`.

## Validation scope

Đây là analysis/evaluation pipeline, không phải training pipeline; không áp
dụng L1 training Validation Pyramid. Validation bắt buộc gồm:

1. `py_compile` và unit tests cho toàn bộ core;
2. synthetic CPU smoke chứng minh H1–H5 metric path chạy và giữ schema;
3. model-backed smoke khi snapshot local tồn tại;
4. server CUDA run với `local_files_only` cho timing/E2E nếu đủ model/draft;
5. report phải phân biệt `PASS`, `FAIL`, `INCONCLUSIVE`, `UNAVAILABLE` và nêu
   sample/document coverage, thay vì suy luận từ việc code chạy thành hypothesis
   đúng.

Thiếu Qwen snapshot hoặc CUDA ở local chỉ làm giới hạn phạm vi run; không được
điền số liệu giả. Các lỗi OOM, output rỗng, non-finite metric, checkpoint/model
không đọc được và stalled run phải được ghi trong manifest và report.

## Review criteria

```yaml
review_criteria:
  metrics:
    - name: H1 adjacent-vs-null persistence
      direction: ">"
      threshold: 0
    - name: H2 drift hazard coefficient
      direction: ">"
      threshold: 0
    - name: H3 grounding incremental test gain
      direction: ">"
      threshold: 0
    - name: H4 oracle utility
      direction: ">="
      threshold: 0.08
    - name: H5 oracle-gain recovery
      direction: ">="
      threshold: 0.50
  performance:
    - report measured E2E only on CUDA; report analytical proxy separately
  observability:
    - phase start/end, progress, run id, model paths and per-document counts
  stability:
    - no silent failures; non-finite values are explicit failures
  custom:
- no raw attention claim without sink/position controls
    - split and bootstrap units are documents, not individual tokens
```

## Execution audit 2026-08-30

- Runtime thực tế dùng Qwen3-4B target và Qwen3-0.6B draft trên Tesla T4 bằng
  `/home/tuantb/miniconda3/bin/python3` ngoài venv; cả hai model được load
  local-only. Qwen3-1.7B không cần dùng vì không có snapshot canonical phù hợp.
- GovReport có 99/100 target hợp lệ, 99 controlled proposals và 10 timing rows
  phủ đủ `k=8`; CNN/DailyMail có 100/100 target, 100 proposals và 12 timing
  rows. H2 được mở rộng thành discrete hazard với drift coefficient điều chỉnh
  relative position và 2.000 bootstrap resamples theo document.
- E0 position-relocation đã chạy bằng fixture Qwen3-4B: cùng evidence span ở
  đầu/giữa/cuối source, chunk 16. No-sink giảm position effect nhưng không xóa
  nó; artifact và diễn giải nằm trong báo cáo master.
- H3/H5 đã được chạy thêm draft-only multi-start ở `1,6,11,16` trên 99/100
  documents mỗi regime, nên lag-drift history và within-document acceptance
  history có dữ liệu. Run này không có verifier timing; main timing vẫn chỉ có
  một start/document. Báo cáo phải giữ giới hạn đó, không nâng discovery thành
  confirmatory hoặc production serving.

## Decision rule

Report conclusion per hypothesis. `FAIL` means the specified evidence contradicts
the claim; `INCONCLUSIVE` means coverage or model/runtime was insufficient;
`PASS` requires fresh metrics, controls and the stated coverage. GroundSync
implementation is only interpreted as supported when H1–H4 pass and H5 is
predictable; otherwise the report recommends the exact stopping point.

## Runtime revision 2026-08-29

- Thực nghiệm GPU dùng `/home/tuantb/miniconda3/bin/python3` ngoài venv,
  `cuda:0`, FP16 trên Tesla T4 15 GiB (driver 550.163.01, CUDA 12.4).
- T4 là `sm75`, không dùng được native Flash SDP của torch cu124 cho context
  dài. Target, draft và verifier dùng chunked causal prefill với
  `prefill_chunk_size=512`; attention của query cuối vẫn được lấy bằng eager.
- Discovery all trên 25 GovReport, 16 target tokens, 4 speculative tokens và
  2 prefix/document tạo 25 target rows + 50 controlled rows có timing. Đây là
  discovery evidence; confirmatory 100 mẫu/cross-regime vẫn là bước mở.

## Impact on Plan

- Subtask 2: bổ sung chunked prefill/mask bottom-right để long-context trace
  chạy được trên T4 `sm75` mà không đổi metric.
- Subtask 3: áp dụng cùng prefill path cho draft và target verifier; thêm CLI
  verifier standalone để tái sử dụng target trace.
- Subtask 5: model-backed validation đã chạy trên T4; report phải giữ các
  `FAIL`/`UNAVAILABLE` của discovery, không diễn giải thành kết luận tổng quát.

## P0 decision experiments — 2026-09-02

Theo quyết định nghiên cứu mới, giai đoạn tiếp theo không triển khai thêm
GroundSync/BurstSpec mà kiểm tra trực tiếp headroom và cơ chế admission.
Artifact của giai đoạn này vẫn nằm trong `src/analyze/groundsync/`.

### P0-1 — corrected within-block transition H2

Với proposal bắt đầu tại `t`, tạo risk-set row cho từng relative position `j`
(`1..Kmax`) nếu proposal còn sống tại vị trí đó. Predictor chính là
`d_transition[t,j] = JS(g[t+j-1], g[t+j])`; controls gồm target entropy tại
`t+j`, draft confidence tại `j`, relative position và output position. Fit
regularized logistic discrete-hazard model và document-bootstrap CI. Không dùng
`drift_at_start` cho quyết định chính. Gate: lower 95% CI của hệ số transition
phải lớn hơn 0 và cùng dấu trên GovReport/CNN-DM; nếu không, transition H2
fail.

### P0-2 — corrected Grounding Oracle H4

Horizon không thấy transition trong `Kmax` bước được mã hóa là `Kmax`, không
phải `1`. Threshold được chọn trên train/dev documents bằng utility khi có
timing, sau đó freeze trước test. So sánh best fixed, entropy/history adaptive
và corrected Grounding Oracle. Oracle chỉ được gọi utility oracle khi có timing
đo riêng; acceptance-only được ghi riêng.

### P0-3/P0-4 — oracle ladder và admission oracle

Ladder sử dụng `k={0,2,4,8,16}`. `k=0` là AR một token với chi phí đo
`autoregressive_time_ms`; `k>0` dùng draft + cached target verification. True-cost
hindsight oracle được tính trên toàn ladder. First-token admission oracle chỉ
dùng bit `accepted_len > 0`, không dùng accepted length đầy đủ; báo cáo recovery
so với best fixed và true-cost oracle.

### P0-5 — burstiness

Within-block hazard báo cáo `h_j=P(R=j|R>=j)` và tỷ số `h_1/mean(h[j>1])`.
Across-round persistence dùng `S_t=1[accepted_len>0]`, sắp theo document và
start position, đo `P(S[t+delta]=1|S[t]=1)` so với marginal tại
`delta={1,2,4,8}`. Multi-start acceptance traces được dùng để tránh suy luận từ
một start/document.

### Coverage và quyết định

Chạy `max_k=16` trên tối đa 100 documents/dataset cho acceptance; chạy timing
trên tối đa 50 documents/dataset nếu runtime chịu được. Mọi run ghi manifest,
status từng document, model/runtime, seed và command. Corrected GroundOracle
beat best generic adaptive baseline tối thiểu 5–8% là ngưỡng tiếp tục GroundSync;
entry oracle recovery trên 40–50% gap là tín hiệu pivot BurstSpec. Nếu entry
oracle yếu, dừng adaptive speculation direction. Đây là discovery/controlled
evidence, chưa phải production E2E.

## P1/P2 execution extension — 2026-09-02

Sau P0, bổ sung ba kiểm tra có điều kiện nhưng vẫn chạy đầy đủ để tránh dừng
ở ceiling oracle: predictor admission rẻ, strong drafter replication và direct
E2E. Predictor chỉ dùng feature tại thời điểm proposal (`target_entropy_at_start`,
confidence token đầu, acceptance history quá khứ và output position), split theo
document, chọn `k`/threshold trước test, rồi đánh giá bằng measured
committed-tokens/ms. Strong drafter dùng cache EAGLE-3 Qwen3-4B ghép canonical
Qwen3-4B; acceptance list của EAGLE được chuẩn hóa bằng cách trừ target fallback
token. P2 đo paired greedy output/timing trên cùng prompt và chỉ gọi là direct
E2E, không gọi là vLLM/API serving.

Multi-News được thêm như regime confirmatory: 50 target traces, 50 controlled
acceptance rows, 10 timing rows cho ladder và 9 starts/document cho persistence.
Thiếu timing hoặc thiếu class được ghi `UNAVAILABLE`/`INCONCLUSIVE`; không nội
suy từ CNN/GovReport.

## Impact on Plan

- Subtask 1–9: giữ nguyên kết quả và artifact; không thay đổi gate P0.
- New Subtask 10: cheap admission predictor trên timing rows, document-disjoint
  train/dev/test, metric classification và realized utility.
- New Subtask 11: EAGLE-3 strong-drafter replication và persistence analysis trên
  GovReport/CNN-DM (thêm Multi-News nếu runtime cho phép).
- New Subtask 12: paired direct E2E benchmark; vLLM/API chỉ được ghi
  `UNAVAILABLE` nếu package/server không có, không dùng proxy thay thế.
- Validation scope: thêm smoke EAGLE, exact-match guardrail và fresh report audit;
  L0/L1 training Validation Pyramid không áp dụng vì đây là evaluation pipeline.

## Target-KV Conditioned Block Drafting — approved revision 2026-09-03

Phần này là revision tiếp theo của kế hoạch, theo proposal Target-KV
Conditioned Block Drafting. Không thay thế kết quả GroundSync/P0/P1/P2 trước đó;
đây là nhánh thí nghiệm mới dùng cùng thư mục artifact.

### Phạm vi và cặp model

E0/E1 dùng cặp local-only chính thức:

- target: Qwen3-4B snapshot trong cache Hugging Face;
- drafter E0: `z-lab/Qwen3-4B-DFlash-b16` snapshot trong cache;
- dtype: FP16 trên Tesla T4 (không dùng BF16 native);
- batch size: 1; inference ngoài `.venv` bằng Python system/Miniconda có CUDA.

Không dùng chain Qwen3-0.6B cho E0 vì đó là weak-drafter artifact khác với
hypothesis Target-KV. Không tải model qua mạng; thiếu cache được ghi
`UNAVAILABLE`.

### E0 — baseline failure map

Chạy K=`{4,8,16}`, `max_new_tokens=32`, greedy/temperature 0. DFlash loader
nhận `block_size` độc lập với native `block_size=16`, nên ba K được chạy trực
tiếp trên cùng checkpoint. Mỗi row lưu prompt/document id, input length, K,
acceptance length từng vòng, first rejection, prefill/decode/draft/verify
timing, peak VRAM và status.

Tập chính là `data/longbench_200/gov_report.jsonl` và
`data/longbench_200/multi_news.jsonl`; CNN/DailyMail chỉ là short/easy control.
Main result không padding nhân tạo. Mẫu vượt `max_position_embeddings=40960`
được loại có lý do; mẫu Multi-News dài bất thường cũng không được cắt im lặng.
Bucket được báo theo dữ liệu tự nhiên, kèm `n` và CI; bucket có ít mẫu không
được dùng cho claim tổng quát.

Metrics chính:

```text
S_L(j) = P(A >= j | context bucket L)
MAT = mean(A)
first-rejection distribution
draft/verify latency
committed tokens/s
```

Kill gate E0: cần thấy survival tại K=8/16 giảm theo context một cách có CI
hoặc interaction context × depth có ý nghĩa. Nếu không, dừng động lực
"long-context KV advantage" và không mở rộng E2/E3.

### E1 — representation sufficiency probe

Target frozen. Tạo feature cache theo anchor/document-disjoint split, không cho
future target tokens lọt vào representation. So sánh cùng một lightweight block
predictor và cùng ngân sách trainable parameters:

- R1: target hidden hiện tại `h_t`;
- R2: token-wise hidden sequence `{h_i}`;
- R3: multi-layer hidden sequence;
- R4: target KV `{K_i,V_i}` ở số layer tương ứng R3.

Các representation được đưa qua interface dimension chung; số tham số adapter
và predictor được ghi vào manifest, nếu không equal được thì dùng frozen
projection chung và báo cáo parameter audit. Probe dự đoán 16 token target,
đánh giá CE, Acc@1, Acc@5 và `P(A>=j)` theo `j=1..16`, context bucket và
dataset. Có negative controls shuffled-KV, wrong-document-KV, recent-only-KV và
source-only-KV.

Kill gate E1: KV phải hơn token-wise hidden (không chỉ hơn `h_t`) ở ít nhất hai
regime hoặc có CI rõ ràng, và hiệu ứng phải tăng theo `j` hoặc context. Nếu
R4 chỉ hơn R1 nhưng xấp xỉ R2, kết luận là memory sequence đủ, không ủng hộ
claim KV-specific.

### Validation, quy mô và chuyển pha

Trước model run: unit/schema tests, document split/leakage test, feature shape,
finite-value và parameter-budget audit. Sau đó chạy GPU smoke một mẫu ngắn và
một mẫu 8K trước pilot. Pilot mục tiêu E0 là tối đa 200 GovReport + 200
Multi-News + 100 CNN/DailyMail; E1 bắt đầu 5.000 anchors/dataset, chỉ tăng khi
smoke và coverage đạt. Mọi kết quả dùng document bootstrap CI và ghi rõ bucket
thưa.

Chỉ mở E2 factorial sau khi cả E0 và E1 pass gate. E3 DFlash-KV adapter chỉ
được chạy adapter-only ở context mà T4 xác nhận không OOM; full fine-tuning và
claim 32–40K cần GPU lớn hơn.
