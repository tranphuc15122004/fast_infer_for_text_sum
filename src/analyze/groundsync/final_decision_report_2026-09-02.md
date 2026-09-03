# Báo cáo tổng hợp kiểm định GroundSync/BurstSpec — 2026-09-02

## 1. Kết luận điều hành

Đã hoàn tất các thực nghiệm quyết định P0, P1 và phần P2 có thể chạy trên
host hiện tại. Kết luận chính xác hiện tại là:

> **Không có bằng chứng đủ để tiếp tục GroundSync như một hướng tổng quát.**
> Grounding Oracle thất bại về utility trên cả ba regime. Oracle ladder chứng
> minh có ceiling opportunity, nhưng admission/predictor không nhất quán và
> strong-drafter replication không xác nhận burstiness. Direct E2E có speedup,
> nhưng không đạt lossless exact-match; serving API/vLLM không chạy được trên
> host hiện tại.

| Nhánh | Kết luận | Căn cứ quyết định |
|---|---|---|
| GroundSync transition | **MIXED / không đủ để giữ claim** | corrected H2: GovReport PASS, CNN/DailyMail và Multi-News FAIL |
| GroundSync utility | **NO-GO** | corrected H4 FAIL cả 3 regime; gain so best available lần lượt -5.24%, -33.40%, -45.91% |
| Oracle opportunity | **Có ceiling, chưa có policy** | O3 headroom +72.3%, +23.5%, +43.7% |
| BurstSpec admission | **MIXED, không đủ pivot tổng quát** | P0-4 tốt nhất 73.8%, 30.3%, 77.7%; P1 predictor Gov FAIL, CNN PASS, Multi-News INCONCLUSIVE |
| Strong drafter | **FAIL** | EAGLE-3 Qwen3-4B h1/later < 1 trên cả ba regime; persistence CI không dương nhất quán |
| P2 direct E2E | **Đo được speedup nhưng FAIL guardrail** | speedup 1.819–1.905×, exact-match chỉ 94–98% |
| P2 serving API | **UNAVAILABLE** | canonical server không mount; runtime GPU không có vLLM |

Vì vậy, quyết định cuối là:

```text
NO_GO_GROUNDSYNC_GENERAL
NO_GO_BURSTSPEC_GENERAL
ORACLE_HEADROOM_EXISTS_BUT_ONLINE_POLICY_NOT_VALIDATED
```

## 2. Phạm vi và nguyên tắc kiểm định

Mục tiêu là kiểm tra đúng các câu hỏi quyết định sau P0: (P0-1) transition
attention bên trong block có liên quan rejection hay không; (P0-2) corrected
Grounding Oracle có utility hay không; (P0-3) ceiling của ladder
`k={0,2,4,8,16}`; (P0-4) chỉ cần bit first-token admission có đủ để chọn AR/SPEC
không; (P0-5) acceptance có burstiness trong block và persistence giữa các
round không; (P1) cheap predictor có tái tạo admission oracle không; (P1)
strong drafter có tái tạo hiện tượng không; và (P2) speedup E2E có giữ output
hay không.

Mọi artifact của các thực nghiệm mới được đặt dưới
`src/analyze/groundsync/`. Acceptance-only, controlled timing, direct E2E và
serving preflight được tách riêng; không dùng timing proxy để gọi là serving
throughput.

## 3. Môi trường và model

### GPU run

- Host: `tuantb@teslaT4`, Tesla T4, 15,360 MiB, compute capability 7.5.
- Driver: 550.163.01; CUDA runtime: 12.4.
- GPU runs dùng `/home/tuantb/miniconda3/bin/python3` ngoài `.venv`, theo yêu
  cầu không chạy venv cho thực nghiệm T4.
- GPU runtime: Python 3.13.9, PyTorch `2.6.0+cu124`, Transformers 4.57.6,
  `torch.cuda.is_available()=True` khi chạy ngoài sandbox.
- `.venv` Python 3.12 là runtime dev CPU; không dùng cho số liệu T4.

### Cache

- Target canonical Qwen3-4B:
  `/home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`.
- Weak controlled drafter Qwen3-0.6B:
  `/home/tuantb/models/Qwen3-0.6B`.
- Strong drafter:
  `/home/tuantb/models/Qwen3-4B_eagle3`, checkpoint local
  `AngelSlim/Qwen3-4B_eagle3`, ghép với target Qwen3-4B qua vendored
  `externals/EAGLE`.
- Head EAGLE có architecture `Eagle3LlamaForCausalLM`, không được dùng làm
  canonical target; chỉ dùng như EAGLE head sau khi smoke loader và generation
  thành công.

## 4. Cách thực hiện và validation

### P0-1 — corrected within-block transition H2

Với proposal start `t`, risk set được mở rộng theo relative position `j`:

```text
d_transition[t,j] = JS(g[t+j-1], g[t+j])
```

Event là first rejection tại `j`; các vị trí sau rejection bị loại khỏi risk
set. Mô hình discrete hazard là ridge-logistic, có controls target entropy,
draft confidence, relative position và absolute output position. Hệ số được
chuẩn hóa theo độ lệch chuẩn và CI 95% bootstrap theo document với 2,000
resamples. Gate là lower CI > 0 và phải tái lập giữa regime.

### P0-2 — corrected Grounding Oracle H4

Không thấy transition trong `Kmax` bước được mã hóa thành `Kmax`, không phải
`1`. Threshold được chọn trên train/dev theo document bằng measured utility,
sau đó freeze trên test. So sánh best fixed, entropy/history adaptive và
corrected Grounding Oracle trên common timing population.

### P0-3/P0-4 — oracle ladder và admission

`k=0` là AR cached one-token với `autoregressive_time_ms`; `k>0` là draft
incremental cộng cached target verification. Ladder chỉ dùng row có đủ timing
cho toàn bộ `0,2,4,8,16`. True-cost oracle là hindsight ceiling. Admission
oracle chỉ được nhìn `accepted_len > 0`, không được nhìn accepted length đầy đủ.
Recovery là phần gap giữa best fixed và true-cost oracle được admission oracle
thu hồi.

### P0-5 — burstiness/persistence

Within-block đo `h_j=P(R=j|R>=j)` và `h1/mean(h[j>1])`. Across-round dùng
`S_t=1[accepted_len>0]`, nhiều start/document và `delta={1,2,4,8}`; CI
bootstrap theo document. P0 chính dùng 9 start/document
`1,4,7,10,13,16,19,22,25`.

### P1 — cheap predictor

`p1_predictor.py` fit standardized ridge logistic regression trên:

```text
target_entropy_at_start
draft_confidence_first
recent_acceptance
log1p(start_position)
```

Không dùng grounding, future attention, accepted length hoặc first rejection.
Fit trên train documents, threshold/policy chọn trước test, báo AUROC/AUPRC,
log-loss/Brier/ECE và realized tokens/ms. Test dưới 10 documents được ghi
`INCONCLUSIVE`.

### P1 strong drafter và P2 direct E2E

`p1_strong_drafter.py` dùng EAGLE-3 Qwen3-4B head. Acceptance length EAGLE có
thêm một target fallback token, vì vậy accepted draft tokens được tính là
`max(acceptance_length-1, 0)`. Direct E2E chạy EAGLE và greedy AR trên cùng
prompt sau warmup; prefill loại khỏi decode timing. Exact-match là guardrail
lossless, không được bỏ qua chỉ vì speedup dương.

## 5. Kết quả P0 trên ba regime

| Dataset | Coverage target/spec | Timing complete | Multi-start | H2 | H4 | O3 headroom | Best P0-4 recovery | P0-5 |
|---|---:|---:|---:|---|---|---:|---:|---|
| GovReport | 99/99 | 55 | 891 / 99 docs | PASS | FAIL | +72.3% | 73.8% (`k=8`) | PASS |
| CNN/DailyMail | 100/100 | 50 | 900 / 100 docs | FAIL | FAIL | +23.5% | 30.3% (`k=4`) | PASS |
| Multi-News | 50/50 | 10 | 450 / 50 docs | FAIL | FAIL | +43.7% | 77.7% (`k=4`) | FAIL |

### Chi tiết H2

- GovReport: drift coefficient `0.24378`, CI `[0.01101, 0.59689]`.
- CNN/DailyMail: `0.09078`, CI `[-0.29372, 0.36339]`.
- Multi-News: `0.25900`, CI `[-0.23395, 0.78485]`.

Chỉ GovReport có CI dương. Do đó không thể gọi transition H2 là hiện tượng
tổng quát.

### Chi tiết H4 và ladder

Gain corrected Grounding Oracle so với best available test:

- GovReport: `-5.24%`.
- CNN/DailyMail: `-33.40%`.
- Multi-News: `-45.91%`.

Oracle ladder O3 vẫn có headroom lớn vì hindsight được phép chọn `k=0` và
biết accepted outcome. Đây là ceiling, không phải policy online. Kết quả này
chỉ nói rằng admission/selection opportunity tồn tại trong trace; nó không
chứng minh tín hiệu grounding có thể thu hồi opportunity.

### Chi tiết P0-5

- GovReport: within ratio `1.5623`; persistence CI dương tại `delta=1`.
- CNN/DailyMail: within ratio `1.0452`; persistence CI dương tại `delta=4`.
- Multi-News: within ratio `1.1503`, nhưng mọi persistence CI đều cắt 0.

Vì gate yêu cầu cả within asymmetry và persistence evidence, Multi-News là
`FAIL` dù tỷ số within lớn hơn 1.

## 6. Kết quả P1 cheap predictor

| Dataset | Train/dev/test docs | Test AUROC/AUPRC | Chọn k | Predictor tok/ms | Fixed tok/ms | Entry oracle tok/ms | Recovery | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GovReport | 33/11/11 | 0.611 / 0.611 | 2 | 0.002735 | 0.002735 | 0.003108 | 0.0% | FAIL |
| CNN/DailyMail | 30/10/10 | 1.000 / 1.000 | 4 | 0.012542 | 0.011570 | 0.012935 | 71.2% | PASS |
| Multi-News | 6/2/2 | 1.000 / 1.000* | 4 | 0.008006 | 0.008006 | 0.011395 | 0.0% | INCONCLUSIVE |

`*` Multi-News chỉ có 2 test documents nên AUROC không được dùng như bằng
chứng confirmatory. Cross-regime P1 là `MIXED`; một predictor nhỏ chưa đủ để
biến oracle opportunity thành policy tổng quát.

## 7. Strong-drafter replication và direct E2E

| Dataset | Docs / rounds | Admission rate | h1/later | Persistence gate | EAGLE speedup | Exact-match | Strong decision | P2 direct |
|---|---:|---:|---:|---|---:|---:|---|---|
| GovReport | 50 / 666 | 73.7% | 0.455 | FAIL | 1.819× | 96% | FAIL | FAIL |
| CNN/DailyMail | 50 / 660 | 72.0% | 0.527 | FAIL | 1.905× | 98% | FAIL | FAIL |
| Multi-News | 50 / 668 | 71.7% | 0.513 | FAIL | 1.814× | 94% | FAIL | FAIL |

Trong cả ba regime, h1 thấp hơn hazard continuation trung bình; không thấy
CI persistence nào vượt dương theo gate. Vì vậy kết quả strong drafter không
ủng hộ cách giải thích rằng burstiness trước đó chỉ bị che bởi weak drafter;
ngược lại, hiện tượng không tái lập với EAGLE head này.

Direct E2E có speedup dương nhưng exact-match không đạt 100%. Do đó chỉ được
ghi là speed measurement tham khảo, không được gọi là lossless serving
improvement.

## 8. P2 serving preflight

Preflight được ghi tại
`results/p2-serving-preflight-20260902/p2_serving_preflight.json` và bản
miniconda tương ứng. Canonical server repo
`/workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum` không
được mount trên host hiện tại. GPU miniconda runtime không import được vLLM;
`.venv` import được vLLM nhưng CUDA không khả dụng và theo AGENTS chỉ dùng cho
CPU dev. Vì vậy P2 server/API là `UNAVAILABLE`, không được thay bằng direct
EAGLE benchmark.

## 9. Artifact và lệnh tái lập

Các file nguồn chính:

- [`p0_decision.py`](p0_decision.py), config và report trong
  `results/p0-decision-final9-20260902/`.
- Multi-News P0: config
  [`p0_decision_config_multinews_20260902.json`](p0_decision_config_multinews_20260902.json)
  và `results/p0-decision-multinews-20260902/`.
- [`p1_predictor.py`](p1_predictor.py), test và
  `results/p1-cheap-admission-20260902/`.
- [`p1_strong_drafter.py`](p1_strong_drafter.py), test và các run
  `results/p1p2-eagle3-{gov50,cnn50,multinews50}-20260902/`.
- [`p2_serving_preflight.py`](p2_serving_preflight.py) và preflight artifact.
- Báo cáo P0 chi tiết: [`p0_final_report_2026-09-02.md`](p0_final_report_2026-09-02.md).
- Báo cáo lịch sử H1–H5/E0–E5: [`verification_report_2026-08-29.md`](verification_report_2026-08-29.md).
- Manifest toàn bộ artifact: [`final_artifact_manifest_2026-09-02.json`](final_artifact_manifest_2026-09-02.json).

Lệnh phân tích lại P0/P1:

```bash
python3 -m src.analyze.groundsync.p0_decision \
  --config src/analyze/groundsync/p0_decision_config_20260902.json \
  --output src/analyze/groundsync/results/p0-decision-final9-20260902 \
  --max-k 16 --candidate-ks 0,2,4,8,16 --bootstrap-samples 2000

python3 -m src.analyze.groundsync.p1_predictor \
  --gov-timing src/analyze/groundsync/results/p0-gov50-k16-timing-cached-20260902/speculative_traces.jsonl \
  --gov-timing src/analyze/groundsync/results/p0-gov10-k16-timing-cached-cont-20260902/speculative_traces.jsonl \
  --cnn-timing src/analyze/groundsync/results/p0-cnn50-k16-timing-cached-20260902/speculative_traces.jsonl \
  --multinews-timing src/analyze/groundsync/results/p1p0-multinews10-timing-20260902/speculative_traces.jsonl \
  --output-dir src/analyze/groundsync/results/p1-cheap-admission-20260902
```

## 10. Validation cuối

Validation cần thiết cho analysis pipeline:

```bash
python3 -m pytest -q src/analyze/groundsync/tests
python3 -m compileall -q src/analyze/groundsync
git diff --check
```

Unit tests bao phủ core, target/spec acceptance, corrected P0 metrics, P1
predictor, EAGLE acceptance conversion và serving preflight. Kết quả fresh
cuối: **66 passed**, compileall pass, `git diff --check` pass và artifact audit
pass. Synthetic tests chỉ kiểm tra đường code, không thay cho model-backed
evidence.

## 11. Hạn chế còn lại

1. P0 timing là controlled direct target/draft timing trên T4, không phải
   server/API production.
2. Multi-News timing mới 10 documents và predictor test chỉ 2 documents; đây
   là confirmatory discovery, không đủ để khẳng định performance tổng quát.
3. EAGLE head Qwen3 được dùng vì đã cache và loader chạy được, nhưng chính repo
   EAGLE vẫn ghi official Qwen3 chưa được hỗ trợ đầy đủ; kết quả strong
   drafter là replication của checkpoint cụ thể, không phải mọi EAGLE-3 head.
4. Exact-match direct E2E thấp hơn 100% cần được xử lý nếu sau này muốn chứng
   minh lossless serving; hiện tại không được che khuất bằng speedup.
5. Chưa chạy XSum, 128K context hay hàng nghìn samples vì các quyết định hiện
   tại đã dừng hướng tổng quát; đó là giới hạn đã biết, không phải số liệu bị
   bỏ qua.

---

# Phụ lục hồ sơ đầy đủ

Phần này ghi lại toàn bộ đường đi từ hypothesis ban đầu tới quyết định cuối.
Mục tiêu là để có thể audit báo cáo mà không phải suy ra semantics từ tên
thư mục. Mọi số liệu trong phụ lục đều lấy từ các JSON, manifest và raw trace
đã sinh trong cùng thư mục src/analyze/groundsync/.

## A. Phạm vi bằng chứng và cách đọc trạng thái

Có ba tầng bằng chứng:

1. Discovery lịch sử H1-H5/E0-E5 chạy trước ngày 2026-09-02. Pha này kiểm tra
   source-state, attention sink/position confounder, hazard, predictor nhỏ và
   timing subset.
2. P0 decision experiments là bộ kiểm định hiện tại đã sửa formulation H2/H4
   và thêm k=0, k=16, admission oracle, multi-start và Multi-News.
3. P1/P2 follow-up gồm predictor admission causal, strong-drafter EAGLE,
   direct E2E speed reference và serving preflight.

Ý nghĩa nhãn:

| Nhãn | Ý nghĩa |
|---|---|
| PASS | Metric vượt gate đã khóa và coverage phù hợp với gate. |
| FAIL | Run hợp lệ, nhưng metric không ủng hộ hypothesis hoặc không vượt baseline. |
| MIXED | Có kết quả dương ở một regime nhưng không tái lập ở regime khác. |
| INCONCLUSIVE | Đường code chạy nhưng test/class/coverage quá nhỏ để quyết định. |
| UNAVAILABLE | Thiếu runtime, model, timing hoặc mount; không được diễn giải là fail khoa học. |

Vì vậy, NO_GO_GROUNDSYNC_GENERAL không có nghĩa mọi attention signal đều vô
giá trị. Nó có nghĩa chưa có bằng chứng rằng signal đó tạo ra utility online,
ổn định qua dataset và tốt hơn baseline đơn giản.

## B. Hypothesis, biến đo và gate

### B.1 P0-1 corrected H2: transition drift

Với proposal bắt đầu ở t, g_i là vector source-attention của target tại bước
i. Drift ở vị trí tương đối j được định nghĩa là:

    d_transition[t,j] = JS(g[t+j-1], g[t+j])

R_t=j là event first rejection ở vị trí tương đối j. Proposal ở trong risk set
tại j nếu chưa có rejection trước đó; các vị trí sau rejection bị loại.
Discrete hazard được fit bằng ridge-logistic:

    logit P(R_t=j | R_t>=j)
      = beta0 + beta1*j + betaG*d_transition[t,j] + controls

Controls thực tế: target entropy, draft confidence, relative position và
absolute output position. Drift được z-score trên risk set; betaG là effect cho
một standard deviation drift. CI dùng 2.000 document-bootstrap resamples.

Gate: lower 95% CI của betaG phải lớn hơn 0. Đây là association sau controls,
không phải bằng chứng causal.

### B.2 P0-2 corrected H4: Grounding Oracle

Horizon H_t là transition đầu tiên trong cửa sổ quan sát. Không thấy transition
trong Kmax nghĩa là H_t > Kmax, do đó policy dùng k=Kmax; không chuyển NULL về
k=1.

Threshold được tune trên train/dev theo document bằng utility đo được, sau đó
freeze trên test. Candidate sweep gồm threshold cũ 0.2 và các giá trị 0.005,
0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 0.4, 0.5.

    U = total committed tokens / total measured cost_ms

Best available baseline là policy nhanh nhất trong fixed positive và generic
adaptive entropy/history. Gate H4 là corrected Grounding Oracle phải hơn best
available test baseline ít nhất 5%.

### B.3 P0-3: Oracle Ladder

Ba ladder dùng cùng common timing population:

    O1 = {2, 4, 8}
    O2 = {0, 2, 4, 8}
    O3 = {0, 2, 4, 8, 16}

Với k=0, target chạy một AR step dùng autoregressive_time_ms và commit đúng một
token. Với k>0, committed token là min(k, accepted_len + 1), cost là draft
incremental decode cộng cached target verification. True-cost oracle được phép
nhìn accepted outcome và cost của candidate trong cùng row, nên chỉ là
hindsight ceiling.

Gate ladder: headroom của mỗi oracle so với best fixed positive phải ít nhất
8%. Gate này chỉ trả lời còn opportunity trong trace hay không; không phải
chứng nhận một online policy.

### B.4 P0-4: First-token Admission Oracle

Chỉ dùng một bit hindsight:

    S_t = 1[accepted_len > 0]

Policy là S_t=0 -> k=0 và S_t=1 -> k=candidate_k. Không dùng accepted length,
first rejection hay future attention. Recovery:

    Recovery = (U_entry_oracle - U_best_fixed)
               / (U_true_cost_oracle - U_best_fixed)

Gate: recovery ít nhất 40%.

### B.5 P0-5: burstiness và persistence

Within-block hazard:

    h_j = P(R=j | R>=j)

Asymmetry là h1 / mean(h[j>1]). Across-round dùng S_t và đo:

    P(S[t+delta]=1 | S[t]=1) - P(S[t+delta]=1)

cho delta={1,2,4,8}. CI 95% bootstrap theo document, 2.000 resamples. Gate
P0-5 yêu cầu ratio > 1 và ít nhất một delta có lower CI > 0.

## C. Kết quả discovery H1-H5/E0-E5 trước P0

Các kết quả sau được giữ nguyên từ verification_report_2026-08-29.md để truy
vết; chúng không thay thế kết quả corrected P0.

| Hypothesis/experiment | Kết quả lịch sử | Ý nghĩa |
|---|---|---|
| E0 position relocation và attention sink | Diagnostic: position confounder còn rõ sau no-sink | Raw attention không phải ground-truth attribution. |
| H1 source-state persistence | FAIL composite | No-sink GovReport có persistence, nhưng calibrated dưới gate và CNN không đủ robust. |
| H2 drift -> rejection | GovReport FAIL, CNN PASS trong formulation cũ | Chiều không nhất quán giữa regime. |
| H3 incremental grounding predictor | FAIL primary | Grounding không tăng AUROC đủ sau controls; calibrated CNN chỉ là sensitivity. |
| H4 old oracle/timing | FAIL | Grounding oracle chậm hơn fixed khoảng 44.3% GovReport và 43.9% CNN. |
| H5 horizon predictor | UNAVAILABLE | Có classifier metric nhưng oracle không có headroom hợp lệ. |
| Multi-start cũ | GovReport H2 FAIL, CNN H2 PASS; H4/H5 thiếu timing | Chỉ dùng cho acceptance/history sensitivity. |

Số nổi bật của pha cũ:

- H1 GovReport no-sink persistence excess khoảng 0.023892, lower CI 0.021996;
  calibrated lower CI chỉ 0.018257, nên composite fail.
- H1 CNN/DailyMail no-sink lower CI khoảng 0.010580, thấp hơn gate.
- H2 cũ có beta khoảng -0.065733 ở GovReport; CNN có effect dương nhỏ trong
  formulation khác, cho thấy không thể claim cross-regime.
- H3 cũ có GovReport grounding AUROC gain 0; CNN no-sink gain khoảng -0.0267.
- H4 cũ dùng timing subset nhỏ với fixed k=2,4,8; true-cost oracle còn
  headroom nominal nhưng Grounding Oracle vẫn chậm hơn fixed.

Các điểm yếu này dẫn trực tiếp tới P0: đổi H2 sang drift crossing transition
theo j, sửa NULL horizon, bắt buộc common timing population và thêm k=0 để
kiểm tra admission.

## D. Nhật ký thực hiện, lỗi và sửa lỗi

### D.1 Cache và runtime

Đã kiểm tra snapshot trước khi chạy. Target Qwen3-4B, draft Qwen3-0.6B và
Qwen3-4B EAGLE head đều có local cache. Các run model-backed cuối dùng
local_files_only; không tải mạng.

GPU model runs dùng executable ngoài venv:

    /home/tuantb/miniconda3/bin/python3

Đây là yêu cầu quan trọng của thực nghiệm T4. Analyzer và unit test có thể dùng
.venv vì chỉ đọc JSONL/tính metric. Preflight cho thấy .venv có vLLM nhưng
CUDA false và torch cu130 không phù hợp host T4.

### D.2 T4 OOM và chunked prefill

Eager/math attention và một run SDPA đầu tiên OOM với prompt GovReport dài.
Nguyên nhân là fallback tạo ma trận attention L x L trên T4 15 GiB. Target,
draft và verifier được sửa để dùng chunked causal prefill với
prefill_chunk_size=512. Một long-document smoke sau đó chạy được; các batch
chính chỉ được chạy sau smoke này.

### D.3 Coverage và censoring

Discovery ban đầu dùng 25 GovReport. Sau đó chạy mở rộng 100 GovReport và 100
CNN/DailyMail cho target/acceptance; Multi-News bổ sung 50 target/acceptance.
Một GovReport target/proposal lỗi OOM được giữ trong manifest và loại khỏi
phân tích tương ứng.

Proposal EOS sớm không bị coi là đã sống tới Kmax; analyzer censor phần chưa
quan sát. Một GovReport timing row status ok nhưng thiếu timing tới k=16 cũng
bị loại khỏi common ladder. Vì vậy timing Gov có 56 row raw nhưng chỉ 55 row
complete.

### D.4 Timing và P1 common filter

Semantics timing cuối:

- k=0 dùng AR one-token cached timing;
- k>0 dùng draft incremental time cộng target cached verification;
- prompt prefill không cộng lại vào cost từng round khi cache đã có.

P1 ban đầu có nguy cơ so các candidate k trên subset khác nhau. Đã sửa để lọc
row phải có đủ timing của mọi k={0,2,4,8,16} trước khi fit/replay. Các bảng
P0/P1 hiện tại dùng artifact sau sửa.

### D.5 EAGLE runner

Smoke EAGLE phát hiện benchmark generic xử lý sai reference dạng list và
runner mới unpack sai số lượng giá trị generation. Runner EAGLE riêng đã được
sửa, smoke chạy thành công rồi mới chạy 50 mẫu/dataset. Không chỉnh output để
cải thiện exact-match.

## E. Code và artifact

| File/thư mục | Vai trò |
|---|---|
| trace_target.py | Target greedy canonical trace, source attention và manifest. |
| trace_speculative.py | Draft proposal, accepted prefix và timing theo k. |
| p0_decision.py | H2, H4, ladder, admission, burstiness và bootstrap. |
| p1_predictor.py | Causal cheap admission predictor, split và policy utility. |
| p1_strong_drafter.py | EAGLE load/generation, acceptance normalization, persistence và direct E2E. |
| p2_serving_preflight.py | vLLM/CUDA/canonical mount checks. |
| tests/test_p0_decision.py | Test semantics k=0, NULL, timing, hazard, oracle. |
| tests/test_p1_predictor.py | Test feature causal, split, metric và policy. |
| tests/test_p1_strong_drafter.py | Test EAGLE acceptance/E2E aggregation. |
| tests/test_p2_serving_preflight.py | Test trạng thái runtime unavailable. |
| p0_decision_config_20260902.json | Mapping trace GovReport/CNN. |
| p0_decision_config_multinews_20260902.json | Mapping trace Multi-News. |
| final_artifact_manifest_20260902.json | Manifest artifact và trạng thái cuối. |

Raw P0:

- Gov target: results/qwen3-4b-gov100-gpu-protocol-20260830/
- Gov acceptance: results/p0-gov100-k16-20260902/
- Gov timing: results/p0-gov50-k16-timing-cached-20260902/ và
  results/p0-gov10-k16-timing-cached-cont-20260902/
- Gov multi-start: results/p0-gov100-multistart9-k4-20260902/
- CNN target: results/qwen3-4b-cnn100-gpu-protocol-20260830/
- CNN acceptance: results/p0-cnn100-k16-20260902/
- CNN timing: results/p0-cnn50-k16-timing-cached-20260902/
- CNN multi-start: results/p0-cnn100-multistart9-k4-20260902/
- Multi-News target: results/p1p0-multinews50-target-20260902/
- Multi-News acceptance: results/p1p0-multinews50-spec-20260902/
- Multi-News timing: results/p1p0-multinews10-timing-20260902/
- Multi-News multi-start: results/p1p0-multinews50-multistart-20260902/

Analyzer/follow-up:

- results/p0-decision-final9-20260902/
- results/p0-decision-multinews-20260902/
- results/p1-cheap-admission-20260902/
- results/p1p2-eagle3-gov50-20260902/
- results/p1p2-eagle3-cnn50-20260902/
- results/p1p2-eagle3-multinews50-20260902/
- results/p2-serving-preflight-20260902/

## E.1. Command sinh raw trace model-backed

Các command dưới đây là template tái lập với path canonical của host; output
thực tế đã chạy được lưu trong manifest của từng run. Phải giữ executable GPU
ngoài .venv:

    env CUDA_VISIBLE_DEVICES=0 /home/tuantb/miniconda3/bin/python3 \
      -m src.analyze.groundsync.trace_target \
      --model /home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
      --input data/representative_100/govreport_representative.jsonl \
      --output src/analyze/groundsync/results/qwen3-4b-gov100-gpu-protocol-20260830/target_traces.jsonl \
      --max-samples 100 --max-new-tokens 32 --chunk-size 128 \
      --skip-source-tokens 8 --prefill-chunk-size 512 \
      --device cuda:0 --dtype float16 --seed 42

Để chạy CNN/DailyMail hoặc Multi-News, thay input/output và số mẫu tương ứng.
Controlled acceptance/timing dùng:

    env CUDA_VISIBLE_DEVICES=0 /home/tuantb/miniconda3/bin/python3 \
      -m src.analyze.groundsync.trace_speculative \
      --draft-model /home/tuantb/models/Qwen3-0.6B \
      --verification-model /home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
      --input data/representative_100/govreport_representative.jsonl \
      --target-traces src/analyze/groundsync/results/qwen3-4b-gov100-gpu-protocol-20260830/target_traces.jsonl \
      --output src/analyze/groundsync/results/p0-gov50-k16-timing-cached-20260902/speculative_traces.jsonl \
      --max-samples 50 --max-k 16 --max-starts 1 \
      --start-offset 1 --stride 1 --prefill-chunk-size 512 \
      --device cuda:0 --dtype float16 --seed 42

Acceptance-only trace bỏ verification-model và dùng output
p0-gov100-k16-20260902; multi-start dùng max-k 4, max-starts 9, start-offset
1, stride 3. Các trường này được ghi trong speculative_traces.manifest.json,
vì vậy không cần đoán lại từ tên run.

## F. Protocol model-backed chung

### F.1 Input và prompt

Input representative:

    data/representative_100/govreport_representative.jsonl
    data/representative_100/cnn_dailymail_representative.jsonl
    data/representative_100/multinews_representative.jsonl

Target Qwen3-4B và draft Qwen3-0.6B render cùng prompt, greedy, seed 42, batch
size 1. Target decision trace tối đa 32 output tokens. P0 multi-start dùng
start 1,4,7,10,13,16,19,22,25.

### F.2 Target và acceptance trace

Target prefill theo chunk 512, sau đó decode greedy. Source attention lưu dưới
dạng vector đã gom chunk cùng metadata.

Tại mỗi prefix, draft sinh block k token. Target verifier đối chiếu từng draft
token với canonical target next-token. accepted_len là số draft token đúng liên
tiếp trước first rejection. Speculative round commit accepted prefix cộng
target fallback, giới hạn bởi k.

Acceptance-only trace có timing_basis draft_only_no_target_check. Timing trace
có draft_time_by_k_ms, verification_time_by_k_ms và autoregressive_time_ms.
Hai loại trace không trộn lẫn.

### F.3 Split, common population, bootstrap

Split train/dev/test theo document, không split row/token. Threshold và fixed k
chọn trên train/dev; test chỉ dùng báo cáo cuối. Mọi ladder dùng row complete
cho toàn bộ k cần so sánh.

Bootstrap lấy document làm đơn vị rồi lấy toàn bộ row của document được chọn.
Điều này làm CI rộng hơn pooled-token bootstrap nhưng tránh coi các token cùng
document là mẫu độc lập.

## G. Coverage cuối

| Dataset | Target | Acceptance Kmax=16 | Timing raw/complete | Multi-start raw/complete |
|---|---:|---:|---:|---:|
| GovReport | 99/100 | 99 | 56/55 | 891/891 |
| CNN/DailyMail | 100/100 | 100 | 50/50 | 900/897 |
| Multi-News | 50/50 | 50 | 10/10 | 450/448 |

P0 test split sau common timing: GovReport 11 documents, CNN/DailyMail 10,
Multi-News 2. P1 predictor dùng Gov 33/11/11, CNN 30/10/10, Multi 6/2/2.
Strong-drafter dùng 50 documents mỗi dataset.

## H. Kết quả P0 chi tiết

### H.1 P0-1 corrected H2

| Dataset | Documents | Risk rows | Event rows | betaG/SD | Odds ratio/SD | CI 95% document bootstrap | Decision |
|---|---:|---:|---:|---:|---:|---|---|
| GovReport | 99 | 302 | 96 | 0.243778 | 1.276061 | [0.011007, 0.596886] | PASS |
| CNN/DailyMail | 100 | 539 | 97 | 0.090784 | 1.095032 | [-0.293717, 0.363390] | FAIL |
| Multi-News | 50 | 208 | 50 | 0.259002 | 1.295636 | [-0.233945, 0.784846] | FAIL |

GovReport đúng chiều và CI dương. CNN/DailyMail và Multi-News có point estimate
dương nhưng CI bao phủ 0. H2 tổng quát là MIXED, không phải PASS.

### H.2 P0-2 corrected Grounding Oracle

| Dataset | Train/dev/test | Threshold | Oracle test tok/ms | Best fixed test | Best adaptive test | Gain vs best available | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| GovReport | 33/11/11 | 0.010 | 0.002591 | k2: 0.002735 | entropy: 0.001289 | -5.24% | FAIL |
| CNN/DailyMail | 30/10/10 | 0.005 | 0.007706 | k4: 0.011570 | entropy: 0.006007 | -33.40% | FAIL |
| Multi-News | 6/2/2 | 0.005 | 0.004331 | k4: 0.008006 | test subset rất nhỏ | -45.91% | FAIL |

Grounding Oracle full common-population utility lần lượt là 0.003330,
0.008970 và 0.004985 tok/ms; chỉ test utility dùng cho gate. Oracle cao hơn
adaptive entropy khoảng 101.1% ở GovReport và 28.3% ở CNN/DailyMail, nhưng best
fixed vẫn nhanh hơn. Thắng adaptive yếu không đủ giữ GroundSync.

Threshold calibration metadata: Gov 44 calibration rows và 12 candidate test
rows trước common-complete filter; final H4 dùng 11 test rows. CNN dùng 40/10;
Multi-News dùng 8/2.

### H.3 P0-3 Oracle Ladder

Tokens/ms trên cùng common timing population:

| Dataset | k=0 | k=2 | k=4 | k=8 | k=16 | True-cost oracle | O1 | O2 | O3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GovReport, n=55 | 0.002990 | 0.003169 | 0.003015 | 0.002471 | 0.001817 | 0.005460 | +46.8% | +64.1% | +72.3% |
| CNN/DailyMail, n=50 | 0.006780 | 0.010472 | 0.011621 | 0.009645 | 0.005709 | 0.014357 | +20.4% | +22.3% | +23.5% |
| Multi-News, n=10 | 0.006156 | 0.007111 | 0.007546 | 0.005642 | 0.003269 | 0.010843 | +30.5% | +43.7% | +43.7% |

O1/O2/O3 đều PASS gate headroom 8% ở cả ba dataset. O2 tăng mạnh so O1 ở
GovReport và Multi-News, cho thấy admission/AR là nguồn ceiling quan trọng.
O3 chỉ tăng thêm so O2 ở GovReport và CNN; Multi-News không tăng.

Một số mean committed/cost:

| Dataset | Policy | Mean committed | Mean cost ms | Selected k |
|---|---|---:|---:|---|
| GovReport | fixed k2 | 1.2364 | 390.15 | 2:55 |
| GovReport | fixed k8 | 2.0727 | 838.97 | 8:55 |
| GovReport | O3 | 2.2182 | 406.26 | 0:35, 2:8, 4:6, 8:5, 16:1 |
| CNN/DailyMail | fixed k4 | 3.2600 | 280.53 | 4:50 |
| CNN/DailyMail | fixed k16 | 5.1400 | 900.30 | 16:50 |
| CNN/DailyMail | O3 | 4.3800 | 305.08 | 0:7, 2:10, 4:19, 8:13, 16:1 |
| Multi-News | fixed k4 | 2.4000 | 318.05 | 4:10 |
| Multi-News | O3 | 2.8000 | 258.24 | 0:5, 4:4, 8:1 |

### H.4 P0-4 First-token Admission Oracle

| Dataset | k | Admission rate | Utility tok/ms | Recovery | Decision |
|---|---:|---:|---:|---:|---|
| GovReport | 2 | 23.6% | 0.003584 | 18.1% | FAIL |
| GovReport | 4 | 23.6% | 0.004330 | 53.8% | PASS |
| GovReport | 8 | 23.6% | 0.004676 | 73.8% | PASS |
| GovReport | 16 | 23.6% | 0.004412 | 71.2% | PASS |
| CNN/DailyMail | 2 | 86.0% | 0.010719 | 6.3% | FAIL |
| CNN/DailyMail | 4 | 86.0% | 0.012449 | 30.3% | FAIL |
| CNN/DailyMail | 8 | 86.0% | 0.010694 | 22.3% | FAIL |
| CNN/DailyMail | 16 | 86.0% | 0.006469 | 8.8% | FAIL |
| Multi-News | 2 | 50.0% | 0.008115 | 26.9% | FAIL |
| Multi-News | 4 | 50.0% | 0.010109 | 77.7% | PASS |
| Multi-News | 8 | 50.0% | 0.008762 | 60.0% | PASS |
| Multi-News | 16 | 50.0% | 0.005649 | 31.4% | FAIL |

Best candidate: GovReport k=8 với 13/55 admitted; CNN k=4 nhưng chỉ 30.3%
recovery; Multi-News k=4 với 5/10 admitted. Cross-regime result là MIXED:
admission signal có giá trị ở GovReport/Multi-News subset nhưng không tổng quát.

### H.5 P0-5 Burstiness và persistence

Within-block:

| Dataset | Count | h1 | Mean later hazard | h1/later | Decision within |
|---|---:|---:|---:|---:|---|
| GovReport | 891 | 0.306397 | 0.196121 | 1.562287 | PASS |
| CNN/DailyMail | 900 | 0.236667 | 0.226441 | 1.045158 | PASS |
| Multi-News | 450 | 0.268889 | 0.233763 | 1.150264 | PASS riêng within |

Across-round excess, CI 95%:

| Dataset | delta=1 | delta=2 | delta=4 | delta=8 | Decision |
|---|---|---|---|---|---|
| GovReport | 0.0354 [0.0118,0.0583] | 0.0084 [-0.0125,0.0297] | 0.0052 [-0.0235,0.0359] | 0.0469 [-0.0830,0.1771] | PASS, delta=1 |
| CNN/DailyMail | 0.0045 [-0.0133,0.0225] | 0.0075 [-0.0123,0.0271] | 0.0240 [0.0016,0.0481] | -0.0035 [-0.0342,0.0291] | PASS, delta=4 |
| Multi-News | 0.0022 [-0.0298,0.0326] | 0.0167 [-0.0103,0.0437] | 0.0033 [-0.0316,0.0381] | -0.0021 [-0.0730,0.0750] | FAIL |

Multi-News có within ratio >1 nhưng mọi persistence CI cắt 0; theo gate phải
ghi FAIL. P0-5 chung vì vậy chỉ MIXED/FAIL cross-regime.

## I. P1 cheap online admission predictor

### I.1 Phương pháp

Model là standardized ridge-logistic, L2=1.0. Feature causal thực sự có trong
trace:

    target_entropy_at_start
    draft_confidence_first
    recent_acceptance
    log1p(start_position)

Không dùng grounding, future attention, accepted_len hoặc first_reject_rel.
Candidate k là 2,4,8,16; fixed k chọn train/dev, admission threshold chọn dev,
sau đó freeze test. Gate: ít nhất 10 test docs, utility predictor > fixed và
recovery >=50%.

### I.2 Kết quả

| Dataset | Train/dev/test | Test pos/neg | AUROC/AUPRC | Log-loss | Brier/ECE | k | Predictor | Fixed | Entry oracle | Recovery | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GovReport | 33/11/11 | 2/9 | 0.611/0.611 | 0.4400 | 0.1330/0.1350 | 2 | 0.002735 | 0.002735 | 0.003108 | 0.0% | FAIL |
| CNN/DailyMail | 30/10/10 | 8/2 | 1.000/1.000 | 0.2273 | 0.0758/0.1193 | 4 | 0.012542 | 0.011570 | 0.012935 | 71.2% | PASS |
| Multi-News | 6/2/2 | 1/1 | 1.000/1.000* | 1.0411 | 0.3766/0.4753 | 4 | 0.008006 | 0.008006 | 0.011395 | 0.0% | INCONCLUSIVE |

Multi-News AUROC/AUPRC chỉ dựa trên 2 test docs, nên không phải evidence
confirmatory. Gov predictor chọn k=2 cho cả 11 test rows. CNN chọn AR 3 rows
và k=4 7 rows. Multi không thu hồi entry gap. P1 cross-regime là MIXED.

## J. Strong-drafter replication và direct E2E

### J.1 Thiết kế

Dùng base Qwen3-4B local và head Qwen3-4B EAGLE-3 tại
/home/tuantb/models/Qwen3-4B_eagle3, loader từ externals/EAGLE. Mỗi dataset
chạy 50 samples, max_input_tokens=1024, max_new_tokens=32, total_token=16,
depth=4, top_k=4, batch 1, greedy.

EAGLE generation có thêm target fallback token; accepted draft count được
chuẩn hóa là max(acceptance_length - 1, 0). Direct E2E so cùng prompt EAGLE với
naive AR. Timing chỉ decode, prefill bị loại bởi EaModel.

### J.2 Burstiness và persistence

| Dataset | Docs | Rounds | Admission | h1 | Later mean | h1/later | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| GovReport | 50 | 666 | 73.72% | 0.262763 | 0.577511 | 0.454992 | FAIL |
| CNN/DailyMail | 50 | 660 | 71.97% | 0.280303 | 0.531492 | 0.527389 | FAIL |
| Multi-News | 50 | 668 | 71.71% | 0.282934 | 0.551882 | 0.512671 | FAIL |

Persistence excess CI đều có lower <=0. Ví dụ Gov delta 1 là -0.00206
[-0.02172,0.01879], delta 2 là 0.01271 [-0.00705,0.03298], delta 4 là
0.01551 [-0.00757,0.03902], delta 8 là 0.00168 [-0.02589,0.03028].
CNN tương ứng delta 1 0.00432 [-0.01549,0.02397], delta 2 0.00915
[-0.01199,0.02832], delta 4 -0.01489 [-0.03817,0.00689], delta 8 -0.01923
[-0.04946,0.01045]. Multi-News delta 1 0.00984 [-0.01449,0.03574], delta 2
0.02076 [-0.00362,0.04423], delta 4 -0.00023 [-0.02553,0.02608], delta 8
0.01247 [-0.03215,0.05405].

Trong cả ba regime h1/later <1, trái với burst hypothesis. Đây là fail của
checkpoint/setup EAGLE đã cache, không phải kết luận mọi EAGLE head đều fail.

### J.3 Direct E2E speed reference

| Dataset | EAGLE tok/s | Naive tok/s | Speedup | Exact-match | P2 direct |
|---|---:|---:|---:|---:|---|
| GovReport | 25.7721 | 14.1721 | 1.8185x | 96% | FAIL |
| CNN/DailyMail | 24.9189 | 13.0775 | 1.9055x | 98% | FAIL |
| Multi-News | 25.8387 | 14.2467 | 1.8137x | 94% | FAIL |

Gate P2 direct yêu cầu timing paired complete, speedup >1 và exact-match=100%.
Speedup dương không được gọi là lossless hoặc production serving khi exact
match chỉ 94-98%.

## K. P2 serving preflight

| Runtime | Python | Torch | CUDA | vLLM | Canonical mount | Status |
|---|---|---|---|---|---|---|
| .venv | 3.12.13 | 2.11.0+cu130 | false | true | false | UNAVAILABLE |
| GPU miniconda | 3.13.9 | 2.6.0+cu124 | true | false | false | UNAVAILABLE |

Canonical server path được kiểm tra:

    /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum

GPU preflight artifact ghi:

    status = UNAVAILABLE
    reason = vllm is not importable in the active system runtime;
             canonical server repository is not mounted on this host

Do đó chưa có P2 server/API benchmark. Đây là blocker môi trường, không phải
model fail; direct EAGLE benchmark không được dùng để thay thế serving.

## L. Lệnh tái lập

### L.1 P0 analyzer

    python3 -m src.analyze.groundsync.p0_decision \
      --config src/analyze/groundsync/p0_decision_config_20260902.json \
      --output src/analyze/groundsync/results/p0-decision-final9-20260902 \
      --max-k 16 --candidate-ks 0,2,4,8,16 --bootstrap-samples 2000

Multi-News dùng p0_decision_config_multinews_20260902.json và output
p0-decision-multinews-20260902.

### L.2 P1 analyzer

    python3 -m src.analyze.groundsync.p1_predictor \
      --gov-timing src/analyze/groundsync/results/p0-gov50-k16-timing-cached-20260902/speculative_traces.jsonl \
      --gov-timing src/analyze/groundsync/results/p0-gov10-k16-timing-cached-cont-20260902/speculative_traces.jsonl \
      --cnn-timing src/analyze/groundsync/results/p0-cnn50-k16-timing-cached-20260902/speculative_traces.jsonl \
      --multinews-timing src/analyze/groundsync/results/p1p0-multinews10-timing-20260902/speculative_traces.jsonl \
      --output-dir src/analyze/groundsync/results/p1-cheap-admission-20260902

### L.3 EAGLE direct

    env CUDA_VISIBLE_DEVICES=0 /home/tuantb/miniconda3/bin/python3 \
      -m src.analyze.groundsync.p1_strong_drafter \
      --base-model /home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
      --eagle-model /home/tuantb/models/Qwen3-4B_eagle3 \
      --dataset data/representative_100/govreport_representative.jsonl \
      --output-dir src/analyze/groundsync/results/p1p2-eagle3-gov50-20260902 \
      --max-samples 50 --max-new-tokens 32 --max-input-tokens 1024 \
      --total-token 16 --depth 4 --top-k 4 --include-naive

CNN/DailyMail và Multi-News thay dataset/output-dir; manifest của từng run
ghi command/runtime đầy đủ.

### L.4 Serving preflight và validation

    python3 -m src.analyze.groundsync.p2_serving_preflight \
      --output src/analyze/groundsync/results/p2-serving-preflight-20260902/p2_serving_preflight.json

    python3 -m pytest -q src/analyze/groundsync/tests
    python3 -m compileall -q src/analyze/groundsync
    git diff --check

## M. Validation và audit

Validation fresh sau khi hoàn thiện code/report:

    66 passed, 1 warning
    compileall: PASS
    git diff --check: PASS
    artifact audit: PASS

Cảnh báo duy nhất đến từ local .venv CUDA/driver mismatch khi test preflight;
đây là trạng thái đã biết, không ảnh hưởng các model run T4 bằng miniconda.

Logic được test gồm k=0 commit=1, NULL horizon, common timing population,
first rejection/risk set, censoring, document-bootstrap, causal feature
exclusion, document split, EAGLE fallback-token conversion và unavailable
serving state. Synthetic tests chỉ kiểm tra đường code; kết luận dùng
model-backed raw trace/metrics.

## N. Ma trận quyết định cuối

| Câu hỏi | Kết quả | Quyết định |
|---|---|---|
| Grounding transition có tái lập? | Gov PASS; CNN/Multi FAIL | MIXED, không giữ claim general |
| Grounding horizon có utility? | -5.24%, -33.40%, -45.91% vs best available | NO-GO GroundSync |
| Có oracle headroom? | O3 +72.3%, +23.5%, +43.7% | Có ceiling opportunity |
| Admission bit đủ mạnh? | 73.8%, 30.3%, 77.7% recovery | MIXED, follow-up có điều kiện |
| Burstiness/persistence ổn định? | Gov/CNN P0 pass, Multi fail | Không đủ general |
| Cheap predictor tái tạo oracle? | Gov fail, CNN pass, Multi inconclusive | MIXED, chưa deploy |
| Strong drafter tái lập? | Cả ba fail | Không xác nhận BurstSpec general |
| Direct E2E lossless? | Speedup dương nhưng exact 94-98% | P2 direct fail |
| Production serving? | GPU thiếu vLLM và canonical mount | UNAVAILABLE |

## O. Quyết định hành động

### O.1 GroundSync

Không tiếp tục train horizon estimator, attention calibration hay controller
GroundSync như hướng tổng quát. H2 chỉ dương ở GovReport; quyết định mạnh
nhất là corrected Grounding Oracle thua best fixed trên cả ba dataset. H2
GovReport được giữ như finding regime-specific, không làm cơ sở kiến trúc.

### O.2 BurstSpec

Không bật go cho BurstSpec general/production. Có thể giữ follow-up có điều
kiện cho GovReport-like hoặc Multi-News nếu mục tiêu là nghiên cứu admission:
O2/O3 cho thấy ceiling lớn và admission oracle có recovery cao ở một số regime.
Tuy nhiên CNN/DailyMail không đạt gate, predictor chỉ pass một dataset,
strong-drafter replication fail và direct E2E chưa lossless.

Nếu tiếp tục, cần mở rộng test document Multi-News, kiểm tra một strong drafter
aligned/official khác và chỉ chạy serving sau khi predictor pass cross-regime.
Không dùng kết quả hiện tại để tuyên bố production speedup, lossless generation,
causal grounding hoặc tính phổ quát của EAGLE/BurstSpec.

### O.3 Trả lời trạng thái hoàn thiện

- Đã hoàn thiện phần thực nghiệm quyết định có thể thực hiện trên host hiện tại:
  P0, P1 predictor, strong-drafter replication và P2 direct E2E.
- Đã ghi đầy đủ trong một file master: phương pháp, coverage, runtime/cache,
  raw artifact, debug correction, kết quả, gate, validation và quyết định từng
  thực nghiệm.
- P2 serving production chưa chạy được vì canonical server repository không
  mount và vLLM không import được trong GPU runtime; trạng thái này được ghi
  trung thực là UNAVAILABLE.

## P. Target-KV follow-up — 2026-09-03

Các thực nghiệm tiếp theo theo proposal Target-KV Conditioned Block Drafting
được triển khai sau ngày chốt của report này. Toàn bộ phương pháp, quá trình
xây dựng, raw artifact, kết quả E0/E1, giới hạn T4 và quyết định không mở E2/E3
được trình bày đầy đủ trong các mục P.1–P.8 bên dưới để file này vẫn là master
report duy nhất. Bản [`target_kv_decision_report_2026-09-03.md`](target_kv_decision_report_2026-09-03.md)
là technical companion giữ riêng phần phân tích Target-KV.

Tóm tắt quyết định mới: E0 tái lập zero-admission ở các run FP16 đã chạy nhưng
long-context context-drop là `INCONCLUSIVE` vì thiếu natural bucket dài; E1
không cho thấy KV vượt token-wise hidden sequence ở GovReport/Multi-News.

### P.1. Câu hỏi nghiên cứu và thiết kế Target-KV

Sau các kết quả GroundSync/BurstSpec trước đó, câu hỏi được thu hẹp thành hai
thí nghiệm quyết định:

1. **E0 — feasibility:** với target Qwen3-4B và DFlash drafter tương ứng, khi
   chạy các block (K\in\{4,8,16\}), có tồn tại accepted draft token ở các
   context đã quan sát hay không? Nếu acceptance bằng 0 ngay ở context ngắn và
   vừa, không thể xây controller dựa trên target KV trên cặp model đó.
2. **E1 — representation value:** target KV có giúp dự đoán suffix tốt hơn
   hidden state/hidden sequence trong một probe được kiểm soát hay không? Đây là
   kiểm định cho claim KV có thông tin bổ sung, không phải benchmark latency.

E0 và E1 được tách rời để tránh hai lỗi diễn giải: dùng một probe offline để
   bù cho việc hệ thống sinh thật không có acceptance, hoặc dùng acceptance
   trace để suy ra trực tiếp chất lượng representation.

Các quyết định được khóa trước khi chạy:

| Hạng mục | Protocol đã khóa | Mục đích |
|---|---|---|
| Model | Qwen3-4B target + Qwen3-4B-DFlash-b16 local cache | Đúng cặp snapshot có sẵn |
| Block | (K=4,8,16) | Bao phủ block ngắn, trung bình và native block 16 |
| Acceptance | (A=\max(0,\text{raw acceptance length}-1)) | Loại fallback token của API DFlash |
| Kết luận acceptance | max-new 32 | Kiểm tra không bị giới hạn bởi output budget 1 |
| Context | Không padding/truncation im lặng | Giữ phân phối và độ dài tự nhiên |
| Dataset | GovReport, Multi-News; CNN/DM control | Long/multi-document và short-context control |
| Bootstrap | Theo document | Không coi các anchor/round trong cùng doc là độc lập |
| E1 fairness | Cùng anchors, horizon, optimizer, decoder size | Cô lập tác động của representation |
| E1 gate | KV phải thắng hidden sequence ở ít nhất 2 regime hoặc CI dương | Tránh kết luận từ một run/seed |

### P.2. Quá trình xây dựng và luồng thực thi

Quá trình xây dựng gồm các bước tuần tự:

1. Kiểm tra model cache, tokenizer, config target/DFlash, runtime GPU và schema
   dữ liệu.
2. Viết E0 runner chỉ ghi raw observations: token ids, acceptance, timing,
   memory, trạng thái lỗi và exact-match; không để runner tự chọn kết luận.
3. Viết analyzer riêng để tính survival, MAT, bucket và bootstrap; nhờ đó có
   thể tái phân tích cùng raw data mà không chạy model lại.
4. Viết E1 extraction tách khỏi probe; mỗi representation được lưu shard và
   chuyển tensor trung gian về CPU để phù hợp VRAM T4.
5. Dùng một probe decoder chung cho mọi variant; giữ cùng trainable parameter
   count, interface, horizon và optimization budget.
6. Chạy theo thứ tự pilot → context feasibility → confirmation max-new32 → E1;
   chỉ chạy bước sau khi bước trước không còn lỗi protocol/accounting.
7. Chạy test, compile, diff và artifact audit trước khi tổng hợp quyết định.

Luồng E0:

```text
JSONL -> filter/tokenize -> target AR prefill/continuation
      -> DFlash proposal K=4,8,16 -> target verification
      -> acceptance accounting + exact ids + timing/memory
      -> per-round JSONL -> bucket/survival/MAT/bootstrap report
```

Luồng E1:

```text
valid anchors -> target forward -> hidden/KV extraction -> CPU shards
              -> document split -> common probe decoder
              -> CE/Top-1/Top-5/prefix@8 -> report and gate
```

Các thành phần code tương ứng:

| File | Trách nhiệm |
|---|---|
| `target_kv_experiments.py` | E0 runner, target/DFlash loading, K loop, raw records |
| `e0_dflash_failure_map.py` | failure-map/context-drop run |
| `e0_report.py` | survival, MAT, bucket và CI |
| `target_kv_e1.py` | target representation extraction và shard |
| `e1_representation_probe.py` | probe công bằng cho hidden/KV variants |
| `e1_report.py` | tổng hợp metric E1 |
| `tests/test_target_kv_*.py` | kiểm tra contract, metric, split và report |

Data contract E0 gồm `dataset`, `doc_id`, `prompt_tokens`, `k`,
`max_new_tokens`, `raw_acceptance_length`, `accepted_draft_tokens`, token ids,
`target_ms`, `draft_ms`, `verify_ms`, `total_ms`, `peak_memory_mb`, `status` và
`error`. E1 lưu representation, target suffix, document id, anchor position,
horizon và split. Điều này cho phép truy nguyên từ một con số trong report về
record raw tương ứng.

### P.3. Chuẩn bị dữ liệu và coverage thực tế

Pipeline đọc JSONL local, loại record thiếu input/tokenization lỗi, tính độ dài
bằng tokenizer target, giữ document id và không pad/truncate để tạo bucket.
E1 chỉ giữ anchor còn đủ suffix 16 token; split theo document để không rò rỉ
anchor của cùng một tài liệu giữa train/dev/test.

| Run | Documents | K rows | Round rows | Bucket |
|---|---:|---:|---:|---|
| GovReport pilot | 13 | 39 | 312 | 2–4K: 3; 4–8K: 10 |
| Multi-News pilot | 49 | 147 | 1,176 | 0–2K: 27; 2–4K: 15; 4–8K: 7 |
| CNN/DM pilot | 30 | 90 | 720 | 0–2K: 30 |
| GovReport confirmation | 5 | 15 | 480 | max-new 32 |
| Multi-News confirmation | 5 | 15 | 480 | max-new 32 |

Pilot có 92 documents, 276 K rows và 2,208 round rows. Tính cả confirmation
là 306 K rows và 3,168 round rows; confirmation dùng subset protocol để kiểm
tra `max_new_tokens=32`, không phải 10 documents độc lập mới.

E1 có 17 documents/34 anchors trên GovReport và 49 documents/98 anchors trên
Multi-News. Split lần lượt là 20/6/8 rows và 58/18/22 rows cho train/dev/test.
Mỗi variant E1 có 158,021,760 trainable parameters, interface 128×64 và
horizon 16.

Coverage này đủ cho quyết định feasibility trong các bucket đã quan sát, nhưng
chưa đủ cho production claim hoặc kết luận natural context 8–16K+: GovReport
chưa phủ đủ bucket dài, CNN/DM chủ yếu là control ngắn, và E1 còn ít anchors.

### P.4. Nhật ký tiến hành, debug và các hiệu chỉnh

#### P.4.1. Preflight và accounting

GPU run dùng Python hệ thống Miniconda ngoài `.venv` theo yêu cầu. Trước khi
chạy dataset, pipeline xác nhận GPU/CUDA/torch, model config, tokenizer và load
được cả target lẫn DFlash.

Một official DFlash cross-check cho raw acceptance `[1, ...]` xác nhận API đã
tính một target fallback token. Vì vậy mọi kết quả dùng
`accepted_draft_tokens = raw_acceptance_length - 1`, clamp tại 0. Nếu không
hiệu chỉnh, zero/positive acceptance sẽ bị đếm sai một cách có hệ thống.

#### P.4.2. Memory và context-drop

Context-drop feasibility trên 8-bit được đo ở 11,052, 16,384 và 28,156 input
tokens, peak lần lượt khoảng 8.08, 9.50 và 12.65 GiB. Input 41,651 bị loại vì
vượt target `max_position_embeddings=40,960`. Ba run 8-bit đều raw acceptance
1, tức accepted draft 0.

8-bit chỉ được dùng để khảo sát khả năng chứa context, không trộn với kết luận
FP16 chính. Vì không có natural long-context bucket đủ lớn, trạng thái của
long-context E0 là `INCONCLUSIVE`, không gán thành FAIL.

#### P.4.3. max-new confirmation

GovReport và Multi-News được chạy lại với `max_new_tokens=32`. Cả hai vẫn có
0 accepted draft token trong toàn bộ K. Do đó zero acceptance không chỉ do
output budget 1 trong scope short/mid-context.

#### P.4.4. E1 OOM và cách sửa

Extraction ban đầu giữ quá nhiều hidden/KV tensor trên GPU. Pipeline được sửa
bằng `inference_mode`, chỉ giữ tensor cần thiết, chuyển chunk về CPU ngay sau
mỗi sample, lưu shard nhỏ và chỉ đưa mini-batch probe cần thiết lên GPU. E1
sau đó chạy hoàn tất với cùng budget giữa các variant; công thức label và kiến
trúc probe không bị đổi.

#### P.4.5. Exact-match guardrail

Exact-match giữa target continuation và token ids dùng trong verification là:
GovReport pilot 36/39; Multi-News 147/147; CNN/DM 90/90; GovReport confirmation
12/15; Multi-News confirmation 15/15. Ba GovReport mismatch cùng thuộc một
document; bảy token đầu trùng, token cuối khác (8397 so với 29340).

Các row mismatch không dùng cho diễn giải speed/quality nhưng vẫn giữ trong
artifact. Nguyên nhân chính xác của mismatch cuối token chưa được xác định;
report không gán tùy tiện cho model, tokenizer hay CUDA.

### P.5. Kết quả E0 và decision gate

Trong GovReport pilot (312 rounds), Multi-News pilot (1,176), CNN/DM pilot
(720), GovReport confirmation (480) và Multi-News confirmation (480), mọi
(K\in\{4,8,16\}) đều cho:

\[
\mathrm{MAT}=0,\qquad S_L(j)=0\quad\text{với mọi }j\ge 1.
\]

Tức là trong mọi proposal hợp lệ đã quan sát, DFlash không chấp nhận token
draft nào trước token fallback. Không thể fit một survival curve dương, không
thể kiểm tra burst acceptance từ E0 này và không có cơ sở đo speedup của
Target-KV block drafting.

| Claim E0 | Trạng thái | Căn cứ |
|---|---|---|
| Có acceptance dương ở short/mid context | **FAIL** | 0/2,208 pilot rounds; confirmation cũng 0 |
| max-new 32 khôi phục acceptance | **FAIL** | GovReport và Multi-News confirmation đều 0 |
| Acceptance ở natural context 8–16K+ | **INCONCLUSIVE** | Không đủ natural bucket dài |
| E0 hiện tại chứng minh serving speedup | **FAIL / không được phép** | Không có accepted draft token |

Diễn giải đúng là “không thấy acceptance trong scope đã chạy”, không phải
“mọi DFlash hoặc mọi Target-KV setup đều bất khả thi”. E0 không cung cấp bằng
chứng để xây một policy online trên cặp snapshot hiện tại.

### P.6. Kết quả E1 và decision gate

#### P.6.1. GovReport

| Variant | CE | Top-1 | Top-5 | Prefix@8 |
|---|---:|---:|---:|---:|
| hidden | 11.7570 | 0.0625 | 0.0625 | 0 |
| hidden_sequence | 11.7173 | 0.0312 | 0.0703 | 0 |
| multi_layer_hidden | 11.8831 | 0.0469 | 0.0469 | 0 |
| kv | 11.9097 | 0.0000 | 0.0312 | 0 |
| kv_shuffled | 11.8993 | 0.0078 | 0.0078 | 0 |
| kv_recent | 11.9146 | 0.0000 | 0.0078 | 0 |
| kv_wrong_document | 11.8871 | 0.0781 | 0.0781 | 0 |

#### P.6.2. Multi-News

| Variant | CE | Top-1 | Top-5 | Prefix@8 |
|---|---:|---:|---:|---:|
| hidden | 11.7273 | 0.0227 | 0.0426 | 0 |
| hidden_sequence | 11.6988 | 0.0341 | 0.0653 | 0 |
| multi_layer_hidden | 11.8607 | 0.0284 | 0.0682 | 0 |
| kv | 11.8899 | 0.0170 | 0.0426 | 0 |
| kv_shuffled | 11.8718 | 0.0227 | 0.0597 | 0 |
| kv_recent | 11.8892 | 0.0057 | 0.0341 | 0 |
| kv_wrong_document | 11.8646 | 0.0369 | 0.0682 | 0 |

Kết quả nhất quán:

- `hidden_sequence` là baseline mạnh nhất hoặc gần mạnh nhất theo CE/Top-5;
- `kv` thua `hidden_sequence` trên cả GovReport (11.9097 so với 11.7173) và
  Multi-News (11.8899 so với 11.6988);
- các controls KV không tạo ra mẫu hình lợi ích ổn định;
- `prefix@8=0` cho mọi variant, không chứng minh được khả năng draft chính xác
  một block 8 token.

| Gate E1 | Quan sát | Trạng thái |
|---|---|---|
| KV thắng hidden sequence ở ít nhất 2 regime | Không thắng ở cả hai | **FAIL** |
| KV có lợi ích nhất quán qua controls | Không có | **FAIL** |
| Probe chứng minh exact block drafting | Prefix@8 đều 0 | **FAIL** |
| E1 cho phép suy ra tokens/ms | E1 không đo serving | **Không được suy luận** |

`kv_wrong_document` đôi lúc có accuracy cao hơn một số variant không có nghĩa
KV hữu ích; đây là control trên sample nhỏ. Gate chính vẫn là so sánh `kv` với
baseline mạnh nhất `hidden_sequence` dưới cùng decoder/budget.

### P.7. Diễn giải thống kê, phạm vi và quyết định tiếp theo

E0 hiện có một kết quả điểm rất mạnh (zero acceptance) nhưng chưa có đủ biến
thiên để ước lượng long-context survival. Do đó cần tách hai loại kết luận:

- **FAIL trong scope:** short/mid-context của cặp snapshot hiện tại, kể cả
  confirmation max-new32.
- **INCONCLUSIVE ngoài scope:** natural context 8–16K+ vì không có đủ bucket
  dài; 8-bit context-drop chỉ là feasibility/memory check.

E1 có cùng kết luận âm trên hai dataset, nhưng sample anchor nhỏ nên không được
diễn giải thành định luật cho mọi model. E1 đủ để đóng gate KV-specific của
phase này, không đủ để phủ định toàn bộ các thiết kế representation khác.

Ma trận quyết định cuối của Target-KV:

| Nhánh | Điều kiện | Kết quả | Hành động |
|---|---|---|---|
| Tiếp tục Target-KV | E0 acceptance dương và E1 KV có signal | Cả hai gate không đạt | Dừng phase |
| Mở rộng block controller | MAT dương và survival theo vị trí | MAT=0 | Không triển khai |
| Tối ưu KV adapter | KV thắng hidden sequence | KV thua trên 2 dataset | Không đầu tư |
| Chạy E2 serving | E0/E1 pass | Chưa pass | Chưa chạy |
| Chạy E3 strong-drafter | Có hiện tượng cần replicate | Chưa có tín hiệu | Chưa chạy |

Để mở lại hướng này cần một protocol mới gồm: sanity acceptance dương ở
context ngắn, snapshot target/drafter được xác nhận aligned, natural long
context đủ documents, E0 MAT dương, E1 split lớn hơn với CI document-level và
KV thắng hidden sequence; chỉ sau đó mới chạy E2 latency.

Không được dùng các kết quả hiện tại để nói rằng long-context hypothesis đã bị
bác bỏ ở mọi context, rằng CE/accuracy là speedup, rằng nguyên nhân là
misalignment/tokenizer/kernel khi chưa có thí nghiệm nguyên nhân, hoặc rằng
DFlash nói chung không hoạt động.

### P.8. Artifact và lệnh tái lập Target-KV

Các artifact nằm trong `src/analyze/groundsync/`:

```text
target_kv_experiments.py       # E0 runner
e0_dflash_failure_map.py       # E0 failure map/context feasibility
e0_report.py                   # E0 aggregation
target_kv_e1.py                # E1 extraction
e1_representation_probe.py     # E1 common probe
e1_report.py                   # E1 aggregation
tests/test_target_kv_*.py      # tests cho code và report
results/tkv-*                  # raw JSONL, summary, probe outputs
```

Model snapshots:

```text
/home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
/home/tuantb/.cache/huggingface/hub/models--z-lab--Qwen3-4B-DFlash-b16/snapshots/61ab4992e5b5ec5913c7f8a9618367b4309533a3
```

Lệnh protocol đại diện:

```bash
cd /home/tuantb/fast_infer_text_sum
export CUDA_VISIBLE_DEVICES=0
export TARGET_MODEL=/home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
export DFLASH_MODEL=/home/tuantb/.cache/huggingface/hub/models--z-lab--Qwen3-4B-DFlash-b16/snapshots/61ab4992e5b5ec5913c7f8a9618367b4309533a3

python3 src/analyze/groundsync/target_kv_experiments.py \
  --target-model "$TARGET_MODEL" --drafter-model "$DFLASH_MODEL" \
  --data-file data/govreport/test.jsonl \
  --output-dir src/analyze/groundsync/results/tkv-e0-pilot-gov-20260903 \
  --k-values 4 8 16 --max-new-tokens 32

python3 src/analyze/groundsync/e0_report.py \
  --input src/analyze/groundsync/results/tkv-e0-pilot-gov-20260903 \
  --output src/analyze/groundsync/results/tkv-e0-pilot-gov-20260903/report.json

python3 src/analyze/groundsync/target_kv_e1.py \
  --target-model "$TARGET_MODEL" --data-file data/govreport/test.jsonl \
  --output-dir src/analyze/groundsync/results/tkv-e1-gov-20260903 \
  --max-anchors 34 --horizon 16

python3 src/analyze/groundsync/e1_representation_probe.py \
  --input-dir src/analyze/groundsync/results/tkv-e1-gov-20260903 \
  --output-dir src/analyze/groundsync/results/tkv-e1-gov-20260903/probe \
  --variants hidden hidden_sequence multi_layer_hidden kv kv_shuffled kv_recent kv_wrong_document

python3 src/analyze/groundsync/e1_report.py \
  --input-dir src/analyze/groundsync/results/tkv-e1-gov-20260903/probe \
  --output src/analyze/groundsync/results/tkv-e1-gov-20260903/probe_report.json
```

Multi-News và CNN/DM dùng cùng runner, chỉ thay `--data-file` và output
directory; confirmation giữ `max_new_tokens=32` và giới hạn documents như bảng
coverage. Không được thay acceptance accounting khi tái lập.

Kiểm chứng code/artifact sau khi bổ sung report:

```text
python3 -m pytest -q src/analyze/groundsync/tests   -> 94 passed
python3 -m compileall -q src/analyze/groundsync   -> exit 0
git diff --check -- src/analyze/groundsync         -> exit 0
```

### P.9. Kết luận cuối cho toàn bộ phase

File master này hiện bao gồm cả quá trình GroundSync/BurstSpec P0–P2 trước đó
và follow-up Target-KV E0/E1: từ thiết kế hypothesis, cố định protocol, xây
dựng code, chuẩn bị dữ liệu, thực thi trên T4, xử lý giới hạn/OOM/exact-match,
đến raw artifact, metric, gate và quyết định.

Kết luận tổng hợp không thay đổi:

```text
NO_GO_GROUNDSYNC_GENERAL
NO_GO_BURSTSPEC_GENERAL
NO_GO_TARGET_KV_IN_CURRENT_SCOPE
ORACLE_HEADROOM_EXISTS_BUT_ONLINE_POLICY_NOT_VALIDATED
E0_NATURAL_LONG_CONTEXT_INCONCLUSIVE
P2_SERVING_UNAVAILABLE_OR_GUARDRAIL_FAIL
```

E2/E3 Target-KV chưa chạy có chủ đích vì E0/E1 không đạt gate tiền đề; đây là
quyết định dừng theo protocol, không phải thiếu báo cáo hay bỏ sót artifact.
