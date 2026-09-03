# Báo cáo tổng hợp kiểm định P0 — GroundSync và BurstSpec

Ngày chốt: 2026-09-02  
Nguồn số liệu chính: [`results/p0-decision-final9-20260902/`](results/p0-decision-final9-20260902/)  
Analyzer: [`p0_decision.py`](p0_decision.py)  
Config: [`p0_decision_config_20260902.json`](p0_decision_config_20260902.json)

## 1. Kết luận điều hành

Nhóm P0 đã được thực hiện đầy đủ trên GovReport và CNN/DailyMail với
acceptance trace `Kmax=16`, timing trace có đủ `k={0,2,4,8,16}`, và
multi-start trace đủ 9 round/document để đo cả `delta=1,2,4,8`.

Quyết định tổng hợp là:

```text
NO_GO_GROUNDSYNC_GENERAL;
conditional BurstSpec follow-up only where admission passes
```

Diễn giải chính:

- GroundSync transition có tín hiệu dương ở GovReport nhưng không tái lập ở
  CNN/DailyMail; không đủ bằng chứng cho claim cross-regime.
- Corrected Grounding Oracle không vượt best available baseline trên test ở
  cả hai dataset. Nó có thể tốt hơn adaptive baseline nếu chỉ so với adaptive,
  nhưng fixed baseline nhanh hơn nên GroundSync vẫn không có utility triển khai.
- True-cost oracle của ladder có headroom lớn ở cả hai dataset. Đây là ceiling
  do biết trước accepted length/chi phí, không phải speedup của một policy online.
- First-token admission oracle thu hồi `53.8–73.8%` gap ở GovReport, nhưng chỉ
  `30.3%` ở mức tốt nhất trên CNN/DailyMail. Vì vậy BurstSpec chỉ đáng follow-up
  có điều kiện, chưa được xác nhận là hướng tổng quát.
- Acceptance burstiness đạt tiêu chí nominal ở cả hai dataset khi dùng 9 start
  và document-bootstrap CI; tuy nhiên độ lớn persistence còn nhỏ và không phải
  bằng chứng production.

## 2. Câu hỏi, gate và cách kiểm định

| Mã | Câu hỏi | Gate quyết định |
|---|---|---|
| P0-1 corrected H2 | Drift đi qua transition có liên quan đến first rejection không? | Hệ số drift của discrete hazard có lower 95% document-bootstrap CI `>0` |
| P0-2 corrected H4 | Grounding horizon có utility khi chọn `k` không? | Corrected oracle phải hơn best available test baseline ít nhất `5%`; threshold chỉ tune trên train/dev |
| P0-3 ladder | Headroom nằm ở admission/`k=0`/`k=16` hay chỉ ở chọn `k`? | Mỗi oracle ladder đạt headroom `>=8%` so với best fixed positive |
| P0-4 admission | Chỉ biết `accepted_len > 0` có đủ để quyết định speculate không? | Thu hồi ít nhất `40%` gap từ best fixed tới true-cost oracle |
| P0-5 burstiness | Có asymmetry trong block và persistence giữa các round không? | `h1/later_mean > 1` và ít nhất một `delta` có lower 95% document-bootstrap CI của excess `>0` |

Trong P0-1, với proposal bắt đầu tại `t`, predictor chính là:

```text
d_transition[t,j] = JS(g[t+j-1], g[t+j])
```

Risk set chỉ giữ proposal còn sống tại relative position `j`; controls gồm
target entropy, draft confidence, relative position và absolute output
position. Bootstrap lấy document làm đơn vị, không coi các token cùng document
là mẫu độc lập.

Trong P0-2, transition không xuất hiện trong `Kmax` được mã hóa là `Kmax`,
không phải `1`. Threshold được chọn bằng utility trên train+dev và đánh giá
trên test document. Best available baseline là policy nhanh nhất trong fixed
positive và hai adaptive baseline (`adaptive_entropy`, `adaptive_history`).

Trong P0-3/P0-4, `k=0` có chi phí AR cached one-token đo riêng. Với `k>0`, chi
phí là draft incremental decode cộng cached target block verification. Mọi
ladder policy dùng cùng common timing rows, đủ chi phí cho toàn bộ
`k={0,2,4,8,16}`. Admission oracle chỉ đọc bit `accepted_len > 0`, không đọc
accepted length đầy đủ.

Trong P0-5, multi-start dùng 9 start:
`1,4,7,10,13,16,19,22,25`, `max_k=4`. Nhờ vậy round offset `1,2,4,8` đều
được biểu diễn; proposal kết thúc sớm được censor sau số token quan sát thay vì
coi là còn sống tới `Kmax`.

## 3. Môi trường và model

### GPU runner

- Host: `tuantb@teslaT4`.
- GPU: NVIDIA Tesla T4, compute capability 7.5, 15,360 MiB.
- Driver: `550.163.01`; CUDA runtime: `12.4`.
- GPU executable: `/home/tuantb/miniconda3/bin/python3`, chạy ngoài `.venv`.
- Torch GPU stack: `2.6.0+cu124`.
- `CUDA_VISIBLE_DEVICES=0`, batch size 1, greedy decoding, seed `42`.
- T4 `sm75` không dùng native Flash SDP phù hợp context dài trong stack này;
  target/draft/verifier dùng chunked causal prefill với
  `prefill_chunk_size=512`. Đây là biện pháp runtime, không thay đổi metric.

### Model/cache

- Target: Qwen3-4B local snapshot tại
  `/home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`.
- Draft: Qwen3-0.6B tại `/home/tuantb/models/Qwen3-0.6B`.
- Cả hai model đều được load local-only; không tải mạng trong lúc chạy.
- Qwen3-4B EAGLE head không được dùng làm target vì khác architecture; P0 là
  controlled target-vs-draft acceptance, chưa phải EAGLE-3 serving.

Analyzer deterministic chạy bằng `.venv/bin/python3` Python `3.12.13` vì chỉ
đọc JSONL và tính metric; điều này không vi phạm điều kiện GPU: mọi run model
trên T4 đều dùng executable ngoài venv nêu trên. Manifest analyzer ghi platform
và command đầy đủ.

## 4. Coverage và raw artifacts

| Dataset | Target | Acceptance `Kmax=16` | Timing request / đủ ladder | Multi-start cuối |
|---|---:|---:|---:|---:|
| GovReport | 99/100 ok; 1 target error | 99 ok; 1 thiếu target tương ứng | 56 row `ok`, 55 đủ mọi `k` | 891 ok = 99 × 9 |
| CNN/DailyMail | 100/100 ok | 100 ok | 50/50 đủ mọi `k` | 900 ok = 100 × 9 |

Các nguồn raw:

- Gov target/acceptance: `results/qwen3-4b-gov100-gpu-protocol-20260830/`,
  `results/p0-gov100-k16-20260902/`.
- CNN target/acceptance: `results/qwen3-4b-cnn100-gpu-protocol-20260830/`,
  `results/p0-cnn100-k16-20260902/`.
- Corrected timing: `results/p0-gov50-k16-timing-cached-20260902/`,
  `results/p0-gov10-k16-timing-cached-cont-20260902/`,
  `results/p0-cnn50-k16-timing-cached-20260902/`.
- Multi-start cuối: `results/p0-gov100-multistart9-k4-20260902/` và
  `results/p0-cnn100-multistart9-k4-20260902/`.

Một GovReport timing row có `status=ok` nhưng chỉ có timing tới `k=6`; analyzer
loại row này khỏi common ladder, vì vậy GovReport chỉ còn 55 row hoàn chỉnh.
GovReport multi-start có đúng một error do
`govreport_GAO_GAO-20-170SP` không có target trace. CNN/DailyMail có 3 proposal
ngắn hơn 4 token do EOS; analyzer censor phần chưa quan sát, không bịa rejection.

## 5. Kết quả chi tiết

### P0-1 — corrected within-block transition H2

| Dataset | Risk rows | Event rows | Drift coefficient / SD | 95% document-bootstrap CI | Decision |
|---|---:|---:|---:|---:|---|
| GovReport | 302 | 96 | 0.24378 | `[0.01101, 0.59689]` | PASS |
| CNN/DailyMail | 539 | 97 | 0.09078 | `[-0.29372, 0.36339]` | FAIL |

GovReport có odds ratio khoảng `1.276` cho mỗi 1 SD drift sau controls. Tuy
nhiên CNN/DailyMail có CI bao phủ 0, nên quyết định cross-regime là `MIXED`,
không xác nhận GroundSync transition hypothesis tổng quát.

### P0-2 — corrected Grounding Oracle H4

`tokens/ms` là tổng committed tokens chia tổng chi phí đo được. Gain được tính
trên test document; adaptive-only gain được báo riêng để không che khuất fixed
baseline nhanh hơn.

| Dataset | Threshold | Test docs | Corrected oracle | Best fixed test | Best generic adaptive test | Gain vs best available | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| GovReport | 0.010 | 11 | 0.002591 | fixed-k2: 0.002735 | entropy: 0.001289 | -5.24% | FAIL |
| CNN/DailyMail | 0.005 | 10 | 0.007706 | fixed-k4: 0.011570 | entropy: 0.006007 | -33.40% | FAIL |

Grounding Oracle cao hơn adaptive entropy nếu chỉ so hai policy: khoảng
`+101.1%` ở GovReport và `+28.3%` ở CNN/DailyMail. Nhưng best fixed nhanh hơn
cả hai; do đó tiêu chí đúng cho quyết định hệ thống vẫn là FAIL ở cả hai
dataset. Đây là lý do không tiếp tục tối ưu estimator GroundSync ở P0.

### P0-3 — Oracle Ladder

| Dataset | `k=0` | `k=2` | `k=4` | `k=8` | `k=16` | True-cost oracle | O1 | O2 | O3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GovReport | 0.002990 | 0.003169 | 0.003015 | 0.002471 | 0.001817 | 0.005460 | +46.8% | +64.1% | +72.3% |
| CNN/DailyMail | 0.006780 | 0.010472 | 0.011621 | 0.009645 | 0.005709 | 0.014357 | +20.4% | +22.3% | +23.5% |

O1 là `{2,4,8}`, O2 là `{0,2,4,8}`, O3 là `{0,2,4,8,16}`. Cả ba level đều
PASS theo gate headroom `>=8%` ở từng dataset. Kết quả này chứng minh có
opportunity ceiling, đặc biệt admission/AR và chọn block theo từng round; nó
không chứng minh policy online có thể đạt oracle.

### P0-4 — First-token Admission Oracle

Recovery là phần gap từ best fixed tới true-cost oracle được thu hồi khi chỉ
dùng bit `accepted_len > 0`.

| Dataset | k=2 | k=4 | k=8 | k=16 | Dataset decision |
|---|---:|---:|---:|---:|---|
| GovReport | 18.1% | 53.8% | **73.8%** | 71.2% | PASS |
| CNN/DailyMail | 6.3% | 30.3% | 22.3% | 8.8% | FAIL |

Admission oracle là tín hiệu mạnh ở GovReport-like regime, với k=8 tốt nhất;
nhưng không tái lập trên CNN/DailyMail. Cross-regime decision là `MIXED`.

### P0-5 — Acceptance Burstiness / Persistence

| Dataset | h1 | Mean h[j>1] | h1/mean later | Delta có CI lower > 0 | Decision |
|---|---:|---:|---:|---|---|
| GovReport | 0.3064 | 0.1961 | 1.5623 | `delta=1`: excess 0.0354, CI `[0.0118, 0.0583]` | PASS |
| CNN/DailyMail | 0.2367 | 0.2264 | 1.0452 | `delta=4`: excess 0.0240, CI `[0.0016, 0.0481]` | PASS |

Chi tiết excess (conditional probability trừ marginal) và 95% document-bootstrap
CI:

| Dataset | delta=1 | delta=2 | delta=4 | delta=8 |
|---|---|---|---|---|
| GovReport | `0.0354 [0.0118, 0.0583]` | `0.0084 [-0.0125, 0.0297]` | `0.0052 [-0.0235, 0.0359]` | `0.0469 [-0.0830, 0.1771]` |
| CNN/DailyMail | `0.0045 [-0.0133, 0.0225]` | `0.0075 [-0.0123, 0.0271]` | `0.0240 [0.0016, 0.0481]` | `-0.0035 [-0.0342, 0.0291]` |

P0-5 PASS nghĩa là có ít nhất một persistence offset đạt CI gate trong mỗi
dataset, không nghĩa mọi offset đều dương. Hiệu ứng nhỏ; cần strong-drafter
replication trước khi coi đây là phenomenon ổn định.

## 6. Kiểm tra tính đúng đắn và validation

Các sửa có ảnh hưởng trực tiếp đến tính hợp lệ của P0:

1. H2 dùng drift crossing transition tại từng `j`, không dùng
   `drift_at_start` làm predictor chính.
2. `NULL horizon -> Kmax`, và threshold được freeze sau train/dev.
3. `k=0` dùng target cached one-token timing riêng; draft prefill không bị tính
   lẫn vào chi phí per-round.
4. Mọi fixed/oracle ladder policy dùng cùng complete timing rows, tránh bias do
   một `k` có subset khác.
5. Start position được bound để canonical continuation đủ độ dài; draft EOS sớm
   được censor đúng theo số token quan sát.
6. Persistence dùng document-bootstrap 2.000 lần; risk/hazard H2 cũng bootstrap
   theo document 2.000 lần.

Validation cuối:

```text
57 passed
python3 -m compileall -q src/analyze/groundsync       PASS
git diff --check                                      PASS
```

Artifact cuối gồm [`p0_metrics.json`](results/p0-decision-final9-20260902/p0_metrics.json),
[`p0_metrics.csv`](results/p0-decision-final9-20260902/p0_metrics.csv),
[`p0_decision_report.md`](results/p0-decision-final9-20260902/p0_decision_report.md),
ba PNG và [`p0_manifest.json`](results/p0-decision-final9-20260902/p0_manifest.json).
`p0_manifest.json` ghi command, Python, platform, config và danh sách artifact.

## 7. Quyết định nghiên cứu sau P0

### GroundSync

Không triển khai tiếp GroundSync estimator/horizon predictor như hướng tổng
quát ở thời điểm này. Lý do quyết định là H2 chỉ `MIXED` và corrected H4
`FAIL` khi so với best available baseline ở cả hai regime.

### BurstSpec

Giữ BurstSpec ở trạng thái follow-up có điều kiện, không phải go cho production:

- GovReport cho admission recovery tốt và oracle ladder có headroom lớn;
- CNN/DailyMail có oracle headroom nhưng admission bit chưa đủ mạnh;
- burstiness có tín hiệu nominal nhưng effect nhỏ và chưa có strong-drafter
  replication.

Nếu tiếp tục P1, thứ tự hợp lý là: train cheap admission predictor trên
GovReport-like regime với split theo document; replicate burst/admission bằng
strong aligned drafter; chỉ sau đó mới chạy E2E serving. Không nên train
GroundSync horizon predictor hay triển khai controller production trước các
kiểm tra đó.

### Phạm vi chưa claim

P0 chưa phải benchmark EAGLE-3/vLLM production, chưa đo Multi-News/XSum, chưa
đánh giá strong drafter, và chưa chứng minh causal attribution của attention.
Các giới hạn này được giữ rõ trong report thay vì chuyển thành PASS giả.
