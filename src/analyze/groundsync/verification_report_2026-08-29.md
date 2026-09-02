# Báo cáo toàn bộ quá trình kiểm định giả thiết GroundSync

**Ngày thực hiện:** 29/08/2026 (UTC)  
**Ngày bổ sung audit:** 30/08/2026 (UTC)  
**Máy:** `tuantb@teslaT4`  
**Phạm vi:** kiểm định mở rộng bằng Qwen3-4B làm target và Qwen3-0.6B làm draft, với trace target greedy, controlled speculative acceptance, attention source-state, calibration, sensitivity, document split và timing subset trên CUDA. GovReport là regime chính; CNN/DailyMail là cross-regime độc lập.  
**Artifact chính:** [`results/qwen3-4b-gov100-gpu-protocol-20260830/`](results/qwen3-4b-gov100-gpu-protocol-20260830/) và [`results/qwen3-4b-cnn100-gpu-protocol-20260830/`](results/qwen3-4b-cnn100-gpu-protocol-20260830/)

## 1. Kết luận ngắn gọn

Các giả thiết **chưa được xác nhận thành công ở mức kết luận tổng quát**.
Run mở rộng có 99/100 GovReport target hợp lệ, 99 controlled proposals và 10
proposal test có timing phủ đủ `k=8`; CNN/DailyMail có 100/100 target,
100 controlled proposals và 12 proposal test có timing phủ đủ `k=8`. Bảng
headline dùng GovReport làm regime chính; CNN/DailyMail được báo riêng để
kiểm tra khả năng chuyển miền:

| Giả thiết | Kết quả discovery | Diễn giải ngắn |
|---|---|---|
| H1 — source-state có tính bền theo thời gian | `FAIL` | No-sink đạt gate, nhưng calibrated không đạt; kết luận robust phải fail. |
| H2 — drift dự báo speculative rejection | `FAIL` GovReport; `PASS` CNN/DailyMail | Hazard có điều chỉnh theo vị trí và bootstrap theo document; hai regime cho chiều khác nhau. |
| H3 — grounding signal tăng khả năng dự báo ngoài control | `FAIL` | GovReport AUROC gain `0`; CNN/DailyMail no-sink gain `-0,0267`; calibrated CNN có gain `+0,0267` nhưng không cứu estimator chính. |
| H4 — oracle horizon tạo lợi ích tính toán | `FAIL` | GovReport grounding oracle chậm hơn fixed `k=8` 44,3%; CNN/DailyMail chậm hơn 43,9%. |
| H5 — horizon dự đoán được online | `UNAVAILABLE` | Cả hai regime có predictor metric, nhưng oracle horizon không nhanh hơn fixed policy nên recovery không hợp lệ. |

Vì vậy, kết quả hiện tại không ủng hộ claim tổng hợp GroundSync: H1 không
robust qua calibration và regime, H2 không tái lập cùng chiều giữa hai regime,
H3/H4 không đạt gate ở estimator chính, còn H5 chưa đủ điều kiện utility. Đây
vẫn là **discovery/controlled evidence**, chưa phải kết luận cuối cho mọi
dataset, model, context length hoặc implementation speculative production.

## 2. Các giả thiết và tiêu chí quyết định

Proposal GroundSync được tách thành năm giả thiết thực nghiệm:

| Mã | Claim cần kiểm định | Cách đo chính | Gate quyết định |
|---|---|---|---|
| H1/E1 | Vector source-utilization tạo ra state có persistence theo thời gian. | So sánh `1 - JS` giữa các bước kề nhau với null shuffle trong từng document; bootstrap ở mức document; kết quả phải ổn định ở no-sink và calibrated. | Cận dưới CI 95% của persistence excess `>= 0,02` ở estimator chính và calibrated control. |
| H2/E2 | Drift của source-state tại đầu block dự báo first speculative rejection. | Hazard discrete theo relative position; logistic drift coefficient điều chỉnh vị trí; cluster bootstrap theo document; median split là mô tả phụ. | Cận dưới CI 95% của drift coefficient `> 0`. |
| H3/E3 | Grounding signal mang thông tin tăng thêm sau entropy, draft confidence, acceptance history, vị trí, sentence boundary, copyability và `max_k`. | Logistic predictor split theo document; so sánh AUROC/AUPRC/log-loss/Brier baseline với full feature set; có position-only, temporal-shift và shuffle controls. | AUROC gain `>= 0,02`, có đủ test variation và không chỉ xuất hiện ở raw estimator. |
| H4/E4 | Biết horizon giúp chọn block speculative có utility về số token committed trên thời gian. | Replay fixed và oracle trên cùng proposal; dùng draft time + target verification time đo theo từng `k`. | Speed gain oracle so với fixed `>= 0,08`. Acceptance-only không được dùng để claim speed. |
| H5/E5 | Grounding horizon có thể dự đoán online từ tín hiệu hiện tại/quá khứ. | Chọn horizon threshold trên train/dev; predictor document-split từ entropy, vị trí, source concentration, drift, draft confidence và recent acceptance; đánh giá policy trên timing test. | Oracle-gain recovery `>= 0,50`; chỉ tính khi oracle nhanh hơn fixed và có đủ hai lớp nhãn. |

`E0` là kiểm soát confounder attention sink/position bias. Pipeline lưu và
so sánh biến thể `raw` với `nosink`; attention được xem là tín hiệu quan sát,
không phải causal attribution hay ground truth.

Quy ước trạng thái:

- `PASS`: metric vượt gate và coverage hợp lệ.
- `FAIL`: có coverage hợp lệ nhưng metric không ủng hộ claim.
- `UNAVAILABLE`: metric không thể tính hợp lệ, thường do thiếu class, thiếu
  horizon hoặc thiếu timing.
- `INCONCLUSIVE`: có một phần dữ liệu nhưng coverage dưới ngưỡng tối thiểu.

### 2.1. Phân biệt kết quả đã chạy với protocol đầy đủ

Bảng dưới đây là checklist quan trọng để tránh hiểu nhầm rằng mọi mục trong
proposal đều đã được chạy đầy đủ:

| Hạng mục | Trạng thái thực tế | Có dùng làm kết luận chính? |
|---|---|---|
| Synthetic metric/evaluator | Đã chạy nhiều fixture, gồm 12 target documents và 120 controlled rows. | Không; chỉ kiểm tra pipeline. |
| Qwen3-4B/Qwen3-0.6B CPU smoke | Đã chạy 1 document để kiểm tra loader, target, draft, verifier và timing. | Không; thiếu coverage. |
| CNN/DailyMail CPU discovery | Đã chạy 10 documents, 80 target steps và 20 proposals. | Chỉ là evidence phụ trợ. |
| GovReport GPU diagnostic OOM | Đã chạy, nhưng các run eager/math prefill bị OOM. | Không; chỉ dùng để chẩn đoán runtime. |
| GovReport GPU chunked smoke | Đã chạy 1 document dài, không OOM sau sửa prefill. | Không; chỉ xác nhận hạ tầng đo. |
| GovReport GPU discovery chính | 25 target rows, 50 controlled speculative rows, timing đầy đủ. | Có; đây là evidence chính của báo cáo. |
| 100 GovReport và cross-regime GPU | Đã chạy 100 GovReport target (99 ok do 1 OOM), 99 controlled proposals và 10 timing rows; CNN/DailyMail có 100/100 target, 100 controlled proposals và 12 timing rows. | Có cross-regime cho H1–H5 ở mức controlled; timing vẫn là subset nhỏ, không phải serving E2E. |
| E0 calibrated/position-relocation/chunk sensitivity | Đã chạy raw/no-sink/calibrated, chunk 64/128/256, sink 4/8/16 và fixture Qwen3-4B với cùng evidence ở đầu/giữa/cuối nguồn. | E0 cho thấy position confounder còn rõ sau no-sink; không dùng raw attention như attribution độc lập. |
| H2 hazard/survival và bootstrap theo proposal | Đã implement first-rejection hazard theo relative position và logistic drift coefficient điều chỉnh vị trí; bootstrap 2.000 lần theo document trên 99 GovReport và 100 CNN/DailyMail proposals. | H2 được kiểm định bằng coefficient gate; vẫn là association, không causal. |
| H3 negative controls và đầy đủ grounding feature set | Đã chạy baseline/full, position-only, temporal shift 10/20/50, shuffle và estimator no-sink/calibrated trên cả hai regime; multi-start bổ sung kích hoạt lag-drift/history. | Metric chính no-sink không đạt incremental AUROC gate; một calibrated CNN result dương được giữ như sensitivity, không thay primary. |
| H4 toàn bộ fixed/adaptive/true-cost/EAGLE serving | Đã chạy fixed `k=2,4,8`, adaptive entropy/history và true-cost oracle trên 10 GovReport và 12 CNN rows phủ đủ `k=8`; chưa có EAGLE/vLLM serving E2E. | Controlled timing evidence, không phải production throughput. |
| H5 oracle-gain recovery | GovReport chọn threshold `0,05` trên 78 train/dev documents, test 21 rows và timing 10 rows; CNN chọn `0,03` trên 80 train/dev documents, test 20 rows và timing 12 rows. | Cả hai predictor có metric; oracle horizon chậm hơn fixed ở cả hai regime nên recovery `UNAVAILABLE`. |

Do đó, từ “đã kiểm định” trong báo cáo được hiểu là **đã có phép đo cho
run discovery tương ứng**, không có nghĩa là toàn bộ protocol confirmatory
trong proposal đã hoàn tất.

### 2.2. Phiên bản tiêu chí được sử dụng

Design document có review criteria tổng quát: H1/H2/H3 cần vượt null hoặc
incremental gain `>0`, H4 cần speed gain `>=0,08`, còn H5 cần
oracle-gain recovery `>=0,50`. Báo cáo vận hành dùng ngưỡng bảo thủ hơn cho
H1 (`CI lower >=0,02`) và H3 (`AUROC gain >=0,02`); H5 chỉ tính recovery khi
oracle nhanh hơn fixed và có đủ hai lớp. H2 dùng gate mới, chặt hơn mô tả
median split: cận dưới CI 95% của coefficient drift trong hazard logistic
điều chỉnh theo relative position phải `>0`; bootstrap resample nguyên
document. Các ngưỡng, threshold selection và đơn vị bootstrap được khóa trong
`metrics.json` của run mới, không thay đổi để làm đẹp kết quả.

Với run mới, H1 no-sink có CI lower `0,021996` nhưng calibrated chỉ
`0,018257`, nên quyết định composite là `FAIL`. H2 GovReport có coefficient
`-0,0657`, CI `[-0,0664; -0,0535]` nên FAIL; CNN/DailyMail có point estimate
`0,00054`, bootstrap CI `[0,0181; 0,0301]` nên PASS trong regime đó. H5 có
predictor metric nhưng không có oracle headroom dương nên recovery là `None`
và `UNAVAILABLE`.

## 3. Quá trình thực hiện

### 3.1. Khảo sát proposal, repository và dữ liệu

Đã đọc proposal GroundSync trong shared conversation, đọc các quy định của
repository và khảo sát những gì có sẵn trong `src/analyze`, `data/` và cache
model. Các nguyên tắc được giữ lại trong implementation:

1. Không dùng attention raw như ground truth.
2. Không coi acceptance rate là speedup nếu chưa đo thời gian.
3. Split train/dev/test theo document, không split theo token trong cùng
   document.
4. Thiếu model, timing hoặc class variation phải tạo `UNAVAILABLE`, không
   suy diễn thành `PASS`.
5. Mọi run mới ghi vào thư mục riêng dưới `src/analyze/groundsync/results/`.

Dữ liệu được dùng:

- Fixture synthetic: kiểm tra đường ống metric và artifact, không dùng làm
  bằng chứng model.
- CNN/DailyMail 10 mẫu: discovery CPU phụ trợ.
- GovReport representative: regime chính; run T4 mở rộng dùng 100 document,
  trong đó 99 target trace hợp lệ và 99 controlled proposals hợp lệ.
- CNN/DailyMail representative: run T4 cross-regime dùng 100/100 target trace,
  100 controlled proposals và 12 timing rows hợp lệ để kiểm định H1–H5 ở
  regime thứ hai.

### 3.2. Audit cache model

Cache được kiểm tra bằng config, tokenizer và mở file safetensors; loader dùng
`local_files_only=True`.

| Model/path | Vai trò | Kiểm tra |
|---|---|---|
| `/home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c` | Canonical target | `Qwen3ForCausalLM`, 36 layers, hidden size 2560, max position 40960, 3 shard safetensors. |
| `/home/tuantb/models/Qwen3-0.6B` | Draft và verifier phụ trợ | `Qwen3ForCausalLM`, 28 layers, hidden size 1024, max position 40960, safetensors mở thành công. |
| `/home/tuantb/models/Qwen3-4B_eagle3` | Không dùng | Là `Eagle3LlamaForCausalLM`, chỉ một head EAGLE, không phải canonical Qwen3-4B target. |
| Qwen3-1.7B | Không dùng | Không có snapshot local và không cần cho cặp target/draft cuối. |

Kích thước đã kiểm tra: target khoảng 8.06 GB trên 3 shard; draft khoảng
1.52 GB; thư mục EAGLE candidate khoảng 437 MB.

Qwen3-0.6B ban đầu không có trong cache cần thiết, nên đã được tải bổ sung
vào `/home/tuantb/models/Qwen3-0.6B` theo yêu cầu thực nghiệm. Sau khi tải,
config, tokenizer và safetensors đã được kiểm tra offline; không dùng mạng
trong các run model-backed. Qwen3-1.7B không được tải vì không cần cho cặp
target/draft cuối.

### 3.3. Thiết kế và implementation

Code được viết dưới `src/analyze/groundsync/`, gồm:

- `core.py`: normalize distribution, Jensen–Shannon divergence, lag
  similarity, segment length, grounding horizon, accepted prefix, bootstrap
  CI và policy replay.
- `trace_target.py`: render prompt, xác định source span, sinh canonical
  target greedy trace, lấy attention query cuối, gom attention source theo
  chunk và tính entropy/copyability/sentence boundary.
- `trace_speculative.py`: draft Qwen3-0.6B sinh continuation greedy; target
  Qwen3-4B sinh continuation canonical; accepted prefix được so sánh token-id
  chính xác; lưu first rejection, drift, draft confidence và timing theo
  từng `k`.
- `report.py`: tổng hợp H1–H5, document split, bootstrap, policy replay,
  raw/no-sink/calibrated estimator, relative-position hazard, sensitivity,
  negative controls, threshold selection train/dev, CSV/PNG/Markdown và trạng
  thái bảo thủ.
- `run_experiment.py`: orchestrator cho các phase `synthetic`, `target`,
  `speculative`, `analyze`, `all`.

Với mỗi output position, attention của các layer/head được collapse thành
vector source; source token được gom thành các chunk có kích thước mặc định
trong run cuối là 128, bỏ 8 token đầu để tạo biến thể `nosink`. Similarity
được tính là `1 - JS(p_t, p_{t+1})`.

### 3.4. Sai khác giữa design và implementation thực tế

Các sai khác dưới đây được ghi rõ để báo cáo không làm cho implementation
hiện tại trông đầy đủ hơn thực tế:

| Thành phần trong design | Thực tế đã implement/chạy | Hệ quả |
|---|---|---|
| E0: raw, no-sink, calibrated và position-relocation fixture | Đã lưu raw/no-sink; học positional prior 32 bins trên train documents; chạy calibrated/sensitivity chunk 64/128/256, sink 4/8/16; fixture model-backed 3 case đầu/giữa/cuối đã chạy. | E0 hoàn tất ở mức diagnostic nhỏ; position effect còn rõ nên attention không phải ground truth. |
| H1: lag đến 32, segment `L_G`, raw/no-sink/calibrated, sensitivity 64/256 | Đã chạy lag 1/2/4/8/16/32 trên 99 GovReport và 100 CNN; output 32 nên lag 16/32 có cặp; báo riêng raw/no-sink/calibrated. | H1 fail robust vì calibrated dưới gate; CNN cũng dưới gate. |
| H2: hazard/survival theo relative draft position và cluster bootstrap | Accepted prefix/first rejection đo chính xác; hazard risk-set/event theo vị trí 1..8; logistic coefficient điều chỉnh vị trí và bootstrap 2.000 lần theo document trên 99/100 proposal. | H2 là association; GovReport FAIL, CNN/DailyMail PASS theo coefficient gate, không causal. |
| H3: full model thêm grounding features, estimator variant và negative controls | Full có source concentration, drift, missingness, lag drift/history và horizon; controls gồm position-only, shift 10/20/50, shuffle; so sánh no-sink/calibrated trên cả hai regime. | H3 no-sink primary không đạt gate; calibrated CNN là sensitivity dương nhưng không thay primary. |
| H4: fixed `k=2,4,8`, adaptive history/entropy, oracle và true-cost oracle | Đã chạy trên 10 rows phủ đủ `k=8`; timing cached target check tách riêng; có fixed/adaptive/grounding oracle/true-cost oracle. | Grounding oracle fail; chưa phải EAGLE/vLLM E2E. |
| H5: online features gồm `g_t`, drift history, entropy, draft confidence, recent acceptance; threshold chọn train/dev; metric oracle-gain recovery | Threshold `0,05` chọn trên train/dev GovReport và `0,03` trên CNN/DailyMail; predictor đánh giá 21/20 test rows; policy timing trên 10/12 rows với điều kiện oracle headroom. | Oracle chậm hơn fixed ở cả hai regime nên recovery `UNAVAILABLE`, không gọi PASS. |

Các sai khác còn lại không làm mất raw evidence của run mới, nhưng giới hạn
mức diễn giải: kết quả phải được gọi là discovery/controlled protocol, chưa
phải validation production E2E của proposal.

### 3.5. Cách xử lý context dài trên T4

Lần thử GPU đầu tiên bị OOM trong prefill eager/math attention. Điều tra cho
thấy T4 có compute capability `7.5`, trong khi Flash SDP của torch cu124 chỉ
có kernel native phù hợp cho `sm80`/`sm90`; ép Flash trên T4 báo
`RuntimeError: No available kernel`. SDPA fallback sang math attention tạo
ma trận `L x L` với prompt GovReport dài, không phù hợp VRAM 15 GiB.

Đã thêm và kiểm thử hồi quy:

1. Prefill prompt theo chunk 512 token.
2. Dùng bottom-right causal mask để mỗi chunk chỉ nhìn thấy prefix quá khứ
   đúng với causal attention.
3. Dùng prefill chunking cho target, draft và verifier.
4. Chỉ bật attention output ở các bước decode incremental cần trace; prefill
   không materialize full attention map.

Đây là thay đổi bảo đảm khả năng đo, không thay đổi định nghĩa các hypothesis.
Sau thay đổi, run một GovReport dài 12.377 token và run chính 25 document
không còn OOM.

## 4. Runtime và máy thực hiện

GPU experiment được chạy đúng bằng interpreter ngoài venv, với dạng:

```bash
env -u VIRTUAL_ENV -u FAST_INFER_VENV -u FAST_INFER_PYTHON \
  /home/tuantb/miniconda3/bin/python3 ...
```

Các biến offline dùng trong run là `HF_HUB_OFFLINE=1` và
`TRANSFORMERS_OFFLINE=1`; GPU là `cuda:0`, dtype là `float16`.

| Thành phần | Thông tin đo được |
|---|---|
| Host | `teslaT4`, Ubuntu 20.04.6 LTS, kernel `5.4.0-216-generic`, x86_64 |
| CPU | Intel Xeon Silver 4210R @ 2.40 GHz, 20 logical CPUs |
| RAM | 125 GiB tổng, khoảng 103 GiB available tại thời điểm audit |
| GPU | NVIDIA Tesla T4 / TU104GL, 1 GPU |
| VRAM | 15,360 MiB theo `nvidia-smi`; torch thấy khoảng 14,917.7 MiB |
| Driver/CUDA | NVIDIA driver `550.163.01`, CUDA runtime/toolkit `12.4` |
| Compute capability | `7.5` (`sm75`) |
| Python dùng cho GPU | `/home/tuantb/miniconda3/bin/python3`, Python 3.13.9 |
| PyTorch | `2.6.0+cu124` |
| Transformers | `4.57.6` |
| Thư viện phân tích | NumPy 2.3.4, scikit-learn 1.8.0, safetensors 0.7.0, Pillow/Matplotlib 3.10.7 |

`.venv` của repository có Python 3.12.13, torch `2.11.0+cu130` và
Transformers `5.12.1`; môi trường đó được dùng cho test/evaluator CPU, **không
dùng cho thực nghiệm GPU**. Sandbox thông thường không expose thiết bị
NVIDIA, nhưng kiểm tra trực tiếp host bằng interpreter miniconda xác nhận
`torch.cuda.is_available()=True`, một Tesla T4 và không có process GPU khác
đang chạy ở lần audit cuối.

## 5. Các run đã thực hiện

### 5.1. Synthetic evaluator

Fixture gồm 12 target document và 120 controlled rows. Run sinh đủ JSONL,
metrics JSON/CSV, biểu đồ PNG và Markdown report. Các test synthetic giúp
kiểm tra tính đúng của JS, bootstrap, accepted prefix, horizon, predictor và
policy replay. Kết quả synthetic không được dùng để kết luận về Qwen3.

### 5.2. CPU local-only discovery

Smoke Qwen3-4B/0.6B trên CPU xác nhận loader, target trace, draft proposal,
verifier và timing block chạy end-to-end trên một fixture document.

Run CNN/DailyMail 10 mẫu có 80 target token steps và 20 controlled proposals
cho kết quả phụ trợ:

- H1 `FAIL`: persistence excess `0.003060`, CI 95% khoảng
  `[0.001755, 0.004421]`, thấp hơn gate `0.02`.
- H2 `FAIL`: rejection rate nhóm drift cao `0.20`, nhóm thấp `0.40`, delta
  `-0.20`.
- H3/H4/H5 `UNAVAILABLE` do thiếu coverage/class/timing phù hợp.

CPU run chỉ dùng để phát hiện lỗi pipeline và cung cấp regime phụ; không thay
thế run T4 cuối.

### 5.3. GPU diagnostic trước khi sửa prefill

Run 100 GovReport đầu tiên và một run SDPA một mẫu đều OOM ở prompt đầu tiên.
Prompt được đo dài 12.377 token; các allocation lỗi lên tới khoảng 9–19 GiB
trên GPU 14.57 GiB khả dụng. Kiểm tra kernel xác định nguyên nhân là T4
`sm75` không có Flash SDP native trong stack torch cu124, khiến prefill dài
chạy qua math attention.

Sau khi thêm chunked prefill, một run GovReport dài (`qwen3-4b-gov1-gpu-
chunked-20260829`) đạt `document_ok`, xác nhận sửa hạ tầng không còn OOM.
GPU smoke target/draft/verifier một mẫu cũng đo được timing verifier theo
block; smoke này chưa đủ để quyết định hypothesis.

### 5.4. Run GPU discovery chính

Run cuối dùng 25 GovReport, target Qwen3-4B và draft/verifier Qwen3-0.6B.
Target trace dùng 16 output tokens/document. Speculative phase dùng tối đa
`k=4`, hai start position/document; vì vậy có 25 target rows và 50
speculative rows. Context sau tokenizer nằm trong khoảng 2.609–22.630 token.

Lịch sử cần ghi rõ: lệnh `phase all` ban đầu đã hoàn thành target nhưng
speculative phase bản cũ OOM vì draft/verifier lúc đó chưa được chunk. Sau
đó code được sửa, speculative trace được chạy lại độc lập với cùng target
traces bằng `prefill_chunk_size=512`, rồi analyzer chạy lại. Artifact cuối
dưới đây chỉ dùng raw rows sau lần rerun thành công.

### 5.5. Run mở rộng ngày 30/08/2026

Run `qwen3-4b-gov100-gpu-protocol-20260830` dùng Python hệ thống ngoài venv,
Qwen3-4B target, Qwen3-0.6B draft, FP16 và prefill chunk 512:

- GovReport: 100 requested, 99 target `ok`, 1 OOM; 99 draft-only controlled
  proposal ở `start=1`, `kmax=8`; dùng cho H2/H3/H5.
- Timing: 12 IDs test theo document split, 11 `ok`, 1 OOM; sau khi lọc rows
  không phủ đủ `k=8`, H4/H5 có 10 rows timing hợp lệ.
- CNN/DailyMail: 100/100 target `ok`, 100 controlled proposals ở `start=1`,
  `kmax=8`; timing subset có 12/12 rows hợp lệ, dùng cho cross-regime H2–H5.
- Hazard model: first-rejection risk set theo vị trí 1..8; logistic drift
  coefficient điều chỉnh relative position; 2.000 document-bootstrap
  resamples, seed 42. Đây là association test, không phải survival causality.
- Test/evaluator sau mở rộng: 50 tests pass; `compileall` pass.

Việc chọn `start=1` là bắt buộc để `drift_at_start` có nghĩa; việc chọn timing
theo IDs test là để H5 không dùng test documents cho threshold/predictor fit.
Các lần chạy bị dừng giữa chừng do chi phí verifier quá lớn không được tính là
artifact hoàn chỉnh; chỉ file cuối cùng có manifest và metrics được dùng.

### 5.6. Phụ lục toàn bộ run directory đã tạo

Audit thư mục `results/` hiện có 23 run directory. E0 và hai run multi-start
model-backed được thêm ở ngày 30/08; các run trùng vai trò là các
lần audit/re-run để kiểm tra tính ổn định của evaluator; không được cộng dồn
thành số mẫu độc lập.

| Run ID hoặc nhóm run ID | Mục đích và kết quả | Vai trò evidence |
|---|---|---|
| `synthetic-20260829`, `synthetic-20260829-v2`, `synthetic-20260829-audit`, `synthetic-20260829-verified`, `synthetic-20260829-final`, `synthetic-20260829-final-v2`, `synthetic-20260829-blocked-audit` | coverage=12 target documents + 120 controlled rows; chạy fixture synthetic, kiểm tra metric, schema, plots, report và các lần audit evaluator. | Chỉ validation pipeline, không phải model evidence. |
| `qwen3-local-smoke-20260829`, `qwen3-local-smoke-audit`, `qwen3-local-smoke-final`, `qwen3-local-smoke-verified` | Thử target Qwen3-1.7B/draft Qwen3-0.6B local-only; Qwen3-1.7B không có snapshot nên H1–H5 `UNAVAILABLE`. | Blocker/cache audit, không dùng kết luận. |
| `qwen3-4b-06b-actual-smoke-20260829` | 1 document CPU, Qwen3-4B target + Qwen3-0.6B draft/verifier, xác nhận controlled timing chạy được. | Pipeline smoke, không đủ coverage. |
| `qwen3-4b-cnn10-target-20260829` | 10 CNN/DailyMail documents, 80 target steps, 20 proposals; H1/H2 `FAIL`, H3–H5 `UNAVAILABLE`. | CPU discovery phụ trợ. |
| `qwen3-4b-gov100-gpu-20260829` | Thử GovReport GPU trước khi chunked prefill; OOM ở context dài. | Runtime diagnostic, loại khỏi evidence. |
| `qwen3-4b-gov1-gpu-sdpa-20260829` | Thử SDPA trên 1 GovReport; vẫn OOM do fallback math attention trên T4 `sm75`. | Runtime diagnostic, loại khỏi evidence. |
| `qwen3-4b-gov1-gpu-chunked-20260829` | 1 GovReport sau chunked causal prefill; chạy được, không OOM. | Runtime smoke, không đủ coverage. |
| `qwen3-4b-06b-gpu-smoke-20260829` | 1 fixture document GPU, target/draft/verifier và timing theo block. | GPU pipeline smoke, không đủ coverage. |
| `qwen3-4b-gov25-gpu-all-20260829` | 25 GovReport, 25 target rows, 50 final speculative rows, timing đầy đủ; H1/H2/H4 `FAIL`, H3/H5 `UNAVAILABLE`. | Evidence GPU discovery lịch sử. |
| `qwen3-4b-gov100-gpu-protocol-20260830` | 100 GovReport target (99 ok), 99 full acceptance rows, 10 timing rows phủ `k=8`; H1/H2/H3/H4 `FAIL`, H5 `UNAVAILABLE`. | **Artifact chính, evidence mở rộng GovReport.** |
| `qwen3-4b-cnn100-gpu-protocol-20260830` | 100/100 CNN/DailyMail target, 100 full acceptance rows, 12 timing rows phủ `k=8`; H1/H3/H4 `FAIL`, H2 `PASS`, H5 `UNAVAILABLE`. | **Cross-regime controlled evidence cho H1–H5**, không phải production serving. |
| `e0-position-relocation-qwen3-4b-20260830` | Ba fixture source cùng evidence ở đầu/giữa/cuối; 3/3 target trace `ok`; raw/no-sink position mass được tổng hợp riêng. | **E0 diagnostic**, không phải H1–H5 sample. |
| `qwen3-4b-gov100-multistart-20260830` | 99 documents, 396 draft-only proposals tại start `1/6/11/16`; H2 `FAIL`, H3 gain `-0,0053`, H4 `UNAVAILABLE`, H5 `INCONCLUSIVE`. | Multi-start acceptance/history sensitivity; không có timing. |
| `qwen3-4b-cnn100-multistart-20260830` | 100 documents, 400 draft-only proposals tại start `1/6/11/16`; H2 `PASS`, H3 gain `+0,0086`, H4 `UNAVAILABLE`, H5 `INCONCLUSIVE`. | Multi-start cross-regime acceptance/history sensitivity; không có timing. |

Các run OOM không bị xóa để giữ lịch sử chẩn đoán; chúng không được đưa vào
metrics cuối và không được tính như mẫu thất bại của hypothesis.

## 6. Lệnh tái lập chính

### 6.1. Target và orchestrator

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MPLCONFIGDIR=/tmp/groundsync-mpl \
env -u VIRTUAL_ENV -u FAST_INFER_VENV -u FAST_INFER_PYTHON \
/home/tuantb/miniconda3/bin/python3 -m src.analyze.groundsync.run_experiment \
  --phase all \
  --run-id qwen3-4b-gov25-gpu-all-20260829 \
  --model /home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --draft-model /home/tuantb/models/Qwen3-0.6B \
  --input data/representative_100/govreport_representative.jsonl \
  --max-samples 25 --max-new-tokens 16 --max-k 4 --max-starts 4 \
  --chunk-size 128 --skip-source-tokens 8 --prefill-chunk-size 512 \
  --device cuda:0 --dtype float16 --horizon-threshold 0.2 \
  --max-horizon 16
```

### 6.2. Rerun speculative có verifier timing

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
MPLCONFIGDIR=/tmp/groundsync-mpl \
env -u VIRTUAL_ENV -u FAST_INFER_VENV -u FAST_INFER_PYTHON \
/home/tuantb/miniconda3/bin/python3 -m src.analyze.groundsync.trace_speculative \
  --draft-model /home/tuantb/models/Qwen3-0.6B \
  --verification-model /home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --input data/representative_100/govreport_representative.jsonl \
  --target-traces src/analyze/groundsync/results/qwen3-4b-gov25-gpu-all-20260829/target_traces.jsonl \
  --output src/analyze/groundsync/results/qwen3-4b-gov25-gpu-all-20260829/speculative_traces.jsonl \
  --max-samples 25 --max-k 4 --max-starts 2 --stride 1 \
  --prefill-chunk-size 512 --device cuda:0 --dtype float16
```

### 6.3. Phân tích lại artifact

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
MPLCONFIGDIR=/tmp/groundsync-mpl \
env -u VIRTUAL_ENV -u FAST_INFER_VENV -u FAST_INFER_PYTHON \
/home/tuantb/miniconda3/bin/python3 -m src.analyze.groundsync.run_experiment \
  --phase analyze \
  --run-dir src/analyze/groundsync/results/qwen3-4b-gov25-gpu-all-20260829 \
  --input data/representative_100/govreport_representative.jsonl \
  --max-k 4 --max-horizon 16 --horizon-threshold 0.2 --threshold 0.2
```

### 6.4. Protocol mở rộng 30/08/2026

Target GovReport và CNN/DailyMail chạy bằng Python hệ thống ngoài venv với
`--max-samples 100 --max-new-tokens 32 --chunk-size 128
--sensitivity-chunk-sizes 64,128,256 --sink-sizes 4,8,16
--prefill-chunk-size 512 --device cuda:0 --dtype float16`. GovReport target
ghi tại `qwen3-4b-gov100-gpu-protocol-20260830/`.

Acceptance full dùng draft-only ở `--max-k 8 --max-starts 1 --start-offset 1`;
đây là acceptance data cho H2/H3/H5, không dùng để claim speed. Timing dùng
cùng runner với `--verification-model`, `--max-k 8`, `--max-starts 1`,
`--start-offset 1` và `--sample-ids` gồm 12 document test; output là
`speculative_timing_traces.jsonl`. Analyzer đọc full acceptance cho H2/H3/H5
và timing file riêng cho H4/H5 policy. Cùng protocol này đã chạy trên cả
`qwen3-4b-gov100-gpu-protocol-20260830` và
`qwen3-4b-cnn100-gpu-protocol-20260830`; chỉ khác file input và tập document.

## 7. Kết quả định lượng của run T4 chính

Các mục 7.1–7.6 bên dưới lưu kết quả lịch sử của run 25-document ngày
29/08/2026. Mục 7.7 là kết quả mở rộng mới nhất và được dùng cho bảng kết
luận ở đầu báo cáo.

### 7.1. Audit raw artifact

- Target: `25/25` dòng `status=ok`, không có lỗi.
- Speculative: `50/50` dòng `status=ok`, không có lỗi.
- Input target tokens: min `2.609`, max `22.630`, mean `10.147,5`.
- Target output: 16 token ở mỗi document, tổng `400` token steps.
- Tất cả 50 speculative rows có draft timing và verifier timing theo `k`.
- Accepted length: `{0: 43, 1: 3, 2: 1, 3: 1, 4: 2}`.
- Tỷ lệ row bị rejection: `96%`.
- Grounding horizon khác `null`: `0` row.
- Drift tại start hữu dụng nằm khoảng `0.001608–0.012186`.

### 7.2. H1 — persistence của source-state

Biến thể chính là `nosink`; source attention được gom chunk sau khi bỏ 8
source token đầu. Kết quả:

- Adjacent similarity: `0.9899650`.
- Shuffle-null adjacent similarity: `0.9711250`.
- Persistence excess: `0.0188400`.
- Mean/median meaningful segment length: `16/16`; mỗi document trong run
  có một segment dài 16. Điều này đạt engineering check `L_G >= 4` trong
  GovReport discovery, nhưng chưa đủ gate “GovReport và ít nhất một regime
  khác” vì chưa có cross-regime GPU run.
- Lag similarity: lag 1 `0.9899650`, lag 2 `0.9834300`, lag 4 `0.9732441`,
  lag 8 `0.9624512`; lag 16/32 không có đủ cặp vì trace chỉ dài 16 token.
- Bootstrap theo 25 document, 2.000 resamples, CI 95%:
  `[0.0159799, 0.0218817]`.
- Gate: cận dưới CI `>= 0.02`.
- Quyết định: **`FAIL`** vì `0.0159799 < 0.02`.

Kết quả có một tín hiệu dương so với null, nhưng độ lớn không đạt gate. Đây
không phải tuyên bố rằng persistence bằng zero; nó chỉ không đủ mạnh theo
tiêu chí đã đặt trước. Raw control có adjacent similarity `0.9898447`, gần
với `nosink` `0.9899650`, nên kết quả này không thể được giải thích đơn giản
bởi việc bỏ sink token trong run hiện tại.

### 7.3. H2 — drift và speculative rejection

Report dùng 25 row có `drift_at_start` hữu dụng, chia tại median drift
`0.00506815`:

- Nhóm drift thấp: 13 row, rejection rate `1.0000`.
- Nhóm drift cao: 12 row, rejection rate `0.8333`.
- Delta high minus low: `-0.1667`.
- Quyết định: **`FAIL`**.

Trên dữ liệu này, drift cao không dự báo rejection theo hướng claim; chiều
quan sát được còn ngược lại. 50 speculative rows vẫn được giữ trong raw
artifact và dùng cho H4; H2 chỉ loại các row không thể tính drift tại
position đầu tiên.

### 7.4. H3 — incremental predictive value

Có 25 predictor rows sau khi join target feature với controlled acceptance.
Document split theo thứ tự id tạo:

- Train: 15 document.
- Dev: 5 document.
- Test: 5 document.

Baseline gồm target entropy, position fraction, draft confidence mean, recent
acceptance, sentence boundary, copyability và max-k. Full model thêm
`drift_at_start` và `horizon_normalized`.

Test labels có positive rate `1.0`, tức chỉ có một class. Vì vậy AUROC/AUPRC
không hợp lệ và không có AUROC gain để so sánh. Quyết định: **`UNAVAILABLE`**.

### 7.5. H4 — utility và timing

Có 50 row đủ timing. Với policy fixed, `k=4`; với policy oracle, `k` được
clip từ horizon quan sát, fallback về 1 khi horizon null. Đây là replay
controlled trên proposal/canonical continuation, chưa phải benchmark EAGLE
hoặc vLLM production serving.

| Policy | Mean accepted draft token | Mean committed token | Mean cost (ms) | Committed token/ms |
|---|---:|---:|---:|---:|
| Fixed | 0,32 | 1,28 | 3540,03 | 0,00036158 |
| Oracle | 0,14 | 1,00 | 3466,16 | 0,00028850 |

Speed gain oracle so với fixed:

```text
0.00028850 / 0.00036158 - 1 = -0.20210
```

Gate là `>= 0.08`; kết quả **`FAIL`**. Acceptance-only gain cũng âm
(`-0.21875`), nhưng kết luận H4 dựa trên timing đã đo, không dựa riêng vào
acceptance. Timing basis được ghi trong artifact là
`measured_cached_target_check`: draft được đo và target verifier được đo
trên cached target check theo từng block; chưa đo một server speculative tối
ưu với batching/kernel production.

### 7.6. H5 — online horizon predictor

Có 375 horizon rows (`25 document x 15 position`). Horizon được định nghĩa là
offset đầu tiên mà JS drift từ state hiện tại vượt `0.2`, giới hạn tối đa 16;
nhãn dương là horizon `>=2`.

Trong run này không có grounding horizon khác `null`, nên không tạo được
biến thiên lớp để fit/đánh giá predictor. Kết quả **`UNAVAILABLE`**, không
phải `FAIL` về khả năng dự đoán: trước hết cần một nhãn có thông tin.

### 7.7. Kết quả mở rộng GovReport 99/100 và CNN/DailyMail 100/100

Các số liệu dưới đây là kết quả mới nhất, được sinh lại sau khi bổ sung
calibration, position-adjusted hazard, document bootstrap, negative controls,
adaptive policies và train/dev threshold selection. Artifact máy đọc là
[`metrics.json` GovReport](results/qwen3-4b-gov100-gpu-protocol-20260830/metrics.json)
và [`metrics.json` CNN/DailyMail](results/qwen3-4b-cnn100-gpu-protocol-20260830/metrics.json).

#### H1/E0 và cross-regime

GovReport có 99 target trace hợp lệ trên 100 request; CNN/DailyMail có 100/100.
GovReport dùng 32 output tokens nên đo được lag 1/2/4/8/16/32.

| Estimator/variant | GovReport persistence excess | CI 95% lower | Quyết định gate 0,02 |
|---|---:|---:|---:|
| no-sink, chunk 128 | 0,023892 | 0,021996 | PASS riêng estimator |
| raw, chunk 128 | 0,023440 | 0,021540 | PASS riêng estimator |
| calibrated, 32-bin positional prior | 0,019835 | 0,018257 | FAIL |
| raw, chunk 64 | 0,035414 | 0,032807 | PASS riêng estimator |
| raw, chunk 256 | 0,015895 | 0,014328 | FAIL |
| no-sink sink 4/8/16, chunk 128 | 0,023823 / 0,023892 / 0,024399 | 0,021942 / 0,021996 / 0,022459 | PASS riêng estimator |
| CNN/DailyMail no-sink, chunk 128 | 0,012259 | 0,010580 | FAIL |

GovReport no-sink có adjacent similarity `0,989052`, shuffle null `0,965160`,
excess `0,023892`, CI thấp `0,021996`; mean/median `L_G` đều `32`. Tuy nhiên
calibrated estimator chỉ có CI thấp `0,018257`, còn CNN/DailyMail có CI thấp
`0,010580`. Theo gate robust, H1 **`FAIL`**: có persistence dương ở một số
variant nhưng không đủ ổn định qua calibration và regime.

#### E0 — position relocation và attention confounder

Để kiểm tra positional bias độc lập với dataset representative, đã chạy một
fixture Qwen3-4B gồm ba source có cùng câu evidence duy nhất, lần lượt đặt ở
đầu, giữa và cuối tài liệu. Prompt, target, decoding, `max_new_tokens=8` và
GPU T4 giữ nguyên; source dài 419–420 token, chunk `16`, sink control bỏ 8
token đầu. Artifact là
[`e0_position_relocation.json`](results/e0-position-relocation-qwen3-4b-20260830/e0_position_relocation.json)
và raw trace nằm trong
[`target_traces.jsonl`](results/e0-position-relocation-qwen3-4b-20260830/target_traces.jsonl).

| Variant | Evidence ở đầu | Evidence ở giữa | Evidence ở cuối | Range max–min | Max/min |
|---|---:|---:|---:|---:|---:|
| Raw chunk 16 | 0,5029 | 0,1170 | 0,1929 | 0,3859 | 4,30x |
| No-sink 8, chunk 16 | 0,5185 | 0,2297 | 0,2296 | 0,2890 | 2,26x |

Kết quả cho thấy attention mass thay đổi mạnh theo vị trí của cùng evidence;
no-sink làm giảm nhưng không loại bỏ confounder. Đây là lý do raw attention
không được dùng như ground-truth attribution, và H1/H3 vẫn phải báo cáo
calibration, estimator sensitivity và position controls. E0 là diagnostic
control, không có PASS/FAIL gate riêng.

#### H2 — hazard và rejection

Trên 99 GovReport proposals tại `start=1`, drift median là `0,005166`; nhóm
thấp có 50 rows và rejection `0,8800`, nhóm cao có 49 rows và rejection
`0,8571`, delta high-minus-low `-0,022857` nên H2 **`FAIL`**. Hazard first
rejection theo relative draft position (risk set, events, hazard) là:

| Relative position | At risk | Events | Hazard |
|---:|---:|---:|---:|
| 1 | 99 | 67 | 0,6768 |
| 2 | 32 | 3 | 0,0938 |
| 3 | 29 | 2 | 0,0690 |
| 4 | 27 | 8 | 0,2963 |
| 5 | 19 | 1 | 0,0526 |
| 6 | 18 | 4 | 0,2222 |
| 7 | 14 | 1 | 0,0714 |
| 8 | 13 | 0 | 0,0000 |

Hazard model chính điều chỉnh theo relative position bằng logistic regression
trên risk set; drift được z-score trong fit và bootstrap resample nguyên
document. Kết quả so sánh hai regime:

| Regime | Documents | Risk-set rows | Drift coefficient | Odds ratio / SD | CI 95% document bootstrap | H2 |
|---|---:|---:|---:|---:|---:|---:|
| GovReport | 99 | 251 | -0,065733 | 0,9364 | [-0,066417; -0,053486] | **FAIL** |
| CNN/DailyMail | 100 | 461 | 0,000543 | 1,0005 | [0,018109; 0,030099] | **PASS** |

CNN/DailyMail có median-split delta `+0,020000`, phù hợp với chiều của
hazard coefficient; tuy nhiên độ lớn coefficient point rất nhỏ, và khoảng
bootstrap lệch dương. Vì H2 cho hai chiều khác nhau giữa hai regime, không thể
nâng kết quả CNN thành claim tổng quát. Kiểm định này là association sau khi
điều chỉnh vị trí, không phải bằng chứng nhân quả hay survival causality.

#### H3 — incremental predictor

Có 99 predictor rows, split theo document thành 59 train, 19 dev, 21 test;
nhãn chính là `first_token_rejected`. Baseline gồm entropy, vị trí, draft
confidence, recent acceptance, sentence boundary, copyability và `max_k`;
full thêm source concentration, drift, drift-missing, lag drift/history và
horizon feature. Ở run một-start, lag-history missing ở start=1; run
multi-start bên dưới kiểm tra nó tại các start muộn hơn. Kết quả test:

| Model | AUROC | AUPRC | Log-loss | Brier |
|---|---:|---:|---:|---:|
| Baseline | 1,0000 | 1,0000 | 0,2846 | 0,0796 |
| Full no-sink | 1,0000 | 1,0000 | 0,2253 | 0,0567 |
| Position-only control | 0,5000 | 0,8095 | 0,5794 | 0,1939 |
| Temporal shift +10 | 0,3088 | 0,7571 | 0,6191 | 0,2134 |
| Temporal shift +20 | 0,3971 | 0,7926 | 0,7059 | 0,2519 |

Full cải thiện log-loss/Brier nhưng AUROC gain `0,0000`, dưới gate `0,02`;
calibrated variant cũng có gain `0,0000`. H3 **`FAIL`** theo metric chính,
không phải unavailable vì test đã có đủ hai lớp.

Cross-regime CNN/DailyMail có 100 rows, split `60/20/20` theo document. No-sink
baseline đạt AUROC `0,9067`, AUPRC `0,8254`, log-loss `0,3092`, Brier `0,0909`;
full đạt lần lượt `0,8800`, `0,8052`, `0,3042`, `0,0903`, nên AUROC gain
`-0,0267` và H3 primary **`FAIL`**. Calibrated sensitivity của CNN có AUROC
gain `+0,0267` và được đánh dấu PASS ở estimator đó, nhưng không thay thế
estimator no-sink đã khóa làm primary. Điều này cho thấy kết quả H3 nhạy với
estimator, không đủ để xác nhận claim robust.

Các control temporal shift và shuffle được lưu trong `metrics.json`. Vì run
hiện tại chỉ có một proposal tại `start=1` cho mỗi document, shuffle feature
“trong cùng document” không làm thay đổi giá trị; đây là giới hạn thiết kế,
không được diễn giải là negative control mạnh.

#### H4 — policy và timing

Timing dùng 12 test IDs, trong đó 11 rows chạy được; sau khi yêu cầu arrays
phủ đủ `k=8`, còn 10 rows hợp lệ. Timing basis là
`measured_cached_target_check`, không phải EAGLE/vLLM E2E.

| Policy | Mean committed | Mean cost (ms) | Committed token/ms |
|---|---:|---:|---:|
| Fixed k=2 | 1,2000 | 3.723,6 | 0,0003223 |
| Fixed k=4 | 1,6000 | 3.891,6 | 0,0004111 |
| Fixed k=8 | 2,0000 | 4.203,6 | 0,0004758 |
| Adaptive entropy | 1,6000 | 4.017,5 | 0,0003983 |
| Adaptive history | 2,0000 | 4.203,6 | 0,0004758 |
| Grounding-horizon oracle | 1,0000 | 3.775,2 | 0,0002649 |
| True-cost hindsight oracle | 2,0000 | 3.750,6 | 0,0005333 |

Grounding oracle so với fixed k=8 có speed gain `-0,4433` và acceptance-only
gain `-0,5000`; H4 **`FAIL`**. True-cost oracle có headroom khoảng `+12,1%`,
nhưng đó là hindsight upper bound, không chứng minh horizon quan sát có thể
đạt được.

Cross-regime CNN/DailyMail có 12/12 timing rows phủ đủ `k=8`. Fixed `k=8`
đạt `0,0065909` committed token/ms, grounding oracle đạt `0,0037003`, speed
gain `-0,4386`; H4 **`FAIL`**. True-cost hindsight oracle đạt `0,0104898`
token/ms, tức headroom khoảng `+59,1%` so với fixed `k=8`, nhưng cũng không
phải policy có thể triển khai online. Bảng policy đầy đủ (fixed `k=2,4,8`,
adaptive entropy/history và true-cost oracle) nằm trong hai `metrics.json`.

#### H5 — online horizon

Ngưỡng yêu cầu ban đầu là `0,2`, nhưng train/dev selection trên 78 documents
chọn `0,05` (positive rate calibration `0,5769`), không nhìn test. Predictor
full có 21 test rows: AUROC `0,3500`, AUPRC `0,9536`, log-loss `0,5181`, Brier
`0,1678`. Timing policy test có 10 rows; predicted policy đạt
`0,0002969` committed token/ms, fixed k=4 đạt `0,0004111`, còn grounding
oracle đạt `0,0003873`. Vì oracle chậm hơn fixed, oracle-gain recovery được
đặt `None` thay vì chia hai số âm; H5 **`UNAVAILABLE`**, không phải PASS.

Cross-regime CNN/DailyMail chọn threshold `0,03` trên 80 train/dev documents;
predictor có 20 test rows với AUROC `0,6875`, AUPRC `0,4333`, log-loss
`0,6554`, Brier `0,2324`. Trên 12 timing rows, fixed `k=4` đạt `0,0083522`,
predicted horizon đạt `0,0049563`, còn oracle horizon đạt `0,0049648`
committed token/ms. Oracle cũng chậm hơn fixed nên H5 là **`UNAVAILABLE`**.

| Regime | Threshold train/dev | Predictor test | Timing test | Fixed k=4 | Predicted | Oracle horizon | Recovery | H5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GovReport | 0,05 | 21 | 10 | 0,0004111 | 0,0002969 | 0,0003873 | None | **UNAVAILABLE** |
| CNN/DailyMail | 0,03 | 20 | 12 | 0,0083522 | 0,0049563 | 0,0049648 | None | **UNAVAILABLE** |

Việc chọn threshold chỉ dùng train/dev; test chỉ dùng để báo cáo metric và
policy. Cả hai predictor có metric phân loại hữu dụng ở mức nhất định, nhưng
điều đó không đủ cho H5: protocol yêu cầu phải có oracle headroom dương trước
khi tính recovery `>=0,50`.

#### 7.8. Kiểm tra multi-start và lag-history

Để tránh phụ thuộc vào một start duy nhất, đã chạy thêm draft-only controlled
traces trên cùng 100 target traces ở bốn start positions mỗi document:
`1, 6, 11, 16` (`stride=5`, `max_k=8`). Run này không có verifier timing nên
chỉ dùng cho H2/H3/H5 feature/history, không dùng thay cho H4 timing.

| Regime | Target docs | Proposal rows | Start positions | H2 | H3 AUROC gain | H4 | H5 |
|---|---:|---:|---|---:|---:|---|---|
| GovReport | 99 | 396 | 1/6/11/16 | **FAIL** | -0,0053 | UNAVAILABLE | INCONCLUSIVE |
| CNN/DailyMail | 100 | 400 | 1/6/11/16 | **PASS** | +0,0086 | UNAVAILABLE | INCONCLUSIVE |

GovReport multi-start có hazard drift coefficient CI
`[-0,0901; -0,0841]`; CNN/DailyMail có CI `[0,0061; 0,0106]`. H3 no-sink
đều dưới gate `+0,02` dù đã có lag-drift và within-document acceptance
history. H5 có thể fit predictor nhưng không có timing rows, nên được đặt
`INCONCLUSIVE` thay vì suy diễn utility. Artifact raw và metrics tương ứng:
[`GovReport multi-start`](results/qwen3-4b-gov100-multistart-20260830/)
và [`CNN/DailyMail multi-start`](results/qwen3-4b-cnn100-multistart-20260830/).

## 8. Artifact và kiểm tra tái lập

Artifact discovery cũ nằm trong
[`results/qwen3-4b-gov25-gpu-all-20260829/`](results/qwen3-4b-gov25-gpu-all-20260829/):

- [`target_traces.jsonl`](results/qwen3-4b-gov25-gpu-all-20260829/target_traces.jsonl)
- [`speculative_traces.jsonl`](results/qwen3-4b-gov25-gpu-all-20260829/speculative_traces.jsonl)
- [`metrics.json`](results/qwen3-4b-gov25-gpu-all-20260829/metrics.json)
- [`metrics.csv`](results/qwen3-4b-gov25-gpu-all-20260829/metrics.csv)
- [`hypothesis_report.md`](results/qwen3-4b-gov25-gpu-all-20260829/hypothesis_report.md)
- [`run_manifest.json`](results/qwen3-4b-gov25-gpu-all-20260829/run_manifest.json)
- [`target_manifest.json`](results/qwen3-4b-gov25-gpu-all-20260829/target_manifest.json)
- [`events.jsonl`](results/qwen3-4b-gov25-gpu-all-20260829/events.jsonl)
- `persistence.png`, `drift_rejection.png`, `policy_utility.png`,
  `horizon_labels.png`

Artifact mở rộng dùng cho kết luận hiện tại nằm trong
[`results/qwen3-4b-gov100-gpu-protocol-20260830/`](results/qwen3-4b-gov100-gpu-protocol-20260830/), gồm:

- [`target_traces.jsonl`](results/qwen3-4b-gov100-gpu-protocol-20260830/target_traces.jsonl)
- [`speculative_traces.jsonl`](results/qwen3-4b-gov100-gpu-protocol-20260830/speculative_traces.jsonl)
- [`speculative_timing_traces.jsonl`](results/qwen3-4b-gov100-gpu-protocol-20260830/speculative_timing_traces.jsonl)
- [`metrics.json`](results/qwen3-4b-gov100-gpu-protocol-20260830/metrics.json)
- [`metrics.csv`](results/qwen3-4b-gov100-gpu-protocol-20260830/metrics.csv)
- [`hypothesis_report.md`](results/qwen3-4b-gov100-gpu-protocol-20260830/hypothesis_report.md)
- [`target_manifest.json`](results/qwen3-4b-gov100-gpu-protocol-20260830/target_manifest.json)
- [`speculative_timing_traces.manifest.json`](results/qwen3-4b-gov100-gpu-protocol-20260830/speculative_timing_traces.manifest.json)
- [`speculative_traces.manifest.json`](results/qwen3-4b-gov100-gpu-protocol-20260830/speculative_traces.manifest.json)
- `run_manifest.json`, `events.jsonl`, các PNG summary.

Artifact cross-regime là
[`results/qwen3-4b-cnn100-gpu-protocol-20260830/`](results/qwen3-4b-cnn100-gpu-protocol-20260830/), với target trace 100/100.
Artifact này cũng gồm `speculative_traces.jsonl` (100/100 controlled
proposals), `speculative_timing_traces.jsonl` (12/12 timing rows), hai
manifest tương ứng, `metrics.json`, `metrics.csv`, `hypothesis_report.md` và
các PNG summary. Vì vậy CNN/DailyMail không còn chỉ là target-side evidence;
nó là cross-regime controlled evidence cho H1–H5, với giới hạn timing subset
nhỏ và không có serving E2E.

Artifact E0 position-relocation là
[`results/e0-position-relocation-qwen3-4b-20260830/`](results/e0-position-relocation-qwen3-4b-20260830/),
gồm raw `target_traces.jsonl`, `e0_position_relocation.json` và
`run_manifest.json`; cả ba case đầu/giữa/cuối đều `ok`.

Hai artifact multi-start draft-only là
[`qwen3-4b-gov100-multistart-20260830`](results/qwen3-4b-gov100-multistart-20260830/)
và [`qwen3-4b-cnn100-multistart-20260830`](results/qwen3-4b-cnn100-multistart-20260830/).
Mỗi manifest ghi rõ `max_starts=4`, `stride=5`, `start_offset=1`, timing basis
`draft_only_no_target_check` và số row lỗi nếu target trace không tồn tại.

Report machine-readable chính là `metrics.json`; file Markdown ngắn trong
run directory là summary được sinh tự động. File này là báo cáo diễn giải
toàn bộ quy trình và lịch sử xử lý lỗi.

Đối chiếu với artifact contract trong design: `target_manifest.json` và
`events.jsonl` có mặt trong run cuối; `run.log` và `run_metadata.json` không
có trong thư mục này. Các thông tin tương ứng về phase, seed, model path,
counts, elapsed target time, hardware và package versions đã được lấy từ
`run_manifest.json`, `target_manifest.json`, raw traces và audit host rồi ghi
trực tiếp trong báo cáo này. Vì vậy artifact cuối đủ raw evidence và metric,
nhưng thiếu log file riêng theo contract cũ.

Các kiểm tra code cuối cùng:

- Pytest GroundSync: **50 passed** trong vòng xác minh cuối.
- `compileall` cho package GroundSync: pass với runtime `.venv` và miniconda.
- `git diff --check`: pass.
- Audit artifact mở rộng: 99/100 GovReport target và 99/100 proposal `ok`,
  10 Gov timing rows phủ đủ `k=8`; 100/100 CNN target và proposal `ok`,
  12 CNN timing rows phủ đủ `k=8`; lỗi OOM được ghi thành row/manifest, không
  bị ẩn.

## 9. Giới hạn của kết luận

1. Run GovReport mới đạt 99/100 target do một context OOM; CNN/DailyMail đạt
   100/100 target và 100/100 controlled proposals. Cross-regime đã có H2–H5,
   nhưng timing chỉ là 12 rows và không đại diện production serving.
2. 32 output tokens/document và tối đa `k=8` đã mở rộng coverage; main timing
   run dùng một start/document, còn multi-start acceptance bổ sung mới dùng
   bốn vị trí `1/6/11/16`, chưa bao phủ toàn bộ vị trí decode và chưa có
   multi-start timing.
3. Attention source-state là proxy được quan sát từ model; chưa có human
   attribution hoặc causal intervention để chứng minh “grounding”.
4. T4 `sm75` không có native Flash SDP phù hợp với stack cu124; chunked
   prefill làm cho phép đo chạy được nhưng không biến nó thành production
   kernel benchmark.
5. H4 dùng controlled cached target verification trên 10 rows phủ đủ `k=8`.
   Kết quả có timing thật nhưng chưa phải throughput của EAGLE/vLLM/continuous
   batching.
6. H2 đã có logistic hazard coefficient điều chỉnh vị trí và 2.000 cluster
   bootstrap theo document; đây vẫn là mô hình association nhỏ, chưa phải
   survival model đầy đủ. H3/H5 vẫn là predictor nhỏ với test count hạn chế.
7. H5 threshold được chọn trên train/dev, nhưng oracle horizon chậm hơn fixed
   nên recovery không được coi là metric thành công.
8. E0 position-relocation mới có 3 fixture cases, không có CI hay nhiều seed;
   nó đủ phát hiện confounder lớn nhưng chưa phải ước lượng tổng quát về
   position bias.
9. H1 có persistence dương nhưng không ổn định qua calibration/cross-regime;
   việc fail gate không chứng minh mọi model hoặc dataset đều không có
   persistence.

## 10. Kết luận và bước tiếp theo

Với bằng chứng hiện có, không nên báo cáo GroundSync đã xác nhận đầy đủ.
Bằng chứng mạnh nhất hiện tại là:

- Pipeline target/speculative/timing chạy được trên T4 ngoài venv.
- H1 có persistence dương ở no-sink GovReport nhưng fail kiểm tra robust qua
  calibrated và cross-regime CNN/DailyMail.
- H2 không đi đúng hướng dự báo rejection.
- H4 không cho thấy lợi ích utility/tốc độ của oracle trong controlled run.
- H3 đã có dữ liệu hợp lệ nhưng grounding không tăng AUROC; H5 có predictor
  nhưng thiếu oracle headroom dương để kết luận utility.

Để chuyển từ controlled discovery sang kết luận tổng quát vẫn cần mở rộng E0
với nhiều anchor/seed, thêm nhiều start/output positions và timing multi-start
trên kích thước sample lớn hơn, cùng benchmark serving thực với implementation
speculative tối ưu nếu claim cuối vẫn là tokens/second.
