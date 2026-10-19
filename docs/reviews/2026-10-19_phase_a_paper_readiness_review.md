# Review khắt khe Phase A: từ động cơ, thực nghiệm đến claim và khả năng viết paper

Ngày đánh giá: **19/10/2026**. Phạm vi: Phase A context selection cho DFlash; đánh giá bằng chứng hiện có trong workspace và đối chiếu prior work gốc. Không đánh giá contribution joint A+B như một kết quả đã hoàn thành.

**Bổ sung sau đối chiếu kết quả cũ/mới:** xem [reconciliation](2026-10-19_phase_a_old_new_reconciliation.md). Đã tái tạo 570/570 dòng M1 cũ bằng mean-query top-B và xác nhận số speedup lịch sử 1,202×. Rerun đã đổi metric, query scope, baseline và workload; nó cũng có lỗi riêng. Các kết luận về max bên dưới áp dụng cho cấu hình accepted-prefix và protocol đã đo, không chứng minh regression của ý tưởng M1 gốc hoặc phủ nhận khả năng đạt 1,2× ở điều kiện cũ.

**Phán quyết: chưa sẵn sàng cho submission top-tier. Nếu nộp Phase A hiện tại như một phương pháp mới đã được xác nhận, khuyến nghị Reject, khoảng 2/6 trên thang minh họa do review này quy ước.** Đây là đánh giá hồ sơ nghiên cứu hiện có, không phải chấm một manuscript hoàn chỉnh hoặc dự đoán quyết định của một conference cụ thể. Độ tin cậy: cao với lỗi code/artifact đã tái hiện; vừa với novelty toàn lĩnh vực vì đối chiếu literature có chọn lọc.

Vấn đề quyết định là **chưa xác nhận được giá trị tăng thêm của selector đề xuất so với parent-query và prior work gần nhất**, cùng hai sai lệch runtime mới phát hiện. Có bằng chứng context reduction giúp prototype trong một pilot hẹp. Chưa có chuỗi bằng chứng đủ chặt để chuyển quan sát đó thành contribution phương pháp hoặc claim hiệu quả cho long-context summarization.

## 1. Đã kiểm tra gì và giới hạn của lượt review

Các nguồn chính đã đọc/đối chiếu:

- [Báo cáo Phase A hiện tại](../_INSIGHT/MASTER_PHASE_A_COMPREHENSIVE_REPORT.md), [insight](../_INSIGHT/insight_Phase_A.md), [attention layer](../_INSIGHT/LLM_attn_layer.md), [rerun 19/10](../_INSIGHT/phase_a_paper_rerun_20261019.md).
- [Protocol A1–A3](../../src/TrainingFree/plans/experiment_plan_A_13_10_26.md), [design](../../src/TrainingFree/docs/context_adaptive_dflash/design.md), [paper story ban đầu](../../src/TrainingFree/docs/context_adaptive_dflash/paper_story.md).
- [Động cơ parent-to-next-draft ngày 06/10](../experiments/2026-10-06_dflash_parent_attention_next_draft.md), [review 18/10](2026-10-18_insight_phase_ab_review.md), [nhật ký sửa logic](../_INSIGHT/phase_ab_logic_corrections_20261018.md).
- Collector, evaluator, runtime launcher, generator, config, controller, selection, full KV bank, code dựng hình và các test Phase A.
- [Coverage report](../../outputs/_PAPER_EXP/dflash_sota_selection_benchmark/evidence/coverage/report.json), metric từng document; [runtime report](../../outputs/_PAPER_EXP/dflash_sota_selection_benchmark/evidence/runtime/report.json) gồm 300 request và output IDs; parity diagnostic; provenance 47 hình.
- Các báo cáo P1/layer-selection cũ còn ở các thư mục output khác; pilot một prompt trước rerun.

Kiểm chứng độc lập trong lượt này:

1. Đối chiếu **8 hash** của code/report: khớp hết. Code evaluator, generator và code hình hiện tại khớp hash trong artifacts đã chạy.
2. Kiểm tra **94/94 file PNG/PDF** của 47 hình: hash khớp provenance.
3. Kiểm tra 300 cell runtime: không duplicate, 300 status `ok`; **tất cả đều sinh đúng 64 token và dừng do cap**, không có request natural EOS.
4. Ghép cặp output/counters giữa mean/max; tái hiện cấu hình pooling và tail-controller trên CPU.
5. Tính lại contrasts từ metric từng document và runtime request. CI bổ sung ở đây dùng paired document bootstrap, 10.000 resamples, seed 42. Đây là phân tích **post-hoc exploratory**, không thay thế kiểm định held-out và không có điều chỉnh multiple comparisons.
6. Chạy `test_phase_a_evidence.py` và `test_proposed_context_selectors.py`: **19 passed trong 3,86 giây**. Các test này không phát hiện sai binding pooling của launcher.

Kết quả kiểm chứng và số liệu máy đọc được lưu ở [audit JSON](2026-10-19_phase_a_paper_audit.json). Không sửa implementation, không chạy GPU mới, không truy cập server B200. Raw snapshot v2 của rerun 960 snapshot không có trong thư mục local được kiểm tra; chúng được khai báo nằm trong Modal Volume. Vì vậy lượt này kiểm toán collector/code, aggregates và per-document metrics, **chưa tái tính trực tiếp toàn bộ attention từ raw snapshot**. Hash provenance xác nhận tính nhất quán artifact, không tự chứng minh phép đo khoa học đúng.

## 2. Chuỗi lập luận hiện đang đứng ở đâu?

| Mắt xích trong story | Bằng chứng thực có | Đánh giá |
|---|---|---|
| Parent attention khác attention của drafter, nhất là context dài | Phép đo 06/10: 10 GovReport, 4 context, 4.326 transitions | Có động cơ nghiên cứu; khác phân phối chưa chứng minh selector gây giảm acceptance |
| Một working set nhỏ đủ cho DFlash | Coverage proxy và runtime reduced-context ngắn | Có tín hiệu pilot; chưa biết budget tối thiểu cho full summary |
| Nhiều query tốt hơn parent-query | Gain coverage nhỏ, phụ thuộc reference/chunk; runtime max chưa vượt parent rõ | Chưa xác nhận lợi ích phương pháp |
| Max tốt hơn mean | Offline mean hơi cao hơn tại một số cell; runtime mean gắn sai cấu hình | Chưa có phép so runtime hợp lệ |
| Ensemble/layer lựa chọn giải quyết vấn đề | Ensemble tăng một reference proxy nhưng giảm reference khác, tốn signal hơn | Có ablation; chưa có ưu thế thực dụng |
| Giảm context cải thiện proposal vì giữ đúng thông tin cần thiết | Chưa thấy kết quả A1/A2 same-prefix can thiệp đầy đủ cho selector mới | Thiếu cầu nối nhân quả |
| Phương pháp cải thiện latency thực tế | Pilot tốt hơn dense adapter trên output ngắn, cùng output với adapter | Có kết quả prototype hẹp; thiếu native baseline, full summary, held-out |
| Đây là contribution mới có ý nghĩa | Thiếu đối chiếu trực tiếp Vegas/SpecExtend trong kết quả | Novelty và significance chưa được thiết lập |

## 3. Các vấn đề chặn paper, xếp theo mức độ

### F1 — P0: hàng runtime “mean” đang chạy cấu hình max

Tại [launcher runtime](../../scripts/run_phase_a_paper_evaluation.py), dòng 127–130 khai báo nhãn `mean_L18_accepted` với selector `multi_query_mean_pooling`. Nhưng cả warmup và measured run tại dòng 148–156 đều không truyền `query_pooling`. [AdaptiveConfig](../../src/TrainingFree/context_adaptive/config.py), dòng 32, mặc định `query_pooling="max"`; [generator](../../src/TrainingFree/context_adaptive/generation.py), dòng 1221–1232, truyền đúng trường này xuống hàm tính score. Tên selector không tự chuyển toán tử sang mean.

Tái hiện CPU cho kết quả: cấu hình mang tên mean trả score **bằng explicit max và khác explicit mean**. Trong log hiện có, cả **60/60 cặp mean/max** khớp output và toàn bộ algorithmic counters đã kiểm tra, bỏ riêng peak memory. Điều này nhất quán với lỗi wiring; chênh TPS giữa hai hàng không phải bằng chứng so hai thuật toán khác nhau.

**Hệ quả:** không dùng hàng runtime mean để kết luận mean≈max, mean nhanh nhất hoặc mean/max đã được ablate. Coverage offline vẫn có code mean riêng đúng; lỗi này không tự làm vô hiệu các hàng coverage offline. Pilot một prompt cũ cũng có truyền `query_pooling="mean"` tường minh, nên không quy lỗi này cho mọi thực nghiệm lịch sử.

**Điều kiện xử lý:** nối pooling tường minh theo variant; lưu effective config trong từng request; kiểm tra launcher-to-score trên input phân biệt mean/max; chạy lại phần runtime bị ảnh hưởng. Hash config/controller chưa nằm trong runtime provenance hiện tại, nên cần bổ sung để đóng khoảng trống truy vết dependency.

### F2 — P1: warmup không chạy đúng sparse shape; “B=2048, gamma=15” chỉ đúng ở phần regular rounds

[Warmup](../../scripts/run_phase_a_paper_evaluation.py), dòng 150, dùng `max_new_tokens=8` nhưng `fixed_gamma=15`. `_action_grid` chỉ giữ gamma ≤ remaining−1. [Controller](../../src/TrainingFree/context_adaptive/controller.py), dòng 106–115 và 147–162, không tìm được action cố định rồi chọn từ **full-context** tail actions.

Tái hiện CPU:

| Remaining output | Budget thực được controller chọn | Gamma |
|---:|---|---:|
| 64 hoặc 16 | 2048 | 15 |
| 15 hoặc 12 | full | 11 |
| 8 | full | 7 |
| 4 | full | 3 |
| 3 | boundary AR | 0 |

Do đó warmup 8 token không warm sparse B=2048/gamma15. Cùng logic xảy ra ở đoạn cuối measured request 64 token. Đây là boundary handling trong executor, không phải controller Phase B được bật, nhưng nó thay đổi regime thực đo.

Counters xác nhận nominal budget không mô tả lượng context trung bình:

| Variant max | 8K | 16K |
|---|---:|---:|
| Context positions trung bình mỗi speculative round | 3.298,84 | 5.715,32 |
| Gamma trung bình | 13,19 | 12,86 |

Không được diễn giải kết quả như mọi round chỉ đọc 2.048 context positions. Tại 16K, mean context vượt nominal budget gần 2,8 lần. Chưa có per-round traces trong report runtime để xác nhận tỷ trọng từng tail branch; không gán mọi sai lệch cost cho một nhánh chỉ từ số trung bình.

**Điều kiện xử lý:** warm đúng shapes đo; khai báo rõ tail policy; giữ reduced budget ở tail nếu đó là phương pháp muốn đo, hoặc công khai đây là hybrid sparse/full executor; lưu actual counts, gamma và action reason theo round. Sau đó đo full summary để tail không chi phối đánh giá.

### F3 — P0 về contribution: nền tảng novelty đã có prior work rất gần

[Vegas, trước mang tên SpecAttn ở v1](https://arxiv.org/html/2602.07223v2), đã dùng verification để chọn KV cho lượt draft sau, phân tích nhiều query so với last-accepted query và đề xuất Collect-2-Query. Bản v2 ngày 29/05/2026 được trang arXiv ghi nhận accepted ICML’26; có [code tác giả](https://github.com/platformxlab/vegas). Đây là prior work trực tiếp phải đối chiếu, không chỉ liệt kê FastKV/SnapKV.

[SpecExtend §3.2](https://arxiv.org/html/2505.20776v4) đã dùng target attention từ verification, last-accepted query và chunk retrieval để chọn context cho drafter nhỏ trong long-document summarization. [SnapKV §4](https://arxiv.org/html/2404.14469) cũng đã tổng hợp nhiều observation queries và pooling để chọn KV theo head.

**Suy luận của review:** các ý “dùng target verification”, “gom nhiều query”, “không thêm model forward” và “nén draft context” không đủ làm novelty độc lập. Khác biệt còn có thể nghiên cứu là transfer sang diffusion drafter, accepted-prefix scope, toán tử pooling, layer/head transfer và cách giảm overhead. Hiện chưa chứng minh các khác biệt đó mang lại lợi ích vượt đối chứng gần.

Baseline cần ưu tiên là parent/CMR adaptation và Vegas-inspired query aggregation dưới cùng DFlash executor, budget, protection và cost accounting. Phân biệt rõ adaptation này với benchmark hệ thống gốc; Vegas là self-speculation nên không coi việc thay nhãn một hàm pooling thành tái lập nguyên hệ thống. Nếu claim về tốc độ hệ thống SOTA, cần thêm so sánh hệ thống tương thích thực sự.

### F4 — P1: kết quả hiện tại chưa cho thấy max-query vượt parent; ensemble có bất lợi rõ về latency

Tính ratio trên latency trung bình ba repetitions của mỗi document, rồi lấy geometric mean giữa 10 document:

| So sánh | Context | Ratio latency baseline / variant | Bootstrap CI95 |
|---|---:|---:|---:|
| Max so parent | 8K | 1,0062× | [0,9794; 1,0302] |
| Max so parent | 16K | 0,9982× | [0,9872; 1,0106] |
| Ensemble so parent | 8K | 0,9483× | [0,9288; 0,9681] |
| Ensemble so parent | 16K | 0,9475× | [0,9340; 0,9615] |

Max chưa có lợi ích latency rõ so parent trong phân tích ablation cùng ngân sách B=2048. Ensemble mất khoảng 5% throughput tương đối theo ratio này; latency của nó cao hơn parent khoảng 5,5%. Không nên dùng gain coverage để che kết quả bất lợi đó.
Cần phân biệt rõ hai tầng đánh giá:
1. **Ở cấp độ E2E hệ thống**: Cả Max M1 và Parent M0 đều tự thân giảm context từ full về B=2048, mang lại speedup thực tế so với Dense adapter (khoảng 1,05× ở 8K và 1,09× ở 16K). M1 sở hữu trọn vẹn kết quả tăng tốc này khi vận hành.
2. **Ở cấp độ Ablation phân rã thuật toán**: Việc so sánh M1 với Parent M0 nhằm kiểm tra xem cơ chế gom nhiều query của M1 có tạo ra giá trị gia tăng đáng kể so với cơ chế 1 query của Parent hay không. Kết quả hiện tại cho thấy chênh lệch giữa hai selector là không đáng kể về mặt thống kê.

Chênh E[L] giữa max và parent chỉ **+0,0247** candidate/round ở 8K, CI [−0,0759; +0,1084], và **−0,0016** ở 16K, CI [−0,0381; +0,0377]. Đây là chênh giữa các rollout có lịch round khác nhau, chưa phải ΔL same-prefix để xác nhận cơ chế. Nó cũng không đạt effect +0,25 đã đề xuất trong protocol cho một claim cải thiện acceptance có ý nghĩa thực tiễn.

CI chứa 0 hoặc 1 không chứng minh hai phương pháp tương đương; có thể thiếu power. Tuy nhiên, bằng chứng hiện tại không cho phép chọn max là contribution đã thắng parent trong ablation. Nếu held-out tiếp tục như vậy, phải chấp nhận kết luận âm cho giá trị tăng thêm của pooling.

### F5 — P1: cầu nối coverage → compressed proposal → E2E vẫn thiếu, và chunk thay đổi kết luận

Coverage chính đã chuyển sang next-round và macro theo document; đó là sửa đúng. Nhưng reference vẫn là attention mean của 15 verifier queries trên dense trajectory, gồm discarded suffix. Nó đo attention mass conditional, chưa đo semantic evidence, logits hoặc acceptance sau can thiệp.

Tại 16K/B=2048, reference E1=L18+L35:

| Selector | Token top-B, chunk=1 | Chunk=128 |
|---|---:|---:|
| Parent | 86,8947% | 80,4028% |
| Max accepted | 87,0562% | 80,3317% |
| Mean accepted | 87,0912% | 80,5780% |
| Ensemble accepted | 87,7909% | 80,6170% |

Max tăng **0,1615 điểm phần trăm** ở token selection; CI post-hoc [0,0292; 0,3177]. Đây là bằng chứng proxy dương trong cell này, nhưng effect nhỏ. Sang chunk128, chênh max−parent thành **−0,0711 điểm**, CI [−0,2034; +0,0504]. Vì vậy không chuyển trực tiếp kết luận token selection sang runtime chunk128.

Ensemble tăng E1 coverage +0,8962 điểm nhưng trên reference L18 riêng lại giảm −0,2208 điểm. Reference ensemble không phải semantic ground truth độc lập. Token selection và chunk selection còn khác actual counts: chunk128 thường chỉ dùng khoảng 1.945–1.949 token ở cell trên.

Runtime cũng bảo vệ generated keys khác offline, còn prompt layout coi toàn prompt là source. Vì vậy cần cùng mask/protection khi nối coverage với replay và runtime. Không lấy coverage của cấu hình A giải thích acceptance của cấu hình B.

**Một điểm lý luận cần giữ đúng:** query trên discarded suffix đã được tính trước action kế tiếp, nên dùng nó cho selection không tự động là future leakage. Nó là tín hiệu của nhánh bị từ chối, cần ablate lợi/hại. Accepted-only là một lựa chọn thiết kế, chưa được chứng minh tốt nhất. Max qua query cũng không bảo đảm giữ mọi token quan trọng khi budget hữu hạn.

### F6 — P1: cần tách ba loại parity; không bác bỏ toàn bộ pilot chỉ vì AR mismatch

Đọc trực tiếp output IDs cho kết quả:

- **240/240 reduced-context requests khớp dense adapter** trên toàn 64 token. Tính cả dense self-control là 300/300.
- So với target-only AR: 8K khớp 150/150; 16K khớp 90/150; toàn bộ là 240/300.
- Diagnostic chỉ chạy native ở bốn document 16K mà dense lệch AR: native khớp adapter **3/4**, khớp AR **1/4**.

Do đó quan sát “context reduction tăng tốc cùng output với dense adapter” vẫn có cơ sở trong pilot. Từ số liệu này, tỷ lệ latency max/dense có geometric speedup khoảng **1,0496× ở 8K** và **1,0880× ở 16K**. Đây là kết quả exploratory của executor và tail policy hiện tại, với warmup chưa đúng shapes.

Nhưng chưa thể gọi là lossless acceleration so với target AR, và chưa qua gate native/full-adapter 100% của protocol trên runtime L40S mới. Native mismatch có thể liên quan shape/precision hoặc khác biệt executor; teacher forcing ba ví dụ không giải thích hết mọi divergence. Không coi native luôn là gold correctness nếu chính nó khác AR, cũng không coi mọi AR mismatch là lỗi do selector.

Chưa có native DFlash latency control trong 300 request. Dense adapter vẫn gather/copy full bank; ở 16K, mean gather event time là khoảng 242,6 ms/request, so với 84,1 ms của max variant. Một phần gain có thể đến từ tránh overhead của adapter full gather. Cần native baseline để định lượng, thay vì suy đoán gain nào còn giữ trong engine tối ưu.

Subset AR-exact chỉ có 6 document ở 16K. Nó mô tả hiệu quả có điều kiện trên subset, không thay thế kết quả toàn cohort; tên `confirmed_speedup` trong code hình không biến nó thành held-out confirmation.

### F7 — P1: chưa hoàn thành nghiên cứu A1–A3 theo protocol; nhiều hình không tăng số document độc lập

Protocol gốc đề xuất 24 dev source groups và 48 held-out groups trên GovReport/Multi-News/QMSum; A1 can thiệp region/high-vs-low attention, A2 same-prefix Random/Recency/Target-parent/Previous-DFlash, A3 budget/refresh/full-generation.

Không thấy artifacts hoàn thành campaign đó trong workspace được kiểm tra. Rerun 19/10 là screening mới: 10 GovReport documents đã dùng trong nguồn ngày 06/10, 24 vòng đầu cho coverage, 64 output token cho timing. Không có bằng chứng về locked selector/layer/B trước một evaluation cohort chưa xem. Không kết luận các job chưa từng chạy ở nơi khác; kết luận là hồ sơ hiện tại chưa cung cấp bằng chứng hoàn thành.

960 snapshot, 920 transition, 14.592 group summary và 47 hình vẫn dựa vào **10 document**, cùng các document được lặp qua context. Cách bootstrap theo document hiện tại là đúng; CI đó không xử lý selection bias sau khi thử nhiều layer/budget/reference. Crosscheck hai evaluator cùng metric helper xác nhận tương thích số học, chưa là phép kiểm độc lập toàn bộ collector hoặc giá trị dự báo của metric.

Tất cả runtime dừng ở cap64. Vì vậy chưa có bằng chứng long-summary quality, EOS behavior ở request hoàn chỉnh, late-generation stability hoặc chi phí refresh amortized dài hạn. Không yêu cầu chạy A+B để sửa khoảng trống này: replay và full rollout **riêng Phase A** là đủ để kiểm các claim A.

### F8 — P2: tính đầy đủ của hệ thống và tài liệu chưa đạt mức artifact paper

Signal không miễn phí: mean CPU wall signal của max là khoảng **121 ms/request ở 8K**, **277 ms ở 16K**; ensemble lần lượt **227 ms**, **506 ms**. Không cộng các số CPU wall với CUDA events thành một decomposition cộng được. Các docs mới đã cảnh báo đúng, nhưng nếu contribution là low-overhead selection thì phải xác nhận trên native/engine tương ứng, thay vì dựa vào “không có forward phụ”.

Full `DraftContextBank` vẫn được cấp phát và cập nhật. B/N là tỷ lệ context được chọn cho draft attention, không phải tỷ lệ resident VRAM được cắt giảm. Continuity chưa chứng minh coalescing; chunk128 chưa chứng minh tích hợp PagedAttention. Đây hiện là giới hạn đã thừa nhận, không phải kết quả đã đạt.

Ngoài thư mục mới đã archive đúng, còn các báo cáo chưa gắn nhãn lỗi thời: [P1 coverage cũ](../../outputs/_PAPER_EXP/dflash_p1_coverage_experiment/README.md), dòng 77–83, vẫn ghi coverage 91–95% “đảm bảo” acceptance; [layer-selection cũ](../../outputs/_PAPER_EXP/dflash_layer_selection_experiment/REPORT.md) vẫn ghi M1 max, SOTA ensemble và cấm layer thấp. Các kết luận đó đã bị chính audit nguồn bác bỏ. Không để chúng thành nguồn cho abstract/caption mới.

Protocol cũng còn một câu yêu cầu reference trajectory khớp AR ở mục cohort, trong khi amendment chấp nhận native/full-adapter parity. Cần một protocol version rõ hiệu lực và decision ledger trước confirm; không trộn story joint cũ với Phase A screening mới.

## 4. Claim nào hiện có thể viết?

| Claim | Trạng thái | Cách viết phù hợp |
|---|---|---|
| Target-parent và DFlash có attention khác nhau trong cohort đã đo | Được hỗ trợ trong phạm vi distribution study | Ghi model, cohort, denominator; không suy ra mất semantic information |
| Max accepted tăng coverage so parent tại 16K/B2048/token/E1 | Có bằng chứng exploratory nhỏ | +0,1615 điểm phần trăm, post-hoc; không chuyển thành acceptance gain |
| Reduced context giúp prototype nhanh hơn dense adapter | Có tín hiệu pilot | Khoảng 1,05–1,09×, cùng output adapter, cap64, tail policy và warmup limits rõ |
| Mean và max tương đương về runtime | Chưa được kiểm hợp lệ | Cần chạy mean thật |
| Max/ensemble vượt parent về hiệu quả thực tế | Chưa được hỗ trợ | Max CI chứa 1; ensemble chậm hơn parent trong pilot |
| Selector giữ acceptance vì coverage >90% | Không có suy luận hợp lệ | Cần intervention cùng prefix và non-inferiority đã khóa |
| B2048 giữ nguyên chất lượng full summaries | Chưa được kiểm | Cần natural-generation evaluation |
| Lossless so target AR | Chưa qua gate thực nghiệm | Cần verifier/executor contract và diagnostic đầy đủ |
| Giảm 87,5% resident VRAM, hardware coalescing, SOTA | Chưa có bằng chứng | Không đưa vào contribution hiện tại |
| Đây là ý tưởng đầu tiên dùng verification/multiple queries chọn draft context | Không phù hợp prior work đã xác minh | Định vị contribution tăng thêm so Vegas/SpecExtend |

## 5. Đánh giá theo tiêu chuẩn reviewer

Thang phụ 1–5 do lượt review này quy ước; 5 là mạnh.

| Tiêu chí | Điểm | Lý do |
|---|---:|---|
| Câu hỏi nghiên cứu và động cơ | 4 | Vấn đề draft context dài đáng nghiên cứu; có dữ liệu nền |
| Novelty của Phase A độc lập | 2 | Prior work rất gần; khác biệt chính chưa được chứng minh hữu ích |
| Soundness của bằng chứng cho method claim | 2 | Sai binding mean và measurement regime; thiếu causal/held-out bridge |
| Độ đầy đủ thực nghiệm | 2 | Nhiều proxy, cohort nhỏ, output ngắn, thiếu native timing và controls gần |
| Significance đã thể hiện | 2 | Gain so dense prototype nhỏ; chưa vượt parent; ensemble bất lợi |
| Truy vết và sự thận trọng của docs mới | 4 | Layer IDs/query scope/provenance và giới hạn được sửa đáng kể |

Điểm mạnh đáng giữ: ghi rõ kết quả âm và giới hạn, không đổi missing layer bằng fallback, lưu per-query dimensions, kiểm causal mask, dùng document aggregation, đo request thực, giữ artifact lịch sử và hash. Reviewer nên công nhận các cải thiện này. Tuy nhiên, tính trung thực khi mô tả một pilot chưa hoàn thiện không thay thế bằng chứng cần có cho một contribution đã xác nhận.

## 6. Đề xuất cho Phase A và các gate trước paper

### 6.1. Định vị lại story để có thể bị bác bỏ

Story hiện có thể bảo vệ là: **nghiên cứu khả năng chuyển verification-guided context selection sang pretrained diffusion drafter, và kiểm xem thêm query/layer có đáng chi phí hơn parent-query đơn giản hay không.**

Đây là câu hỏi mở, chưa phải kết quả “max-query + ensemble là lời giải”. Chỉ giữ max/ensemble như method chính nếu chúng vượt đối chứng có thể triển khai ở metric đã khóa. Nếu parent giữ cùng hiệu quả với chi phí thấp hơn, chọn parent làm phương pháp thực dụng và báo pooling/ensemble là kết quả âm. Để thành paper top-tier độc lập, vẫn cần insight mới đủ mạnh hoặc cải thiện hệ thống rõ sau khi so prior work; việc hoàn thành thêm một bảng benchmark không tự giải quyết novelty.

### 6.2. Thứ tự xử lý đề xuất

1. **Đóng measurement defects trước GPU campaign:** sửa binding mean, warm đúng shape, quyết định tail policy, lưu effective configs và per-round actual budgets/gamma. Không cần chạy lại coverage đúng chỉ để sửa một hàng runtime.
2. **Khóa correctness và native baseline:** so native/full-adapter trên toàn cohort, kiểm collector on/off, same-history logits, crop/append, reject/all-accept/EOS/cap. Chọn rõ kết luận relative-native hay exact-AR. Không sửa gate sau khi xem kết quả để giữ một headline.
3. **Phép thử có giá trị thông tin cao nhất:** same-prefix replay với parent, mean thật, max, ensemble, Random/Recency và Vegas-inspired aggregation; cùng actual budget/protection, có low/high-attention deletion nếu giữ mechanistic claim. Dùng regular early/middle/late prefixes, chấm ΔL và logit agreement; coverage là secondary. Đây là bước quyết định có nên đầu tư tiếp vào multi-query/ensemble.
4. **Dev → khóa → held-out:** dùng cohort đa dataset đã lên kế hoạch; audit exposure của 10 document cũ; khóa selector/layer/B/chunk/gamma/baselines và non-inferiority trước test. Family và pooling strategy chọn bằng dev, CI ghép cặp theo source group; CI quá rộng thì kết luận inconclusive.
5. **Full rollout riêng A:** native DFlash, full adapter và các selector đã khóa; natural EOS với output cap phù hợp summarization, repeats/random order, E2E và decode latency, output parity, acceptance, ROUGE qua helper chung khi có reference. Nếu output thật sự exact với baseline thì quality equality đi theo output equality; không cần dựng claim quality độc lập từ coverage.
6. **Xác nhận độ rộng tương ứng claim:** nếu nói phương pháp tổng quát, cần thêm model/checkpoint và kiểm transfer rule; nếu chỉ Qwen3-4B batch1, giới hạn title/abstract/conclusion đúng phạm vi. Claim hardware/serving cần engine/hardware evidence tương ứng. Phase B hoặc joint A+B chỉ cần khi paper đưa chúng vào contribution.

### 6.3. Tiêu chí mở lại quyết định submission

| Gate | Điều kiện tối thiểu |
|---|---|
| Định danh phương pháp | Một cấu hình chính, đầy đủ pooling scope/layer/head/key selection/protection/refresh/tail; config thực khớp nhãn |
| Causal evidence | Selector vượt đối chứng đã chọn trên same-prefix hoặc claim cơ chế được rút đúng phạm vi |
| Giá trị tăng thêm | Vượt parent/CMR và aggregation adaptation gần nhất về trade-off đã khóa; hoặc có finding âm mới, đủ khái quát và hữu ích |
| Correctness | Gate đã khóa đạt trên runtime dùng báo cáo; native/adapter/AR được phân biệt rõ |
| Hiệu năng | Full-summary paired E2E với native control và full overhead; uncertainty phù hợp; gain không chỉ từ dense-adapter overhead |
| Xác nhận | Cohort held-out, khóa trước test, natural/capped và failures được công bố |
| Paper artifacts | Claim–experiment–figure mapping duy nhất; raw/provenance truy cập được; báo cáo cũ không còn dẫn nhầm |

Protocol A3 đã đề xuất geometric macro speedup ≥1,05, CI lower >1 và acceptance non-inferiority ΔL lower >−0,25. Các ngưỡng đó là một điểm xuất phát đã có; không chọn lại theo rerun. **Đạt chúng so full baseline vẫn chưa đủ chứng minh novelty của multi-query nếu parent cũng đạt hoặc tốt hơn.**

## 7. Kết luận theo mức bằng chứng

### CONFIRMED — Đã xác nhận trong phạm vi cụ thể

- Provenance code/report và 94 file hình được đối chiếu nhất quán; log runtime có 300 unique cells.
- Trong 64-token horizon, 240/240 reduced requests giữ đúng output dense adapter; 240/300 tổng requests khớp target AR.
- Launcher hiện tại nối nhãn mean vào default max; CPU reproduction và 60 cặp counters/output cùng hỗ trợ chẩn đoán.
- Controller hiện tại chuyển full budget khi fixed gamma không còn feasible; warmup cap8 đi vào nhánh này.
- **Không có claim “phương pháp đề xuất thắng theo locked full evaluation” được xác nhận trong hồ sơ đã kiểm tra.**

### EXPLORATORY — Quan sát cần xác nhận

- Context reduction có thể cải thiện cả acceptance và latency so dense adapter trên cohort này.
- Max chưa cho thấy giá trị thêm rõ so parent; ensemble tăng một số coverage proxy nhưng tốn latency.
- Một số contrast next-round coverage có CI dương; cần giữ scope proxy và tính post-hoc.

### FAILED / INCOMPLETE — Sai phép thử hoặc chưa hoàn thành

- Runtime ablation mean/max không hợp lệ do wiring; warmup không phủ measured sparse shape.
- Native/full-adapter parity chưa đạt 100% trong diagnostic L40S; exact-AR cũng chưa đạt. Chưa quy được mọi mismatch cho một nguyên nhân.
- Chưa có bằng chứng A1/A2 can thiệp và A3 full-summary held-out đầy đủ cho selector đề xuất trong workspace. Đây là thiếu đánh giá khoa học, không phải bằng chứng về lỗi hạ tầng hoặc thất bại của mọi selector.

### HIGHEST VERIFIED RUNG — Mức kiểm chứng cao nhất

Theo ladder inference trong protocol A, runtime L40S mới có bằng chứng forward/collector ở **R2**; **R3 native/full-adapter parity chưa xanh**. Có artifacts launcher/rollout và số liệu ở quy mô R4/R5, nhưng không được coi là đã thông qua promotion gates cho method đang claim, đặc biệt hàng mean. Chưa đạt **R6 — full run/full study** của Phase A đề xuất. Review memo không tự nâng method lên R7 nếu full evaluation chưa hoàn thành.

Phép đo phân phối parent-to-next-draft ngày 06/10 có thể hoàn tất ở scope nghiên cứu riêng của nó; không chuyển rung đó sang claim hiệu quả context selection.

### EVIDENCE GAPS — Khoảng trống quyết định

Giá trị tăng thêm so parent/prior work, mean runtime thật, native baseline, matched-budget causal replay, full summary và held-out generalization còn thiếu. Correctness/pooling/tail config cần được đóng trước khi mở rộng compute.

### RECOMMENDED NEXT — Phép thử tiếp theo

Sau khi sửa measurement và qua gate correctness, chạy **same-prefix selector replay riêng Phase A với parent/mean/max/ensemble cùng controls gần**, để kiểm tra trực tiếp liệu aggregation có cải thiện proposal đủ đáng chi phí. Đây là pilot R5 cần có trước full study R6. Nếu không có lợi ích so parent, đổi story theo kết quả âm thay vì mở rộng ensemble/controller để tìm một bảng thắng.

**Trạng thái cuối: đã có distribution study và prototype pilot có ích; Phase A phương pháp chưa qua gate correctness R3 trên runtime mới và chưa được xác nhận bằng full study R6. Chưa sẵn sàng làm contribution phương pháp đã hoàn thiện cho paper top-tier.**
