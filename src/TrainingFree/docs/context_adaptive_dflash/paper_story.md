# Paper story và phạm vi claim

Ngày: **2026-10-08**. Đây là proposal; không chứa kết quả speedup của phương pháp mới. [Thiết kế](design.md), [protocol](experiment_protocol.md).

## 1. Tên và contribution một câu

**Context-Adaptive DFlash: Joint Adaptation of Draft Context and Speculation Length for Long-Context Summarization.**

Đề xuất controller training-free cùng điều chỉnh context budget và diffusion block length bằng source relevance, uncertainty và measured cost, trong khi target giữ full context để exact verification.

## 2. Lập luận từ nền tảng đến câu hỏi mới

**Problem.** Long-context summarization kết hợp tài liệu dài với generation nhiều vòng. Chi phí source-context reads, chất lượng draft và số candidate positions được verify cùng quyết định inference latency.

**Established foundation.** DFlash cung cấp pretrained diffusion drafter. SpecExtend cung cấp bằng chứng selective context có thể cải thiện cả drafting cost và acceptance trong speculative decoding. A được xây trên nền tảng đó; paper mới cần đo chuyển giao sang DFlash. [DFlash](https://arxiv.org/abs/2602.06036), [SpecExtend](https://arxiv.org/html/2505.20776v4).

**Motivation from this repo.** Attention analyses đã có cho thấy temporal locality và nhu cầu context khác nhau giữa các layer. Target-parent ranking cũng khác ranking DFlash trên cohort đã khảo sát. Các số đo đó tạo động cơ so target-based selection với draft-based refresh; chưa chứng minh selection nào tạo acceptance/speedup tốt hơn. [Phân tích nguồn](../../../../docs/experiments/2026-10-06_dflash_context_memory_paper_analysis.md).

**Research question.** Khi lượng draft context thay đổi, block length tối ưu thay đổi thế nào? Source relevance và uncertainty có giúp chọn cặp action hiệu quả hơn cấu hình fixed và hai controller độc lập không?

**Central hypothesis.** Selected evidence tác động tới mức drafter khớp target, nên hiệu quả của block dài phụ thuộc budget/context được cấp. Một policy phối hợp có thể giảm estimated cost per committed token và E2E latency. Đây là giả thuyết cần kiểm chứng bằng intervention và rollout.

**Method.** A chọn context từ source + global protection + recent processed output; B ước lượng hiệu quả của gamma bằng parent entropy/history; C tối ưu cặp `(B,gamma)` theo cost/expected commit. Trọng số không thay đổi; target full context và exact verifier quyết định output.

**Evidence.** Chứng minh interaction trên action grid cùng prefix states, sau đó đánh giá real rollout so best fixed pair/A/B/independent A+B. Tính cả retrieval, gather, refresh, controller và prefill trong E2E.

## 3. Định vị prior work

| Nguồn đã xác minh | Liên quan và ranh giới claim |
|---|---|
| [SpecExtend](https://arxiv.org/abs/2505.20776) | Attention-based context selection là nền tảng kế thừa |
| [AdaFlash](https://arxiv.org/html/2607.19223) | Adaptive length với learned head/online updates; cần phân biệt với frozen-weight policy |
| [SparseSpec-L](https://arxiv.org/html/2607.27735) | Có sparse context và entropy/cost-based adaptive length trong self-speculation |
| [LibraSpec](https://arxiv.org/html/2608.08721) | Có training-free adaptive speculation length cho diffusion drafters, gồm DFlash |

Novelty dự kiến phải dựa vào finding/context-dependent action interaction và policy chung trên pretrained diffusion drafter. Không claim lần đầu kết hợp sparsity, entropy hoặc adaptive length. Source-aware selection tự nó chưa là novelty; paper phải chỉ ra tín hiệu nguồn bổ sung giá trị cho quyết định joint trong summarization.

## 4. Ba contribution mục tiêu

1. **Empirical characterization:** cost/acceptance surface theo draft budget, gamma và generation state; kiểm tra budget có làm thay đổi gamma tối ưu.
2. **Joint adaptive method:** controller frozen-weight dùng causal signals và calibration/online statistics để chọn effective context và actual diffusion shape.
3. **Exact inference evaluation:** implementation có full target verification, đo E2E trên source-group heldout với strong fixed/independent controls và đủ overhead accounting.

Entropy chỉ là contribution nếu so history-only cho thấy predictive và latency benefit. Draft refresh chỉ được nhấn mạnh như contribution riêng nếu vượt target-parent/recency ở cùng budget và amortized cost. Memory reduction chỉ được claim khi peak resident bytes thực giảm; full-bank gather không mặc định đạt điều đó.

## 5. Draft giới thiệu đề tài

Chúng tôi nghiên cứu training-free adaptation cho DFlash trong long-context summarization. Việc lựa chọn context phù hợp đã có tiền lệ giúp drafter vừa giảm chi phí vừa dự đoán chính xác hơn; từ nền tảng này, chúng tôi đặt câu hỏi liệu lượng source evidence được cấp cho diffusion drafter có làm thay đổi speculation length tối ưu trong quá trình sinh summary hay không.

Phương pháp đề xuất sử dụng source relevance, uncertainty có sẵn và acceptance history để cùng chọn draft-context budget với block length, dựa trên estimated latency trên mỗi token được commit. Target giữ context đầy đủ và thực hiện exact verification, còn trọng số toàn bộ mô hình được giữ nguyên. Chúng tôi thiết kế intervention trên cùng prefix states và rollout với overhead đầy đủ để kiểm tra lợi ích so với best fixed configuration và adaptation độc lập.

## 6. Hình và bảng gắn với claim

- Figure 1: cost surface `(B,gamma)` tại các state khác nhau; đánh dấu fixed/independent/joint action. Chỉ vẽ dữ liệu đo thật.
- Figure 2: full-target verification và draft-side selection, parent signals từ vòng trước, controller feedback.
- Figure 3: pre-round entropy/history so expected accepted prefix, stratify theo budget/gamma; CI theo document.
- Main table: paired E2E speedup/CI, correctness, acceptance và memory của fixed/A/B/independent/joint.
- Ablation table: entropy, source relevance, selector và refresh cùng cost scope.
- Scaling plot: natural output/input regimes; fixed-token stress plots có nhãn scope riêng.

## 7. Điều chỉnh story theo kết quả

Nếu A có lợi nhưng B không vượt best fixed, giữ selective context finding và giảm adaptive-length claim. Nếu A/B đều có lợi nhưng joint không vượt independent, không claim lợi ích từ phối hợp. Nếu interaction có nhưng controller không khai thác được, báo empirical finding và phần policy còn hạn chế. Nếu speedup chỉ có trên fixed-token stress run, giới hạn claim vào regime đó.

Khi viết abstract kết quả cuối, chỉ bổ sung số đo có completed artifacts, paired CI và gate report. Proposal này không dùng động từ khẳng định đã đạt speedup, generalization hoặc memory saving cho executor chưa triển khai.
