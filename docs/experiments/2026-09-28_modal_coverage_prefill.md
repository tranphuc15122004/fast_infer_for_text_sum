# Modal pilot: coverage-aware prefill selection

Ngày chạy: 2026-09-28  
Run: `coverage-prefill-pilot-20260928-b`  
GPU: Modal A100-80GB  
Model: `Qwen/Qwen3-4B`, cached snapshot `1cfa9a7208912126459214e8b04321603b3df60c`  
Runtime theo `requirements.modal.txt`: Python 3.12, Torch 2.11.0, Transformers 5.12.1  
Precision/backend: BF16 / SDPA

## Câu hỏi

Kiểm tra liệu selector giữ một nửa ngân sách ở đầu tài liệu và phân bổ phần
còn lại qua tám vùng của source có giảm E2E so với full context và Lead cùng
ngân sách hay không, đồng thời giữ chất lượng tóm tắt.

## Protocol

- 20 mẫu đầu của `gov_report` và `qmsum` trong `data/longbench_100_14k`.
- Cùng model, prompt, greedy decoding, tối đa 128 output token và GPU cho cả
  `full`, `lead` và `coverage`.
- Ngân sách selector: 6,144 token theo tokenizer Qwen3-4B. Nếu source ngắn hơn
  budget, cả ba method nhận cùng toàn văn.
- `lead` là prefix cứng gồm 6,144 token đầu (không phải sentence-ranked
  `LeadSelector` hiện có).
- Selector latency được cộng vào `pipeline_e2e_ms`. Generation dùng chung
  `generate_greedy_profiled` của semantic-selection runner.
- Thứ tự method xoay vòng theo mẫu. Có một warmup ngắn trước khi đo.
- Chỉ số chất lượng gồm ROUGE-1/2/L và phép đếm recall số đơn giản. Chưa có
  factuality judge hoặc entity-level audit.

Lệnh:

```bash
MODAL_PROFILE=tdphuc-work modal run scripts/modal_coverage_prefill.py \
  --run-id coverage-prefill-pilot-20260928-b \
  --datasets gov_report,qmsum --max-samples 20 \
  --max-new-tokens 128 --budget 6144
```

Dataset SHA256:

- `gov_report.jsonl`: `3dbba6ee4874e2a00b9d0c706524ea2dfb47a60169491a54464b686189299813`
- `qmsum.jsonl`: `3ebf2c39a0a3a4ffc5a8ff367837b550e2b017bc125791c3ebd13fab06f9a689`

## Kết quả

| Dataset | Method | Mean pipeline E2E (ms) | Speedup vs full (95% paired bootstrap CI) | Mean prefill (ms) | Selector (ms) | ROUGE-L | Output tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| GovReport | Full | 5,958 | 1.000× | 592 | 0 | 0.1291 | 128.0 |
| GovReport | Lead-6144 | 5,723 | 1.041× (1.026, 1.056) | 358 | 26 | 0.1281 | 128.0 |
| GovReport | Coverage-6144 | 5,778 | 1.031× (1.018, 1.044) | 367 | 61 | 0.1304 | 128.0 |
| QMSum | Full | 6,012 | 1.000× | 688 | 0 | 0.1547 | 128.0 |
| QMSum | Lead-6144 | 5,688 | 1.057× (1.042, 1.072) | 376 | 32 | 0.1564 | 128.0 |
| QMSum | Coverage-6144 | 5,676 | 1.059× (1.040, 1.079) | 380 | 83 | 0.1555 | 126.4 |

Bootstrap samples document pairs with replacement (20,000 resamples, seed 734).
For QMSum coverage, 5/20 generations ended before 128 tokens; restricting the
comparison to the 15 equal-output-length pairs gives speedup `1.048×`
(`95% CI 1.029–1.066`) over full context.

Coverage is slower than Lead on GovReport: paired ratio-of-means
`Lead / Coverage = 0.991×` (`95% CI 0.984–0.997`). On QMSum, the two are
effectively tied: `Lead / Coverage = 1.002×` (`95% CI 0.992–1.014`).

Mean ROUGE-L difference against full context was `+0.0013` on GovReport
(`95% CI -0.0031 to +0.0056`) and `+0.0007` on QMSum
(`95% CI -0.0067 to +0.0081`). Against Lead it was `+0.0023` on GovReport
(`95% CI -0.0035 to +0.0081`) and `-0.0009` on QMSum
(`95% CI -0.0074 to +0.0055`). These intervals do not establish quality
non-inferiority or an improvement.

## Kết luận

Coverage selection reduces prefill time by roughly 38–45% on these cohorts and
reduces mean peak allocated GPU memory from 11.3–11.9 GiB to about 10.0–10.2
GiB. After selector overhead, measured E2E improves only 3.1% on GovReport and
5.9% on QMSum versus full context. Plain Lead is faster on GovReport and
statistically indistinguishable on QMSum, with much lower selection cost.

**Decision: do not promote this coverage rule.** The run demonstrates a small
prefill/E2E benefit against full context, but it does not show a practical
efficiency advantage over plain Lead, and quality retention remains
inconclusive. The five early-stop QMSum outputs also confound raw E2E; their
equal-length subset still shows only about a 4.8% speedup over full context.

## Artifacts

- Runner: `scripts/modal_coverage_prefill.py`
- Paired implementation: `scripts/probe_coverage_prefill.py`
- Modal Volume: `outputs/coverage_prefill/coverage-prefill-pilot-20260928-b/paired.jsonl`
- Aggregate: `outputs/coverage_prefill/coverage-prefill-pilot-20260928-b/paired.summary.json`
- Local sample-level copy: `outputs/coverage_prefill/pilot-b-paired.jsonl`
