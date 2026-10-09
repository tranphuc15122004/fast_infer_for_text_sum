# Kế hoạch và ma trận thực nghiệm Context-Adaptive DFlash

Ngày: **2026-10-09**. Đây là kế hoạch thu bằng chứng; **G0–G6 chưa được xác nhận trên GPU**. Định nghĩa thuật toán ở [design](design.md), tiêu chí nghiên cứu ở [protocol](experiment_protocol.md), lệnh chạy ở [runbook](runbook.md), kiểm chứng GPU ở [gpu_validation](gpu_validation.md). Dùng [mẫu báo cáo](results_template.md) để ghi quyết định và sai lệch protocol.

## 1. Giả thuyết, đối chứng và phạm vi kết luận

| Giả thuyết | So sánh bắt buộc | Bằng chứng | Khi không đạt |
|---|---|---|---|
| H-A: context chọn lọc có ích cho draft DFlash | Fixed selector/B so full draft, cùng gamma; thêm `a_only` so full fixed | Prefix replay, accepted prefix và E2E rollout gồm selector/gather/refresh | Báo vùng workload có ích hoặc kết quả âm; không suy speedup từ attention mass |
| H-B: chọn gamma theo cost cải thiện inference | `b_only_history`, `b_only_entropy` so best full fixed gamma | Paired E2E; prediction/regret trên prefix chưa dùng để calibration | Nếu entropy không hơn history-only, bỏ claim về contribution entropy |
| H-C: B thay đổi gamma tốt nhất và joint khai thác được | `joint` so `best_fixed_pair` và `independent_ab`, cả online/frozen | Action-cost surface, interaction vượt timing noise, paired E2E | Nếu chỉ online có gain, quy gain cho policy cùng online statistics; chưa tách được interaction |
| H-S: gain đủ bù overhead trong natural summarization | AR/full DFlash/joint theo input/output bins | E2E, TTFT, TPOT, memory, actual output length và ROUGE | Báo regime thất bại; không dùng fixed-token run thay natural headline |

Primary là **greedy natural summarization**, Qwen3-4B + Qwen3-4B-DFlash-b16, no thinking, batch 1, một process/GPU, output cap 2048. Trọng số giữ nguyên. Calibration là thu thống kê và profile inference, không tối ưu trọng số.

## 2. Đăng ký trước khi chạy

Tạo `evidence/preregistration.md` trong run root, điền trước khi xem dev results:

- Run ID, người vận hành, ngày UTC, code commit/dirty và implementation hash.
- Target/draft/tokenizer local paths và content hashes; GPU, driver, torch/CUDA, backend thực tế và dtype.
- Dataset file hashes, source-group split, exposure audit và scope natural/fixed tokens.
- Action grid, selector/protection/refresh, min support, priors, fallback, seed, repetitions, warmups và lịch cells.
- Fixed-pair search, quy tắc chọn model/policy trên dev và các comparison/CI đã đăng ký.
- G0–G6, ngưỡng bên dưới, các kiểm chứng cần harness riêng và mọi sai lệch đã biết.

Lưu manifest do executor sinh; không đưa HF token hoặc toàn bộ master-env vào hồ sơ. `run_id` nhận diện một campaign cùng code/data/runtime. Trong campaign, sweep dùng output roots riêng; calibration được dùng chung chỉ khi signature khớp. Đổi selector, target layer, refresh, source protection, grid, output scope hoặc runtime cần calibration tương ứng.

## 3. Cohort primary và exposure audit

CLI đọc **mọi `*.jsonl` trong `--data-dir`** và chưa có `--dataset`. Runbook tạo một thư mục chỉ chứa `gov_report`, `qmsum`, `multi_news`; giữ corpus code-completion cho campaign secondary riêng.

Metadata đã kiểm tra: primary pool có 300 records, 220 source groups trước đối soát exposure/cross-dataset duplicates. Không có exposure thì group quotas là:

| Dataset | Calibration | Dev | Test | Đơn vị độc lập |
|---|---:|---:|---:|---|
| GovReport | 20 | 20 | 60 | Report nguồn |
| Multi-News | 20 | 20 | 60 | Cụm bài nguồn |
| QMSum | 4 | 4 | 12 | Conversation; giữ mọi query cùng split |

Counts cuối lấy từ `split_manifest.json`; số query records QMSum tùy nhóm. Bản prepare CPU ngày 2026-10-08 trên cả năm datasets có 500 records/420 groups và **10 exposure IDs chưa resolve**; đây chưa phải split primary đã được audit. Xem [báo cáo khắc phục](implementation_review_2026-10-08.md).

### Hồ sơ đối soát 10 IDs

Đối chiếu probe manifest `outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/manifest.json` với nguồn `data/longbench_200/gov_report.jsonl` đã mirror và primary corpus. Lập `evidence/exposure_audit.json` với mỗi probe ID:

| Field | Nội dung |
|---|---|
| `probe_sample_id`, `probe_manifest_sha256` | Danh tính và artifact gốc |
| `dataset`, `source_split`, `source_index` | Provenance document; ghi null khi chưa có |
| `source_sha256`, `normalized_source_fingerprint` | Hash raw context và hash theo NFKC/whitespace của protocol |
| `canonical_record_ids`, `canonical_source_group_ids` | Mọi record/group tương ứng trong primary pool |
| `decision` | `matched_dev`, `absent_verified` hoặc `unresolved` |
| `evidence`, `reviewed_at_utc` | File/hash và căn cứ quyết định |

Không coi ID khác nhau hoặc raw-context hash khác nhau là bằng chứng document khác nhau: có thể là truncation/version của cùng nguồn. `absent_verified` cần đối soát provenance và các biến thể nguồn; nếu thiếu nguồn gốc, giữ `unresolved`.

Exposure manifest bổ sung mà CLI hiểu có dạng dưới đây; đây là schema minh họa, thay mọi placeholder bằng dữ liệu thật:

```json
{
  "samples": [
    {"dataset": "gov_report", "sample_id": "<canonical_id>", "source_sha256": "<sha256_raw_context>"}
  ],
  "source_fingerprints": ["<fingerprint_normalized_context>"]
}
```

Truyền cả manifest gốc và manifest bổ sung cho `prepare`. Mọi group khớp phải ở dev, kể cả query/truncation variants. Danh sách `unresolved_exposure_sample_ids` trong split không tự làm CLI fail: trước lock phải đối soát từng entry với audit. Chỉ cho phép remaining IDs có `absent_verified` với bằng chứng; còn `unresolved` thì chưa được claim untouched heldout. Chốt exposure và build split trước model smoke.

## 4. Ma trận theo thứ tự

| ID | Phase / cohort | Cells và scope | Điều kiện chuyển bước |
|---|---|---|---|
| E0 | `prepare`, `preflight --model-check` | Primary corpus + snapshot thật, chưa forward | Split/exposure hợp lệ; schema/config/asset compatibility |
| E1 | Smoke + harness G1/G2 | 2 dev groups/dataset, 1 query/group, cap 32; mở rộng 8 GovReport + 8 Multi-News + 4 QMSum groups, cap 256 | Greedy exact; bank/positions/parent/boundary đúng trên backend thật |
| E2-P | Calibration pilot | Tối đa 16 calibration records; full trajectory, grid 4×4, checkpoints 0/512/1024/1536, 3 reps | Artifact/profile/support/fallback có thể audit; chưa thay full calibration |
| E2 | Full calibration | Toàn calibration split, cùng grid/checkpoints/reps | Priors hợp lệ; report support và thiếu context/refresh buckets |
| E3 | Fixed-pair search dev | 16 pairs × AR + fixed pair + full fixed cùng gamma | Chọn fixed pair và full gamma bằng cùng statistic dev |
| E4 | Selector study dev | `target_parent`, `draft_refresh`, `recent_only`, `random`; fixed gamma 15, B∈grid | G3; selector nào chưa có calibration không được dùng cho adaptive rollout |
| E5 | Final dev matrix | 10 variants dưới đây, online; thêm frozen controls | G4/G5 và exact coverage; ghi quyết định giữ/thu hẹp claim |
| E6 | Lock + full test | Toàn `test_variants` trong lock, toàn test split | G6; không đổi policy sau xem test |
| E7 | Secondary | Fixed output 256/512/1024/2048; code datasets hoặc context dài hơn | Run/lock/calibration/output scope riêng; không gộp headline |

E1 phải đọc dữ liệu dev đã xuất riêng: **phase `smoke` hiện không lọc dev bằng split manifest**. Lệnh và cách xuất nằm trong runbook. `--max-samples` chỉ giới hạn số records, không chọn đúng quota groups của E1 mở rộng.

E2 replay trên prefix bất biến ở checkpoint thực tế đầu tiên đạt/vượt mốc; trace ghi `calibration_checkpoint` thực, không mặc định đúng 512/1024/1536. EOS sớm làm thiếu mốc là dữ liệu hợp lệ. Một prefix/action lặp 3 lần chỉ đóng góp **một outcome support**, ba timing observations. QMSum nhiều query không tạo thêm independent documents. Báo cả unique states và source groups hỗ trợ prior.

V1 cost lookup cần đúng context bucket `ceil(logical_length/2048)` và refresh mode; bucket chưa profile phải fallback, không mượn cost bucket gần nhất. Report tỷ lệ unsupported/unseen buckets, full-context fallback và action usage; priors đọc được chưa bảo đảm adaptive policy thực sự hoạt động.

### Final dev và heldout variants

```text
ar, dflash_full_fixed, best_fixed_pair, a_only,
b_only_history, b_only_entropy, independent_ab, joint,
joint_no_entropy, joint_no_source_relevance
```

Giữ cùng `fixed_budget`, `fixed_gamma`, `gamma_reference` và calibration path ngay cả trên AR, vì report kiểm provenance của các field này. `dflash_full_fixed` dùng gamma reference; fixed pair/A-only dùng fixed gamma. Candidate joint và independent phải cùng selector, action grid và statistics mode.

Nếu best pair gamma khác best full gamma, final `a_only→dflash_full_fixed` thay cả context và gamma; chưa tách riêng contribution A. G3 cần thêm root có full/fixed-selector/A cùng gamma (15 trong selector study hoặc gamma đã đăng ký), cùng paired provenance. Report validity không tự kiểm causal control này.

Frozen control tối thiểu gồm AR, full fixed, best fixed, independent và joint trong cùng mode. Nếu muốn kết luận riêng về entropy/source relevance, chạy các signal ablations cùng mode. Tạo report `online` và `frozen` riêng.

### Quy tắc tuning

1. Chỉ dùng calibration để fit priors/cutpoints; dùng dev để chọn fixed actions, selector và policy.
2. Trong mỗi fixed-pair root, ghép AR/fixed/full trên cùng IDs/repetitions. Chọn pair có geometric macro speedup so AR lớn nhất theo protocol; chọn full gamma tương tự trên bốn full settings. AR được đo lại từng root nên ranking speedup chưa nhất thiết giống ranking absolute latency; lưu cả hai, kiểm drift và chạy lại controls khi có contention. Không chọn bằng một dataset hoặc acceptance riêng.
3. Full cùng gamma xuất hiện trong nhiều fixed-pair roots thì lấy trung bình log statistic qua các replicates đã đăng ký, không chọn root nhanh nhất; báo drift giữa roots. Khi bằng nhau trong độ chính xác số, ưu tiên full context, rồi gamma nhỏ hơn, rồi budget lớn hơn; ghi toàn tuning table và uncertainty. Không diễn giải chênh lệch trong timing noise là interaction.
4. Tuning selector dùng cùng fixed gamma 15 và controls. Chọn selector theo E2E gồm overhead; nếu đổi selector thì thu calibration tương ứng trước adaptive dev.
5. Chạy lại final dev matrix với actions/selector đã chọn, đủ IDs và ba repetitions, rồi mới lock. Fixed pair/gamma trong lệnh runbook là placeholders do dev quyết định.

## 5. Lịch chạy, số lượng và chi phí

V1 chạy trọn các repetitions trong một invocation/variant. Shuffle **thứ tự cell blocks** bằng seed 42 và lưu `schedule.json` trước chạy. Seeds 42/43/44 của repetitions là RNG seeds; chúng không tạo ra ba thứ tự variants khác nhau. Interleave/shuffle từng repetition cần scheduler riêng, chưa có trong CLI. Ghi đúng chế độ scheduling vào hồ sơ; sensitivity run đổi thứ tự blocks trên dev nếu drift đáng kể.

Warmup hiện chạy ba lần trên records của cohort đã chọn, cap tối đa 16 tokens, ngoài measured artifacts/timing; từng request tạo cache/statistics mới. Đây chưa phải warmup mọi action shape hoặc mọi context bucket. Kiểm cold shape/JIT qua diagnostic, thêm warmup harness ngoài timing khi cần và ghi chi phí setup. Không lấy latency của warmup làm dữ liệu.

Một process/GPU; có thể phân cells sang GPU đồng loại khi mỗi comparison vẫn ở cùng card/runtime. Không dùng launcher data-parallel nhiều process/card cho primary latency. Ghi GPU UUID, load nền, power mode và thời điểm; có contention thì đánh dấu cell, chạy lại toàn paired cell trước lock.

Ước tính từ pilot, không giả thời gian chạy:

```text
N_dev_requests/cell = N_dev_records × 3
N_final_online = N_dev_records × 3 × 10
N_fixed_search = N_dev_records × 3 × 16 × 3 variants
N_test = N_test_records × locked_repetitions × len(test_variants)
T_phase ≈ sum(cell_requests × measured_mean_e2e)
          + model_load/hash/warmup + calibration_replay + reporting
```

Dùng actual record counts; nhiều queries QMSum tăng compute, không tăng cluster count. Tính riêng capacity cho target KV + **full draft bank** + scratch. Sparse gather giảm attention work nhưng V1 không mặc định giảm resident VRAM. Pilot OOM thì dừng mở rộng, xử lý capacity rồi chạy lại cả cặp; không bỏ requests dài khỏi headline.

## 6. Gate và xử lý kết quả âm

| Gate | Tiêu chí | Hồ sơ |
|---|---|---|
| G0 | Assets/runtime/source mapping/shapes/collector hợp lệ; exposure đã audit | Preflight, corpus/split/exposure và runtime manifests |
| G1 | Full adapter/vendored/AR exact; cache và boundary không violation | Harness parity + paired output IDs trên GPU |
| G2 | Sparse RoPE/rollback/causal signals/protection đúng; target vẫn full | Harness invariants + sparse rollout token equality |
| G3 | Deployable selector-budget ≥1.05× full DFlash, lower CI95>1, E2E đủ overhead | Fixed-selector và A rollout; replay giải thích acceptance |
| G4 | B cost policy ≥1.03× best full gamma, lower CI95>1 | B paired comparisons; entropy so history để quy contribution |
| G5 | Joint ≥1.03× best fixed và independent, lower CI95>1 mỗi cặp; interaction evidence; frozen gain cùng chiều/lower CI>1 | Action surfaces, online/frozen comparisons |
| G6 | Tất cả locked cells complete, exact, cùng provenance; CI/quality/coverage được báo | Lock + test artifacts + gate review |

`headline_valid=true` chứng nhận tính hợp lệ của một comparison theo reporter; **không tự xác nhận ngưỡng gain hoặc mọi gate**. `lock` cũng không kiểm toàn bộ G0–G5. G0/G1/G2 fail thì dừng matrix. G3/G4/G5 fail thì vẫn có thể kết thúc nghiên cứu với kết quả âm hoặc claim hẹp đã đăng ký trước heldout; không báo contribution joint thành công.

Đối chiếu expected coverage với **toàn split và toàn danh sách cells đã đăng ký**, vì reporter chỉ thấy variants hiện diện và cohort count trong manifest. Một report hợp lệ trên pilot hoặc thiếu cả một variant chưa đủ G6.

## 7. Phân tích và hình/bảng cần xuất

| Artifact paper | Nội dung | Nguồn |
|---|---|---|
| Bảng chính | Per-dataset và macro speedup so AR/full/fixed/independent, CI95, exact coverage | Requests + paired reports |
| Bảng ablation | History/entropy/source relevance; online và frozen | Final dev/test cells |
| Hình action surface | E[commits], round cost, ms/commit theo B×gamma và state | Action replay; không phải rollout oracle |
| Hình scaling | E2E theo actual input/output bins, số groups/bin | Natural requests; fixed-token hình riêng |
| Hình overhead | Prefill, signal/controller/gather/bank/draft/verify, refresh/fallback | Request/round traces; không ép component sum=E2E |
| Bảng quality/memory | ROUGE natural, token equality, peak allocated/reserved GiB, bank bytes | Request records và diagnostic memory |

Reporter hiện tự tạo AR→mọi variant, full→A/B, fixed→joint, independent→joint. Các cặp `b_only_history→b_only_entropy`, fixed selector→full, signal ablations và per-dataset/component tables cần phân tích bổ sung từ requests bằng cùng `paired_comparison`/cluster bootstrap; không chia hai aggregate speedup hoặc hai CI để tạo paired CI mới. Instrumented/uninstrumented baseline control và toy sampling audit cần harness riêng, xem validation doc.

Cuối campaign, điền [results_template](results_template.md), ghi pass/fail/pending từng gate, artifact hashes, coverage và mọi deviation. Giữ error/mismatch trong hồ sơ; không có số liệu thì để `chưa đo`.
