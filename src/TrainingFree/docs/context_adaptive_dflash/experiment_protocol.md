# Protocol thực nghiệm Context-Adaptive DFlash

Ngày: **2026-10-08**. Trạng thái: **protocol thực nghiệm cho executor đã có; G0–G6 còn pending**. [Thiết kế](design.md), [measurement contract](measurement_contract.md), [runbook](runbook.md).

## 1. Research questions và evidence

| Mã | Câu hỏi | Bằng chứng cần thu |
|---|---|---|
| RQ-A | Selective context có giảm cost và giữ/tăng acceptance trên DFlash không? | Counterfactual prefix replay và real rollout, cùng checkpoint/budget |
| RQ-B | Entropy có sẵn trước round dự đoán accepted prefix và giúp chọn gamma không? | Heldout predictive comparison + wall-clock B-only so best fixed gamma |
| RQ-C | Budget làm thay đổi gamma tối ưu và joint policy có khai thác được không? | Cost surface theo state + joint so fixed/independent controls |
| RQ-S | Khi nào lợi ích đủ bù prefill, collector, gather và refresh? | E2E theo input/output regime, breakdown, CI và fallback rate |

Selective context có tiền lệ thực nghiệm; các câu hỏi trên xét chuyển giao và interaction trên pretrained DFlash. Không biến diagnostic attention coverage thành acceptance hoặc speedup.

## 2. Runtime và model

Primary: Qwen3-4B + Qwen3-4B-DFlash-b16, no thinking, frozen weights, batch 1, một process/GPU. Server Python 3.12, local snapshots, dependency đã mirror. Khóa actual target/draft attention backend và dtype trong manifest. Chạy baseline và variants trên cùng GPU/backend/sampler; không so target SDPA với target FA4 mà gán chênh lệch cho controller.

GPU thật ở server; máy dev local CPU chỉ dùng deterministic policy/schema/cache-math checks. Không tạo venv baseline riêng hoặc sửa stack CUDA local. Không deploy, download hay chạy cloud GPU từ protocol này.

## 3. Dữ liệu và split theo source document

Primary canonical: `data/longbench_100_14k/{gov_report,qmsum,multi_news}.jsonl`; server root xem [server_environment](../../../../docs/server_environment.md). Kiểm tra local metadata ngày 2026-10-08:

| Dataset | Records | Source groups theo normalized context |
|---|---:|---:|
| gov_report | 100 | 100 |
| qmsum | 100 | 20 |
| multi_news | 100 | 100 |

QMSum có nhiều query trên cùng conversation; không split/CI theo query như independent documents. `lcc`/`repobench-p` chỉ secondary generalization, không trộn vào summarization headline.

Quy tắc source grouping:

```python
import hashlib
import unicodedata

def source_fingerprint(context: str) -> str:
    canonical = " ".join(unicodedata.normalize("NFKC", context).split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def split_order(group_id: str, seed: int = 42) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()
```

Gom context hashes giống nhau; đồng thời merge variants có cùng `(dataset,source_split,source_index)` khi xác định cùng document. Truncation/length variants và nhiều queries giữ cùng group. Cross-dataset duplicate source groups chỉ được vào một split; không dedup chỉ bằng sample ID.

Sau global grouping, mỗi group thuộc allocation stratum là dataset có tên nhỏ nhất theo thứ tự lexicographic trong các records của group; tất cả records thừa hưởng cùng split. Trong từng stratum có N groups, reserve exposed groups vào dev trước. Sort các groups còn lại theo `split_order`; lấy `floor(0.2*N)` vào calibration, rồi thêm vào dev tới `max(exposed_count,floor(0.2*N))`, còn lại test. Yêu cầu ít nhất 1 group/split, thiếu quota thì báo `insufficient_data`, không chuyển exposed group vào test. Với pool không có exposure/cross-dataset duplicates: GovReport/Multi-News là 20/20/60 groups; QMSum là 4/4/12 groups. Số query records thực trong từng split phải ghi manifest, không mặc định QMSum cũng có 20/20/60 records.

Exposure registry nhập các cohort đã phân tích từ artifact source manifests, map cả source index và context fingerprint. Cohort DFlash 10 GovReport ngày 2026-10-06 có manifest tại `outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/manifest.json`; IDs có thể khác canonical nên resolve qua `data/longbench_200/gov_report.jsonl` và source metadata. Những group đã phát triển ý tưởng phải ở dev/exploratory, không ở untouched test. Nếu provenance không đủ để đối soát, không claim test chưa từng quan sát.

Khóa `split_manifest.json`: file hashes, mapping group→split, IDs, exposed registry và lý do loại. Không tự tạo synthetic docs để đủ quota; không có đủ nhóm thì báo `insufficient_data`. Chưa chạy build split hoặc khóa cohort trong lần hoàn thiện tài liệu này.

## 4. Natural summarization và scaling

Primary natural runs: cùng prompt/stopping/EOS, greedy, output cap 2048; không ép mọi summary đạt 2K. Báo actual input/output token distribution bằng tokenizer target. Raw `input_tokens` của dataset được build bằng tokenizer khác không thay measurement thực.

Workload 10K→2K là mục tiêu hệ thống. Để khảo sát output sweep 256/512/1024/2048 có thể dùng fixed-token performance run riêng, tắt EOS đồng nhất trong mọi control và ghi `quality_scope=not_natural_summary`. Không dùng output kéo dài sau EOS để kết luận chất lượng tóm tắt.

Input bins theo prompt thật: `[0,4K)`, `[4K,8K)`, `[8K,12K)`, `[12K,16K)`; canonical 14K không bảo đảm có mọi bin sau tokenizer target. Scaling 16K/32K cần mirror documents đủ dài và positional capacity thật; thiếu dữ liệu báo missing cell, không padding source để giả context dài. Mọi cắt source phải tạo đúng prompt/source spans và giữ grouping với document gốc.

## 5. Các giai đoạn và gate

Các ngưỡng speedup dưới đây là **tiêu chí nghiên cứu mặc định cần khóa trên dev**, không phải số liệu đã đo. Gate fail được ghi đúng, không sửa ngưỡng bằng test result.

### M0 — Preflight và parity, G0/G1

- G0: local asset signatures, source mapping, shapes, backend, collector capability và static schema checks đều hợp lệ.
- CPU: fixtures cho budget/protection, position math, prefix censoring, entropy, statistics và toy sampling; không benchmark GPU local.
- GPU smoke: 2 dev source groups/dataset, một query/group, tối đa 32 output tokens.
- G1: full adapter khớp vendored DFlash và greedy AR; target KV invariant không violation, EOS/cap accounting đúng. Chạy gamma 3/7/11/15 và gamma 0; unsupported shape được loại trước grid.
- Correctness expansion trước rollout lớn: ít nhất 20 dev source groups tổng (8 GovReport, 8 Multi-News, 4 QMSum nếu pool cho phép), cap 256. Không có mismatch hoặc cache corruption.

### M1 — A với fixed budget và prefix intervention, G2/G3

- Fixed gamma 15; budgets `{1024,2048,4096,full}`.
- So `target_parent`, `draft_refresh`, `recent_only`, `random`; lexical/cosine và current-draft oracle là diagnostic extensions.
- Chọn 8 dev source groups/dataset (QMSum dùng tối đa 4 dev groups), một query/group; state tại 0/64/128/256 processed output tokens nếu chưa EOS. Không ép state sau EOS.
- Với mỗi immutable prefix, rebuild/fork đúng target KV, pending anchor và draft bank. Lặp từng action trên state giống nhau; reset bank/collector scratch, không để action trước thay state sau. Clone/setup nằm ngoài microbenchmark round cost và được log riêng.
- Oracle current-draft cần dense forward bổ sung chỉ cho quality reference; không dùng oracle timing như selector deployable.
- G2: sparse positions/rollback/parent signals đúng và full-target output greedy vẫn exact.
- Real rollout trên toàn dev: tính cả score/gather/dense refresh. G3: ít nhất một deployable selector-budget đạt geometric paired E2E speedup >=1.05 so full DFlash và CI lower bound >1. Nếu chỉ attention coverage tốt nhưng rollout không nhanh hơn, A chưa qua gate.

### M2 — B với full context, G4

- Sweep fixed gamma đã hỗ trợ; chọn best fixed gamma bằng mean log E2E speedup trên dev, giữ cùng output scope.
- So history-only cost policy; entropy-threshold policy; entropy+history cost policy. Các policy cùng prior/cost accounting; threshold theo calibration quartiles, không theo token entropy sau verify của current block.
- History-only dùng action-global statistics + request EMA, không entropy bins. Entropy-threshold chọn gamma lớn/giữa/nhỏ theo calibration quartiles, phải ghi mapping cụ thể trước chạy; default bins 1/2→15, bin3→7, bin4→3.
- Phân tích `parent_entropy` và accepted length ở unseen dev groups, theo gamma/block position/context bucket. Token-level entropy sau verification là phân tích riêng.
- Đánh giá discrimination/calibration của predicted prefix survival và cost regret trên counterfactual states; bootstrap theo document. Không gộp rounds độc lập để phóng đại significance.
- G4: B cost policy đạt >=1.03 paired E2E speedup so best fixed gamma, CI lower >1; entropy+history cải thiện so history-only để giữ entropy trong contribution.
- Không pass entropy gate vẫn báo history-only result; không gọi entropy hữu ích chỉ từ correlation offline.

### M3 — C và interaction, G5

- Grid 4 budgets × supported gammas, mỗi gamma được **draft lại**; log `p_i(s,B,gamma)`.
- Calibration chạy 3 round repetitions/action/state sau warmup; prior prefix outcomes lấy từ actual verification, timings cùng hardware/signature. Với greedy, cùng state/action lặp lại chỉ là timing repetitions: prefix outcome được tính một lần, không tăng giả support. Ngoài các state M1, thu 512/1024/1536 processed tokens nếu natural trajectory đủ dài để bổ sung context/cost support. Không gọi neural fitting hoặc thay weights.
- Sau action profiling, đo controller `choose` của từng policy trên cùng causal state/action set với tables đã dựng. Shared action-cost priors không gồm controller; tổng J thêm overhead đúng policy. Total-round observations khác controller không được tái sử dụng như cùng signature.
- Lập cost surface/argmin gamma theo budget và generation state. Ước lượng local oracle gap chỉ trên state replay; không trình bày nó như E2E oracle rollout speedup.
- So joint với best fixed pair, A-only, B-only và independent A+B trong bảng dưới.
- Chạy joint/independent với cùng online-update mode; thêm cặp `statistics_update_mode=frozen` dùng cùng calibration tables/cost priors, history vẫn causal. Không relabel outcome của executed pair thành reference-axis observation. Frozen comparison phải cho thấy joint gain cùng chiều và CI lower >1 mới quy phần gain cho phối hợp action; nếu gain chỉ ở online mode thì phân tích như hiệu quả tổng hợp của policy và online statistics.
- G5: joint đạt >=1.03 paired E2E speedup so cả best fixed pair và independent A+B, CI lower >1 cho mỗi comparison; có evidence budget làm thay đổi gamma tối ưu vượt timing noise. Nếu thiếu interaction/gain thì thu hẹp claim về các thành phần riêng.

### M4 — Lock và heldout, G6

- Khóa selector/layer/refresh, protection, budgets/gammas, priors/cutpoints, margin, fallback và reporting trước test.
- Calibration priors reset mỗi request; không dùng test outcomes của request trước để cải thiện request sau trong primary matrix.
- 3 timed repetitions/request/variant, warmup 3 requests ngoài measured set; order variants shuffle seed 42/43/44, cùng data IDs mỗi repetition.
- G6: đủ toàn bộ locked cells, không missing/error/mismatch bị âm thầm loại; exactness checks pass và paired 95% CI được báo. Chưa đủ coverage thì report partial, chưa có headline claim.
- Chạy sampling correctness audit riêng với toy exact enumeration và ít nhất 10.000 independent draws cho toy states; ngưỡng kiểm định và multiple-comparison correction khóa trước audit. Exactness dựa trên algorithm proof + implementation audit, empirical test không tự chứng minh equality tuyệt đối.

## 6. Định nghĩa baseline và ablation

| ID | Context | Gamma | Policy |
|---|---|---|---|
| `ar` | Full target | 0 | Cùng target/sampler/kernel |
| `dflash_full_fixed` | Full draft | Best full fixed gamma; thêm vanilla 15 | Controller disabled |
| `best_fixed_pair` | Fixed selector/budget | Fixed | Joint grid tuning chỉ trên dev |
| `a_only` | Adaptive B | Best fixed gamma | Joint cost rule xét một gamma |
| `b_only_history` | Full | Dynamic | History-only cost rule |
| `b_only_entropy` | Full | Dynamic | Parent entropy + history cost rule |
| `independent_ab` | Adaptive B | Dynamic | Hai quyết định độc lập như định nghĩa bên dưới |
| `joint` | Adaptive B | Dynamic | Cùng argmin trên `(B,gamma)` |

Independent A+B: context controller chọn B bằng cost/acceptance model tại **fixed dev gamma reference**, không nhận gamma hiện chọn của B-controller. Gamma controller chọn gamma bằng model calibrated **full context**, không nhận B hiện chọn. Cả hai đọc causal state trước round, gộp actions rồi execute; refresh/boundary constraints giống joint và phải log override. Không cho independent controller lặp coordinate descent đến hội tụ vì khi đó nó đã dùng joint interaction.

Quan sát sau round chỉ cập nhật table tại executed `(B,gamma)` cho cả joint và independent. Reference-axis models không được nhận pseudo-label từ sparse/khác-shape outcomes; frozen ablation ở M3 dùng cùng priors cho hai controller và khóa mode trong manifest/locked config.

Selector và score-refresh schedule giữ giống nhau giữa best fixed/A/independent/joint trong một comparison. B-only không cần source collector; cost difference collector là một phần thực của A/C. Thêm `joint_no_entropy` và `joint_no_source_relevance` để xác định đóng góp signals.

SpecExtend/SparseSpec-L là external context baselines nếu có matched checkpoints/runtime/assets. LibraSpec-style `verify_prefix` là strong length baseline nếu có implementation parity. Không có asset tương thích thì báo unavailable; không dùng model khác làm primary paired speedup rồi gán cho algorithm.

## 7. Fairness, observability và report

- Cùng input IDs/hash, target/draft weights, dtype, backend, output cap, EOS, temperature, thinking và batch 1. Không cộng benefit prefill approximation vào claim joint policy.
- Collector overhead dùng same target output kernel; counterpart full-context có control instrumented và uninstrumented để tách diagnostic distortion.
- Không chạy nhiều process cùng GPU trong primary latency experiment. Throughput/concurrency là protocol riêng.
- Chỉ warmup/JIT/model load ngoài measured steady-state scope; bootstrap/refresh của từng request vẫn tính E2E.
- Bắt đầu/kết thúc mỗi phase/cell có message; progress theo request và heartbeat tối đa 60 giây khi đang chạy. Error row ghi component, flush logs, tiếp tục request độc lập nếu CUDA context còn hợp lệ.
- Report E2E/TTFT/TPOT/components, committed tokens/round, acceptance, action distribution, fallback/refresh rates, allocated/reserved memory, correctness và ROUGE natural-output.
- Mọi claim speedup đối soát sample coverage, paired token equality, signature/timing scope và CI theo measurement contract.

## 8. Deliverable của mỗi milestone

M0: preflight/correctness report. M1: intervention trace + real A rollout + G3 decision. M2: entropy/history analysis + B rollout + G4 decision. M3: action surfaces + joint/independent comparison + G5 decision. M4: locked manifests + completed heldout request/round streams + G6 report.

Trạng thái ban đầu của **mọi gate G0–G6 là pending** đối với executor mới. Dataset metadata inspection và prior attention reports không thay các gate này.
