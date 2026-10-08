# Rà soát triển khai Context-Adaptive DFlash — 2026-10-08

## Kết luận

**Chưa sẵn sàng chạy matrix thực nghiệm lớn.** Rà soát phát hiện **10 vấn đề: 7 P1 và 3 P2**, được tái hiện bằng **14 ca thất bại trong 45 ca kiểm tra CPU**. Có 31 ca đạt, xác nhận một phần quan trọng của adapter và greedy verifier trên mô hình nhỏ. Những kết quả này chưa thay thế kiểm chứng checkpoint thật hoặc các gate G0–G6.

P1 cần sửa trước khi mở matrix: lỗi chặn đường chạy, làm sai cấu hình/dữ liệu hoặc che giấu thất bại. P2 cần xử lý trước khi dùng kết quả để đánh giá contribution hay chốt heldout.

Rà soát này không sửa implementation. Nó bổ sung bộ kiểm tra, bằng chứng và tài liệu; các thay đổi AMR DFlash đang có trong workspace không thuộc phạm vi rà soát.

## Phạm vi và bằng chứng

- Đọc executor `src/TrainingFree/context_adaptive/`, CLI, runner và các hợp đồng canonical; đối chiếu trực tiếp với DFlash vendored.
- Runtime: Python 3.12.13, torch 2.13.0+cu130, transformers 5.12.1, pytest 9.1.1; chạy CPU, eager attention, FP32. Không tải checkpoint hoặc dùng mạng.
- Mô hình kiểm tra: Qwen3 target 6 layers, drafter DFlash 2 layers, hidden size 32, GQA 4 query heads/2 KV heads, block size 16. Trọng số ngẫu nhiên chỉ dùng để kiểm tra hợp đồng và parity; không đo chất lượng hoặc speedup.
- Kiểm tra launcher dùng master tạm và executable ghi argv. Kiểm tra pipeline chạy logic CLI thật, thay model loader bằng runtime CPU nhỏ. Các ca EOS/cap dùng target và proposal xác định để ép chính xác nhánh cần kiểm tra.
- Lần chạy toàn bộ: **45 ca, 31 đạt, 14 thất bại, exit code 1**, 13.36 giây. Không có ca bị skip hoặc lỗi setup.
- AST của 17 file Python và `bash -n` của runner/dispatcher đều đạt. SHA256 của 21 file nguồn/tham chiếu không đổi trong lần chạy toàn bộ.

Artifact nằm trong [`outputs/context_adaptive_dflash/implementation_review_20261008/`](../../../../outputs/context_adaptive_dflash/implementation_review_20261008/), được gitignore:

| Artifact | Nội dung |
|---|---|
| [test_review.py](../../../../outputs/context_adaptive_dflash/implementation_review_20261008/test_review.py) | Bộ kiểm tra 45 ca có thể chạy lại |
| [cpu_review_full.log](../../../../outputs/context_adaptive_dflash/implementation_review_20261008/cpu_review_full.log) | Traceback và kết quả toàn bộ |
| [cpu_review_full.xml](../../../../outputs/context_adaptive_dflash/implementation_review_20261008/cpu_review_full.xml) | JUnit từng ca |
| [review_evidence.json](../../../../outputs/context_adaptive_dflash/implementation_review_20261008/review_evidence.json) | Runtime, số ca, snapshot nguồn và giới hạn kiểm chứng |
| [source_snapshot.json](../../../../outputs/context_adaptive_dflash/implementation_review_20261008/source_snapshot.json) | SHA256 nguồn và Git HEAD; workspace đang có thay đổi chưa commit |
| [real_prepare/prepare_manifest.json](../../../../outputs/context_adaptive_dflash/implementation_review_20261008/real_prepare/prepare_manifest.json) | Prepare trên corpus LongBench đang có |

Các artifact này chỉ tồn tại tại workspace đã rà soát, không được commit cùng tài liệu. Khi chia sẻ báo cáo, cần giữ kèm bộ kiểm tra/log và snapshot.

## Các vấn đề theo mức ưu tiên

| ID | Mức | Vấn đề | Ca thất bại |
|---|---|---|---:|
| F1 | P1 | Hiểu sai `num_target_layers`, từ chối cấu hình DFlash hợp lệ | 1 |
| F2 | P1 | Round AR thiếu trường schema, CLI không ghi được baseline AR | 1 |
| F3 | P1 | Sampling verification block truyền tensor 3 chiều vào `multinomial` | 1 |
| F4 | P1 | Runner đưa giới hạn smoke vào phase dev/test từ CLI | 2 |
| F5 | P1 | Canonical caller overrides bị alias cũ của master lấn át | 4 |
| F6 | P1 | Resume thiếu dataset trong request key, bỏ qua mẫu khác dataset | 1 |
| F7 | P1 | Phase có toàn bộ request lỗi vẫn trả exit code 0 | 1 |
| F8 | P2 | Cost prior được dùng cho context ngoài support | 1 |
| F9 | P2 | Report thiếu paired comparisons với strong controls | 1 |
| F10 | P2 | Exposure chỉ có raw ID không khớp namespace của split | 1 |

### F1 — P1: kiểm tra cấu hình DFlash sai

Vị trí: [`scripts/infer_context_adaptive_dflash.py`](../../../../scripts/infer_context_adaptive_dflash.py), dòng 324; [`benchmark.py`](../../context_adaptive/benchmark.py), dòng 249.

Hai nơi yêu cầu `len(target_layer_ids) == num_target_layers`. Trong [`DFlash vendored`](../../../../externals/dflash/dflash/model.py), dòng 27–36, `num_target_layers` là tổng depth của target; hàm `build_target_layer_ids` chọn danh sách feature layers theo số draft layers. Target 6 layers, draft 2 layers cho `[1, 3]`, dùng được trong forward vendored và các ca parity, nhưng preflight mới từ chối với `DFlash target_layer_ids must be unique and match num_target_layers`.

**Ảnh hưởng:** cấu hình hợp lệ với số feature layers khác tổng target depth không qua preflight; `load_runtime` cũng có cùng điều kiện lỗi. Chưa nạp snapshot production để khẳng định riêng cấu hình checkpoint đó.

**Hướng sửa:** kiểm tra tổng target depth với target config, IDs không rỗng/không trùng/trong range, và kích thước feature projection thực tế. Không dùng tổng target depth làm số feature layers.

Bằng chứng: `test_valid_dflash_config_passes_model_preflight`.

### F2 — P1: AR baseline không đạt round schema

Vị trí: [`generation.py`](../../context_adaptive/generation.py), dòng 1126 và 1173; CLI validate ở dòng 960.

`generate_target_only` thiếu 8 trường bắt buộc: `parent_entropy`, `source_concentration`, `history_acceptance`, `selected_context_tokens_by_layer`, `physical_context_tokens_by_layer`, `eos_offset`, `predicted_cost_per_commit`, `wasted_work_ms`. Lệnh gọi `_complete_round_schema` với metadata `variant="ar"` lại nằm cuối `generate_adaptive`, dòng 1054.

**Ảnh hưởng:** AR request có ít nhất một decode round bị CLI chuyển thành error row trước khi ghi sample thành công. Output greedy có thể đúng nhưng baseline artifact không hợp lệ; không có paired AR report đầy đủ để lock.

**Hướng sửa:** hoàn tất schema trong `generate_target_only`, với các giá trị AR có nghĩa hoặc null được schema cho phép; giữ test đường CLI có nhiều hơn một output token.

Bằng chứng: `test_ar_rounds_pass_the_same_cli_schema`; pipeline ở F7 cũng tái hiện lỗi này trên 2 request.

### F3 — P1: sampling block bị crash

Vị trí: [`generation.py`](../../context_adaptive/generation.py), dòng 23–27; verification gọi `_sample(target_output.logits, temperature)` ở dòng 236.

Với `temperature > 0`, verification logits có shape `[1, block, vocab]` đi thẳng vào `torch.multinomial`, chỉ hỗ trợ tensor 1 hoặc 2 chiều. Fixture `[1, 4, 32]`, temperature 0.7 gây `RuntimeError: prob_dist must be 1 or 2 dim`.

**Ảnh hưởng:** speculative sampling không hoạt động khi tới verification. Greedy không đi vào nhánh này; prefill sampling 2 chiều cũng không phát hiện lỗi.

**Hướng sửa:** flatten các chiều trước vocab, sample rồi restore shape. Sau sửa vẫn cần audit phân phối sampling riêng theo protocol; ca này chỉ kiểm tra shape/runtime.

Bằng chứng: `test_sampling_supports_verification_block_logits`.

### F4 — P1: phase CLI bị gắn giới hạn smoke

Vị trí: [`run_context_adaptive_dflash.sh`](../../../../scripts/runners/run_context_adaptive_dflash.sh), dòng 60.

Runner quyết định thêm `--max-samples` bằng `${CAD_PHASE:-smoke}` trước khi xét `--phase` trong các arguments chuyển tiếp. Với master `RUN_SAMPLES=2`, gọi runner `--phase dev` hoặc `--phase test` đều thêm `--max-samples 2`.

**Ảnh hưởng:** dev bị cắt cohort ngoài ý định tại `select_split`; test có locked config hợp lệ vẫn bị từ chối vì không cho phép `--max-samples` (CLI dòng 559). Lệnh dev/test trong runbook bị ảnh hưởng khi chỉ đặt phase qua CLI và master có sample cap.

**Hướng sửa:** resolve phase cuối cùng trước khi tạo argv, hoặc để CLI áp dụng mặc định smoke sau khi parse phase. Không truyền cap mặc định smoke vào dev/test.

Bằng chứng: `test_launcher_does_not_inject_smoke_sample_cap_into_cli_phase[dev/test]`. Ca smoke riêng giữ sample cap đã đạt.

### F5 — P1: canonical overrides bị mất

Vị trí: [`run_context_adaptive_dflash.sh`](../../../../scripts/runners/run_context_adaptive_dflash.sh), dòng 33–37 và 47–62.

`fast_infer_load_config dflash` đã suy ra `TARGET_MODEL`, `DRAFT_MODEL`, `DATA_FILE`, `TEMPERATURE` từ master. Runner sau đó khôi phục caller values của `MODEL_TARGET`, `MODEL_DFLASH_DRAFT`, `DATA_INPUT`, `RUN_TEMPERATURE`; các alias đã suy ra không được tính lại và được ưu tiên khi tạo argv.

**Ảnh hưởng:** model, draft checkpoint, dữ liệu hoặc temperature thực chạy có thể là giá trị master dù caller đã override canonical variables. Cả 4 override đã được tái hiện độc lập bằng argv capture.

**Hướng sửa:** xác định precedence rõ ràng và tính alias sau khi khôi phục canonical caller values; giữ precedence cho alias được caller đặt rõ ràng.

Bằng chứng: 4 ca `test_launcher_preserves_canonical_caller_overrides`.

### F6 — P1: resume bỏ qua request khác dataset

Vị trí: [`scripts/infer_context_adaptive_dflash.py`](../../../../scripts/infer_context_adaptive_dflash.py), dòng 513–515, 888 và 1021–1027; error keys ở dòng 505 cũng cần đồng bộ.

Resume dùng `(sample_id, repetition)`, trong khi corpus đã có `_cad_record_key = dataset::sample_id`. Fixture gồm `d1/same` và `d2/same`: chạy đầu đủ 2 request thành công; mô phỏng ngắt sau `d1`, resume bỏ qua `d2` vì ID trùng.

**Ảnh hưởng:** mất request hợp lệ trong benchmark nhiều dataset; pruning round/token artifacts cũng dựa trên cùng key thiếu dataset. Summary có thể partial nhưng quá trình lại thoát thành công theo F7.

**Hướng sửa:** thống nhất `(dataset, sample_id, repetition)` trong completion/error tracking, execution và artifact pruning; validate identity khi đọc artifact cũ.

Bằng chứng: `test_resume_does_not_skip_same_id_from_another_dataset`.

### F7 — P1: thất bại nhưng exit code thành công

Vị trí: [`scripts/infer_context_adaptive_dflash.py`](../../../../scripts/infer_context_adaptive_dflash.py), dòng 1000–1018.

CLI tính summary `partial` và ghi error rows nhưng `_run_model_phase` luôn `return 0`. Pipeline CPU với 2 request AR, cap 4 token tái hiện `requested_samples=2`, `successes=0`, `errors=2`, `status=partial`, return code 0.

**Ảnh hưởng:** dispatcher hoặc automation dựa vào exit status có thể tiếp tục sau smoke/dev/test thất bại. Error JSONL có tồn tại nhưng không thay thế mã thoát đúng.

**Hướng sửa:** vẫn finalize artifact rồi trả nonzero khi phase thiếu request bắt buộc hoặc có lỗi; tổng hợp trạng thái qua mọi repetition.

Bằng chứng: `test_pipeline_returns_failure_when_every_request_errors`.

### F8 — P2: dùng cost ngoài context support

Vị trí: [`statistics.py`](../../context_adaptive/statistics.py), dòng 135–139; quy định ở [`design.md`](design.md), mục 5, dòng 110.

`cost_ms` backoff sang `context=*` và action-global. Calibration chỉ có context bucket 1, action cost 1 ms, nhưng state 10.000 token (bucket 5) vẫn nhận cost 1.1 ms sau controller cost. Đặc tả yêu cầu không dùng chi phí context ngắn cho context dài ngoài support.

**Ảnh hưởng:** controller có thể coi action chưa được profile ở context dài là đủ support và chọn gamma/budget bằng chi phí sai. Với `full`, chi phí draft/verify phụ thuộc context; fallback global không giữ điều kiện này.

**Hướng sửa:** chỉ dùng cost cùng context bucket được kiểm chứng; thiếu support trả `None` để fallback. Nếu muốn nội suy/ngoại suy cần bổ sung mô hình và calibration rõ ràng, sửa contract tương ứng.

Bằng chứng: `test_unseen_context_bucket_has_no_cost_prior`.

### F9 — P2: report chưa đánh giá được joint contribution

Vị trí: [`report.py`](../../context_adaptive/report.py), dòng 201–214; yêu cầu ở [`experiment_protocol.md`](experiment_protocol.md), G3/G4/G5.

`build_report` chỉ tạo comparisons từ AR tới từng variant. Dù đầu vào đủ AR, full DFlash, best fixed pair, independent A+B và joint, report không có `best_fixed_pair → joint` hoặc `independent_ab → joint`. Các so sánh A với full DFlash và B với best fixed gamma cũng chưa được tạo trực tiếp.

**Ảnh hưởng:** AR speedup không đủ đánh giá gain của adaptation so với các strong controls. Report hiện thiếu paired estimates/CI cần thiết cho G3–G5; trường `gate_status` luôn ở trạng thái pending.

**Hướng sửa:** khai báo các comparison chính theo protocol, dùng cùng kiểm tra coverage/exactness/provenance và source-cluster bootstrap. Báo từng gate bằng ngưỡng đã khóa và trạng thái đủ/thiếu bằng chứng.

Bằng chứng: `test_report_includes_joint_against_strong_controls`.

### F10 — P2: exposure raw IDs không được resolve

Vị trí: [`benchmark.py`](../../context_adaptive/benchmark.py), dòng 51–60; [`splits.py`](../../context_adaptive/splits.py), dòng 75 và 101–118.

Split nhận IDs dạng `dataset::sample_id`, nhưng exposure lookup lấy `sample_id` gốc và không chuẩn hóa namespace. Với registry chỉ có raw IDs, một mẫu đã exposure nằm trong test vẫn giữ nguyên test; ID được ghi unresolved nhưng manifest vẫn `complete`.

**Ảnh hưởng:** nguy cơ giữ document đã dùng phát triển ý tưởng trong heldout khi registry chỉ có IDs hoặc thiếu hash có thể khớp. Exposure chứa source hash tương ứng có thể được resolve bằng nhánh khác; không phải mọi manifest đều bị bỏ sót.

**Hướng sửa:** giữ raw ID và dataset trong split metadata, chuẩn hóa lookup theo hợp đồng exposure; raw ID không có dataset cần xử lý rõ khi trùng nhiều corpus. Unresolved IDs cần được phân loại thành ngoài corpus hoặc chưa resolve trước khi chốt heldout.

Bằng chứng: `test_id_only_exposure_resolves_ids_emitted_by_existing_probes`.

Prepare trên corpus thật có 10 exposure IDs unresolved. Đây chưa phải bằng chứng corpus thật đã bị leakage; ca thất bại F10 dùng dữ liệu tổng hợp có overlap được kiểm soát.

## Phần đã kiểm chứng đạt trên CPU

| Nhóm | Ca đạt | Phạm vi xác nhận |
|---|---:|---|
| Full adapter vs vendored | 4 | Draft logits và context K/V ở gamma 3/7/11/15 |
| Sparse adapter, original positions | 1 | Chọn vị trí 0/100/3000/9000; logits khớp forward vendored trên đúng feature subset và absolute positions |
| Greedy rollout | 12 | 3 selectors × 4 gammas, output 21 token khớp AR, output accounting và adaptive round schema đạt |
| Target-parent collector | 1 | Điểm attention khớp native eager target attention với GQA và causal mask |
| All-accept output caps | 5 | Cap 1/2/4/5/17; transaction accounting đúng trên fixture xác định |
| EOS/rejection transactions | 4 | EOS prefill, trong accepted prefix, correction pending và rejected EOS proposal |
| Prefix statistics | 1 | 2 unique states × 3 timing reps chỉ tính 2 outcome observations; survival đúng |
| Protected/chunk selection | 1 | Giữ protection và whole chunks; budget không khả thi trả `None` |
| Source-group/exposure bằng namespaced ID | 1 | Các query cùng source chung split; exposed source vào dev |
| Launcher smoke | 1 | Sample cap smoke được giữ |

Prepare metadata trên `data/longbench_100_14k` thành công: **500 records, 420 source groups**, chia **84 calibration / 84 dev / 252 test groups**. Không có model forward trong phase này; counts là group counts, không phải số request từng split.

Chưa kiểm chứng: nạp checkpoint thật, parity trên backend GPU production, CUDA timing/memory, E2E speedup, ROUGE thực, sampling distribution audit hoặc G0–G6. Các ca rollout ở đây kiểm tra fixed sparse pairs và refresh; không chứng minh chất lượng quyết định của joint controller trên workload thật.

## Chạy lại và thứ tự xử lý

Từ repo root trên máy local CPU:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m pytest \
  outputs/context_adaptive_dflash/implementation_review_20261008/test_review.py \
  -q --tb=short \
  --junitxml=outputs/context_adaptive_dflash/implementation_review_20261008/cpu_review_rerun.xml
```

Ở snapshot được rà soát, exit code dự kiến là 1 vì 14 ca đang thất bại. Bộ kiểm tra hiện có đường dẫn repo local cố định ở biến `ROOT`; khi chuyển sang workspace khác cần cập nhật biến này. Không cài profile server lên local và không cần tải model.

Thứ tự xử lý đề nghị:

1. Sửa F1/F2/F3 và chạy lại preflight, AR pipeline, block sampling fixtures.
2. Sửa F4/F5/F6/F7; yêu cầu argv đúng, resume đầy đủ mọi dataset và phase lỗi trả nonzero.
3. Sửa F8/F9/F10 trước calibration/lock/heldout để chi phí, comparisons và exposure đúng contract.
4. Chạy lại toàn bộ CPU suite rồi G0/G1/G2 trên checkpoint và GPU server thật. Chỉ mở matrix lớn sau khi có bằng chứng tương ứng; G3–G6 vẫn cần thực nghiệm theo protocol.

