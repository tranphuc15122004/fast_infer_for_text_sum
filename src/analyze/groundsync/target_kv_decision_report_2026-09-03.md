# Báo cáo kiểm định Target-KV Conditioned Block Drafting

Ngày thực hiện: 2026-09-03  
Thư mục mã nguồn và artifact: `src/analyze/groundsync/`  
Trạng thái tổng hợp: **E0 context-drop chưa đủ điều kiện kết luận; E1 không ủng hộ KV-specific advantage; không mở E2/E3**.

## 1. Kết luận điều hành

Nhánh thí nghiệm này được tách khỏi các kết quả GroundSync/BurstSpec P0 đã có
trong [`p0_final_report_2026-09-02.md`](p0_final_report_2026-09-02.md). Mục tiêu
của nhánh mới là kiểm tra trực tiếp hai tiền đề của Target-KV:

1. DFlash có còn acceptance/survival khi context tăng hay không (E0).
2. Target KV có chứa tín hiệu đủ tốt để dự đoán block token kế tiếp, vượt qua
   token-wise hidden sequence control hay không (E1).

Quyết định hiện tại:

| Experiment | Kết quả | Quyết định |
|---|---|---|
| E0 short/mid-context acceptance | `accepted_draft_tokens=0` ở tất cả round quan sát trên GovReport, Multi-News và CNN/DailyMail | Zero-admission được tái lập trong phạm vi đã chạy |
| E0 context-drop | Không có bucket `8-16K` trở lên trong run FP16 chính | **INCONCLUSIVE**, không được gọi là context-drop FAIL |
| E0 max-new confirmation | 5 document/dataset, `max_new_tokens=32`, mọi round vẫn 0 draft token | Củng cố zero-admission; coverage dài vẫn thiếu |
| E1 representation probe | KV không vượt hidden sequence ở GovReport hoặc Multi-News | **FAIL đối với claim KV-specific trên bằng chứng hiện có** |
| E2 factorial / E3 adapter | Chưa chạy | Đúng gate; chưa có cơ sở để mở rộng |

Diễn giải khoa học chính xác là: cặp Qwen3-4B + Qwen3-4B-DFlash-b16 hiện
không tạo ra acceptance opportunity trong các prefix FP16 chạy được trên T4;
đồng thời probe matched-budget không tìm thấy lợi thế của KV so với
token-wise hidden. Tuy vậy, chưa thể tuyên bố Target-KV không thể có lợi ở
context 8--40K FP16 vì T4 không chạy được coverage dài tự nhiên ở chế độ này.

## 2. Hypothesis, metric và gate

### E0 — DFlash failure map

Với mỗi proposal start `t`, DFlash ghi `acceptance_lengths` theo round. DFlash
luôn cộng token target fallback vào độ dài này, do đó metric dùng trong report là:

```text
accepted_draft_tokens = acceptance_length - 1
```

Các metric chính:

```text
S_L(j) = P(accepted_draft_tokens >= j | context bucket L)
MAT    = mean(accepted_draft_tokens)
```

`S_L(j)` và `MAT` được bootstrap ở document level; round trong cùng một document
không được coi là các document độc lập. Context bucket được lấy từ độ dài prompt
thực tế sau chat template, không padding nhân tạo và không truncate im lặng.

Gate E0 yêu cầu có survival drop theo context ở K=8/16 với đủ bucket dài hoặc
interaction context × depth có bằng chứng. Nếu bucket dài không xuất hiện thì
trạng thái bắt buộc là `INCONCLUSIVE`.

### E1 — Representation sufficiency

Target Qwen3-4B được freeze. Tại mỗi anchor/document, target sinh greedy label
16 token tiếp theo. Các representation được đưa qua cùng interface `128×64` và
cùng một `MemoryBlockProbe`:

- `hidden`: hidden state của token cuối prefix;
- `hidden_sequence`: hidden sequence toàn prefix;
- `multi_layer_hidden`: hidden sequence tại layer `[1, 9, 17, 25, 33]`;
- `kv`: K/V sequence ở cùng các layer;
- `kv_shuffled`: negative control xáo trộn trục token của KV;
- `kv_recent`: chỉ giữ 1/4 KV gần nhất;
- `kv_wrong_document`: KV của document khác, negative control.

Gate E1 yêu cầu KV vượt `hidden_sequence` ở ít nhất hai regime hoặc có CI rõ
ràng, với cùng ngân sách trainable. Nếu KV chỉ hơn `hidden` nhưng xấp xỉ hoặc
kém `hidden_sequence`, kết luận là tín hiệu token sequence đã đủ hoặc KV chưa
được chứng minh cần thiết.

## 3. Môi trường thực nghiệm

### Máy và runtime

- Host: `tuantb@teslaT4`.
- GPU: NVIDIA Tesla T4, compute capability `7.5`, tổng VRAM `15360 MiB`
  (`14.568 GiB`).
- Driver: `550.163.01`; CUDA driver/runtime nhìn thấy: `12.4`.
- GPU executable: `/home/tuantb/miniconda3/bin/python3`.
- GPU run được thực hiện **ngoài `.venv`**, `CUDA_VISIBLE_DEVICES=0`, batch size 1.
- Torch: `2.6.0+cu124`.
- Python thực tế của GPU environment: `3.13.9` (được ghi trong manifest); đây
  là runtime Miniconda của máy, không phải production Python 3.12 theo tài liệu
  server.
- Attention: `sdpa`; dtype canonical: `float16` vì T4 không có BF16 native phù
  hợp cho setup này.
- Prefill: chunked causal prefill với `prefill_chunk_size=128` trong các run
  Target-KV. Đây là biện pháp giảm peak memory, không thay đổi thứ tự token hay
  metric.

### Model/cache local

- Target:
  `/home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`.
- DFlash drafter:
  `/home/tuantb/.cache/huggingface/hub/models--z-lab--Qwen3-4B-DFlash-b16/snapshots/61ab4992e5b5ec5913c7f8a9618367b4309533a3`.
- Cả hai loader dùng `local_files_only=True`; không tải model qua mạng.
- Target config: 36 decoder layers, hidden size 2560, 32 attention heads, 8 KV
  heads, head dimension 128, model limit 40960 token.
- DFlash config: block size native 16, target layers `[1,9,17,25,33]`.

## 4. Cách triển khai và các sửa cần thiết

### E0 runner

[`e0_dflash_failure_map.py`](e0_dflash_failure_map.py) thực hiện các bước:

1. Đọc local JSONL, render prompt theo schema dataset và tokenize bằng target
   tokenizer.
2. Lọc riêng các row vượt model limit hoặc T4 experimental cap; ghi lý do vào
   `exclusions.jsonl`.
3. Chạy target AR baseline và DFlash với K độc lập `{4,8,16}` trên cùng input.
4. Dùng target prefill chunking với bottom-right causal mask.
5. Dùng `SelectiveHiddenTarget` hook để chỉ giữ các target layer DFlash yêu cầu;
   không yêu cầu Transformers materialize toàn bộ hidden-state tuple.
6. Ghi acceptance từng round, timing, peak VRAM và `exact_match_target_ar`.
7. Flush raw row sau từng K; vì vậy run dài vẫn để lại artifact audit nếu bị
   dừng.

[`target_kv_experiments.py`](target_kv_experiments.py) chuẩn hóa fallback token,
context bucket, document bootstrap và decision gate. [`e0_report.py`](e0_report.py)
tạo Markdown report theo run.

### E1 extraction và probe

[`e1_representation_probe.py`](e1_representation_probe.py) thực hiện target
prefill tại các anchor, sinh 16 greedy labels, tạo bảy representation/control,
pool về `128×64`, rồi lưu `features.npz` và `metadata.jsonl`. Split train/dev/test
được thực hiện theo `document_id`, nên hai anchor của cùng document luôn nằm
cùng partition.

Trong quá trình chạy phát hiện hai lỗi runtime và đã sửa:

- hidden chunks được chuyển về CPU ngay sau forward, tránh giữ thêm sequence
  activation trên T4 trong khi target KV cache còn sống;
- toàn bộ extraction được bọc `torch.inference_mode()`, tránh dựng autograd graph
  qua cache.

Nếu không có hai guard này, E1 bị OOM giả tạo ngay cả ở prefix 4K. Sau sửa,
E1 chạy được 4K GovReport và 8K Multi-News FP16.

## 5. E0 — coverage và kết quả

### 5.1 Các run chính

| Run | Input/cấu hình | Document thành công | K-record | Round-record | Bucket quan sát được | Decision |
|---|---|---:|---:|---:|---|---|
| [`tkv-e0-pilot-gov-20260903`](results/tkv-e0-pilot-gov-20260903/) | 40 source rows đầu, cap 8192, max-new 8 | 13 | 39 | 312 | 2--4K: 3; 4--8K: 10 | `INCONCLUSIVE` |
| [`tkv-e0-pilot-multinews-20260903`](results/tkv-e0-pilot-multinews-20260903/) | 50 source rows, cap 8192, max-new 8 | 49 | 147 | 1176 | 0--2K: 27; 2--4K: 15; 4--8K: 7 | `INCONCLUSIVE` |
| [`tkv-e0-pilot-cnn-20260903`](results/tkv-e0-pilot-cnn-20260903/) | 30 source rows, cap 8192, max-new 8 | 30 | 90 | 720 | 0--2K: 30 | `INCONCLUSIVE` |

`K-record` là số document × 3 K; `round-record` là số round acceptance sau khi
chạy generation. Tất cả generation row của ba run đều `status=ok`.

### 5.2 Survival/MAT

Trong mọi bucket có quan sát và mọi K `{4,8,16}`:

```text
MAT = 0.0000
S(1) = 0.0000
```

Do `S(1)=0`, các survival sâu hơn cũng bằng 0. Điều này có nghĩa là mọi round
quan sát đều chỉ commit target fallback, không có draft token nào được accepted.
Không có khác biệt survival giữa 0--2K, 2--4K và 4--8K trong dữ liệu hiện có;
nhưng chưa được phép suy ra “không có context effect” vì chưa có 8--16K trở lên.

Chi tiết do analyzer ghi:

- GovReport: `2-4k` có 3 document, `4-8k` có 10 document; cả hai đều
  `MAT=0`, `S(1)=0`.
- Multi-News: 27/15/7 document tương ứng ba bucket 0--2K/2--4K/4--8K; cả
  ba đều `MAT=0`, `S(1)=0`.
- CNN/DailyMail: 30 document ở 0--2K; `MAT=0`, `S(1)=0`.

### 5.3 Confirmation max-new 32

Để tránh khả năng max-new 8 không đủ biểu diễn persistence của block, đã chạy:

| Run | Document | K | max-new | Round-record | Số round có draft acceptance |
|---|---:|---|---:|---:|---:|
| [`tkv-e0-confirm32-gov-20260903`](results/tkv-e0-confirm32-gov-20260903/) | 5 | 4/8/16 | 32 | 480 | 0 |
| [`tkv-e0-confirm32-multinews-20260903`](results/tkv-e0-confirm32-multinews-20260903/) | 5 | 4/8/16 | 32 | 480 | 0 |

Mọi round của hai run có `accepted_draft_tokens=0`. Đây là confirmation cho
zero-admission qua 32 output token, nhưng không thay thế long-context coverage.

### 5.4 Đối chiếu runner chính thức

Để loại trừ lỗi trong memory-safe wrapper, đã chạy runner DFlash chính thức trên
1 CNN/DailyMail row ngắn. Artifact:
[`tkv-dflash-official-crosscheck-20260903.jsonl`](results/tkv-dflash-official-crosscheck-20260903.jsonl).

Kết quả:

```text
acceptance_lengths = [1,1,1,1,1,1,1,1]
```

Theo định nghĩa DFlash, đây là 0 accepted draft token ở cả 8 round. Vì vậy
zero-admission của E0 wrapper phù hợp với runner chính thức, không phải do trừ
fallback sai.

### 5.5 Long-context feasibility riêng

FP16 target trên T4 không đạt được coverage tự nhiên 16--40K ổn định cho E0/E1
chính. Đã chạy riêng quantized target 8-bit để xác nhận khả năng cấp phát:

| Input thực tế | Peak VRAM | DFlash result | Vai trò |
|---:|---:|---|---|
| 11052 | 8.08 GiB | 8 round, tất cả raw length 1 | feasibility only |
| 16384 | 9.50 GiB | 8 round, tất cả raw length 1 | feasibility only |
| 28156 | 12.65 GiB | 8 round, tất cả raw length 1 | feasibility only |
| 41651 | — | loại vì vượt model limit 40960 | exclusion đúng |

Các row 8-bit không được gộp vào kết luận canonical FP16. Chúng chỉ cho thấy
T4 có thể chạy một số context dài khi đổi precision; chúng không cung cấp phép
so sánh công bằng với E0 FP16.

### 5.6 Exact-match guardrail

E0 kiểm tra output speculative có trùng target AR không:

| Run | Exact-match |
|---|---:|
| GovReport pilot | 36/39 |
| Multi-News pilot | 147/147 |
| CNN pilot | 90/90 |
| GovReport max-new 32 | 12/15 |
| Multi-News max-new 32 | 15/15 |

Ba row lệch của GovReport đều thuộc cùng một document
`gov_report_42094bc4d2f5e1d0` và lặp lại ở K=4/8/16. Debug rerun đã lưu token
IDs cho row này: bảy token đầu trùng, token cuối khác (`8397` so với `29340`).
Vì acceptance vẫn bằng 0, điều này không làm thay đổi quan sát admission;
nhưng các row này không được dùng cho claim speed/quality. Các run mới lưu cả
`output_token_ids` và `baseline_token_ids` để audit tiếp tục.

### 5.7 Quyết định E0

**E0 short/mid-context observation: có bằng chứng mạnh về zero-admission trong
phạm vi đã chạy.**  
**E0 long-context context-drop gate: `INCONCLUSIVE`.**

Không được ghi E0 là `PASS` cho long-context KV advantage và cũng không được ghi
`FAIL` chỉ vì thiếu bucket dài. Kết luận thực dụng là cặp DFlash này không đáng
mở E2/E3 trong short/mid regime trên T4; claim long-context cần GPU lớn hơn và
FP16 rows dài hơn.

## 6. E1 — coverage và kết quả

### 6.1 Coverage

| Run | Cap FP16 | Documents | Anchors | Excluded | Split train/dev/test |
|---|---:|---:|---:|---:|---|
| [`tkv-e1-gov-4k-20260903`](results/tkv-e1-gov-4k-20260903/) | 4096 | 17 | 34 | 153 source rows trong lát cắt | 20/6/8 anchors |
| [`tkv-e1-multinews-20260903`](results/tkv-e1-multinews-20260903/) | 8192 | 49 | 98 | 1 | 58/18/22 anchors |

GovReport cần cap 4K vì các prefix dài hơn làm FP16 target prefill vượt VRAM
T4; Multi-News chạy được cap 8K sau khi loại autograd graph. Các số trên là
coverage feature cache, không phải số document trong test partition.

Tất cả bảy representation dùng đúng `158,021,760` trainable parameters,
`horizon=16`, `interface=128×64`, và cùng document-disjoint split trong từng
run.

### 6.2 Kết quả GovReport 4K

CE thấp hơn là tốt hơn; Acc là trung bình theo 16 vị trí; `prefix@8` là xác
suất đúng trọn prefix 8 token.

| Representation | CE | Acc@1 | Acc@5 | Prefix exact @8 | Parameters |
|---|---:|---:|---:|---:|---:|
| `hidden` | 11.7570 | 0.0625 | 0.0625 | 0.0000 | 158021760 |
| `hidden_sequence` | **11.7173** | 0.0312 | **0.0703** | 0.0000 | 158021760 |
| `multi_layer_hidden` | 11.8831 | 0.0469 | 0.0469 | 0.0000 | 158021760 |
| `kv` | 11.9097 | 0.0000 | 0.0312 | 0.0000 | 158021760 |
| `kv_shuffled` | 11.8993 | 0.0078 | 0.0078 | 0.0000 | 158021760 |
| `kv_recent` | 11.9146 | 0.0000 | 0.0078 | 0.0000 | 158021760 |
| `kv_wrong_document` | 11.8871 | 0.0781 | 0.0781 | 0.0000 | 158021760 |

KV kém `hidden_sequence` khoảng `0.1924` CE và không có exact prefix 8. Run
GovReport nhỏ nên đây là directional evidence, nhưng hướng hiệu ứng không ủng
hộ KV-specific claim.

### 6.3 Kết quả Multi-News 8K

| Representation | CE | Acc@1 | Acc@5 | Prefix exact @8 | Parameters |
|---|---:|---:|---:|---:|---:|
| `hidden` | 11.7273 | 0.0227 | 0.0426 | 0.0000 | 158021760 |
| `hidden_sequence` | **11.6988** | 0.0341 | 0.0653 | 0.0000 | 158021760 |
| `multi_layer_hidden` | 11.8607 | 0.0284 | **0.0682** | 0.0000 | 158021760 |
| `kv` | 11.8899 | 0.0170 | 0.0426 | 0.0000 | 158021760 |
| `kv_shuffled` | 11.8718 | 0.0227 | 0.0597 | 0.0000 | 158021760 |
| `kv_recent` | 11.8892 | 0.0057 | 0.0341 | 0.0000 | 158021760 |
| `kv_wrong_document` | 11.8646 | 0.0369 | 0.0682 | 0.0000 | 158021760 |

KV kém `hidden_sequence` khoảng `0.1911` CE. `kv_shuffled` và
`kv_wrong_document` không làm mất tín hiệu theo cách cần thiết để chứng minh
KV alignment là nguồn thông tin riêng; thậm chí một số metric của control cao
hơn KV do sample nhỏ. Không representation nào đạt prefix exact 8.

### 6.4 Quyết định E1

E1 **FAIL đối với gate “KV vượt token-wise hidden ở hai regime”**: KV không
vượt `hidden_sequence` ở GovReport hoặc Multi-News. Đây chưa phải chứng minh
toán học rằng một kiến trúc probe lớn hơn không thể khai thác KV; nó là kết quả
đủ để không mở E2/E3 adapter/factorial trong phạm vi nguồn lực hiện tại.

## 7. Validation và tính tái lập

### Unit/static validation

- Target-KV tests: **28 passed** trong lần audit sau cùng của riêng các module
  E0/E1/report.
- Toàn bộ tests dưới `src/analyze/groundsync/tests/`: **94 passed**.
- `python3 -m compileall -q src/analyze/groundsync`: **PASS**.
- `git diff --check -- src/analyze/groundsync`: **PASS**.
- Full repository suite đã được thử: `64 passed, 1 failed`. Failure là test cũ
  `tests/test_runtime_compat_fixes.py::test_llama31_rope_theta_is_recovered_from_modern_config`
  trong `common/model_compat.py`, không thuộc thay đổi Target-KV; vì vậy không
  được ghi toàn repository là pass.

### Artifact map

| Thành phần | File/artifact |
|---|---|
| E0 schema/metrics | [`target_kv_experiments.py`](target_kv_experiments.py) |
| E0 model runner | [`e0_dflash_failure_map.py`](e0_dflash_failure_map.py) |
| E0 report renderer | [`e0_report.py`](e0_report.py) |
| E1 pooling/probe | [`target_kv_e1.py`](target_kv_e1.py), [`e1_representation_probe.py`](e1_representation_probe.py) |
| E1 report renderer | [`e1_report.py`](e1_report.py) |
| Tests | [`tests/test_target_kv_experiments.py`](tests/test_target_kv_experiments.py), [`tests/test_target_kv_e1.py`](tests/test_target_kv_e1.py), [`tests/test_target_kv_reports.py`](tests/test_target_kv_reports.py) |
| E0 raw/metrics | Các thư mục `results/tkv-e0-*/` nêu ở trên |
| E1 features/probes | Các thư mục `results/tkv-e1-*/` nêu ở trên |

### Lệnh tái lập tiêu biểu

E0 pilot:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/tuantb/miniconda3/bin/python3 -m src.analyze.groundsync.e0_dflash_failure_map \
  --target-model /home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --draft-model /home/tuantb/.cache/huggingface/hub/models--z-lab--Qwen3-4B-DFlash-b16/snapshots/61ab4992e5b5ec5913c7f8a9618367b4309533a3 \
  --data-file data/longbench_200/multi_news.jsonl --max-samples 50 \
  --max-new-tokens 8 --input-token-limit 8192 --prefill-chunk-size 128 \
  --output-dir src/analyze/groundsync/results/tkv-e0-pilot-multinews-20260903
```

E1 extraction và train được ghi trực tiếp trong `manifest.json` của từng run;
training sử dụng:

```bash
/home/tuantb/miniconda3/bin/python3 -m src.analyze.groundsync.e1_representation_probe \
  --mode train --output-dir src/analyze/groundsync/results/tkv-e1-multinews-20260903 \
  --interface-dim 64 --hidden-dim 64 --horizon 16 --epochs 2 --batch-size 4 \
  --device cuda:0
```

## 8. Quyết định tiếp theo

Không chạy tiếp các hạng mục sau trong nhánh này tại T4 hiện tại:

- E2 factorial K/context/representation;
- E3 train DFlash-KV adapter;
- full 32--40K FP16 claim;
- production vLLM/controller benchmark.

Nếu cần đóng câu hỏi long-context thay vì dừng hướng này, yêu cầu tối thiểu là
GPU lớn hơn để chạy target Qwen3-4B FP16 với natural buckets `8-16K`, `16-32K`
và `32-40K`, giữ nguyên DFlash checkpoint, seed, greedy decoding và paired
AR guardrail. Khi đó phải rerun E0 từ đầu; không gộp 8-bit feasibility với FP16.

Với bằng chứng hiện có, quyết định nghiên cứu là:

```text
NO-GO E2/E3 FOR TARGET-KV ON CURRENT T4 EVIDENCE
KEEP E0 LONG-CONTEXT CLAIM INCONCLUSIVE
DO NOT INVEST IN KV-SPECIFIC ADAPTER BEFORE LONG-FP16 REPLICATION

## 9. Quá trình thiết kế và xây dựng thực nghiệm

### 9.1. Câu hỏi nghiên cứu được chốt lại

Sau khi các kết quả GroundSync/BurstSpec trước đó cho thấy oracle có headroom nhưng estimator chưa chứng minh được utility, hướng nghiên cứu được chuyển sang **Target-KV Conditioned Block Drafting**. Hướng này có hai câu hỏi tách biệt:

1. **E0 — feasibility của hiện tượng:** khi target model nhận thêm một prefix dài hơn, acceptance của DFlash có sống sót ở những vị trí tương ứng hay không? Nếu acceptance đã bằng 0 ngay ở prefix ngắn, việc xây dựng bộ điều khiển dựa trên target KV không có tín hiệu để khai thác.
2. **E1 — giá trị của thông tin KV:** với cùng target hidden state, hidden sequence và các điều kiện kiểm soát, target KV có giúp dự đoán suffix/block tốt hơn không? Đây là kiểm định trực tiếp cho claim “KV conditioning có thông tin bổ sung so với token-wise hidden”.

Việc tách E0 và E1 là quyết định thiết kế quan trọng. E0 kiểm tra **tín hiệu tồn tại trong hệ thống sinh thật**; E1 kiểm tra **giá trị biểu diễn trong một probe có kiểm soát**. Không dùng E1 để bù cho một E0 thất bại, và không dùng acceptance trên một probe để tuyên bố speedup serving.

### 9.2. Các quyết định thiết kế trước khi viết code

Các ràng buộc sau được cố định trước khi chạy chính thức:

| Hạng mục | Quyết định | Lý do |
|---|---|---|
| Target/drafter | Qwen3-4B + Qwen3-4B-DFlash-b16 local cache | Đúng cặp model đã có trên máy, không phụ thuộc internet |
| E0 block sizes | (K\in\{4,8,16\}) | Bao phủ block ngắn, trung bình và block native 16 của DFlash |
| E0 metric chính | (A=\max(0,\text{raw acceptance length}-1)) | Loại token target fallback luôn được tính trong API DFlash |
| E0 protocol | max-new 32 cho kết luận acceptance; context-drop dài chỉ là feasibility | Tránh nhầm failure của `max_position_embeddings` với failure của hypothesis |
| Context | Không padding, không truncation im lặng | Giữ đúng độ dài ngữ cảnh và tránh làm thay đổi phân phối token |
| Dataset chính | GovReport và Multi-News | Một nguồn single-document dài và một nguồn multi-document |
| Dataset control | CNN/DailyMail | Kiểm tra ngữ cảnh ngắn, không dùng làm bằng chứng long-context |
| E1 labels | Target suffix tokens tại cùng anchor | Đo đúng nhiệm vụ dự đoán nội dung kế tiếp |
| E1 fairness | Cùng sample, anchor, horizon, optimizer budget và decoder | Mọi probe chỉ khác thông tin đầu vào |
| E1 gate | KV phải thắng token-wise hidden trong ít nhất 2 regime hoặc có CI dương | Tránh chọn một thắng lợi do seed hoặc một dataset |
| Statistical unit | Document-level bootstrap | Không coi các anchor trong cùng một tài liệu là độc lập |

### 9.3. Các giai đoạn xây dựng

Quá trình được triển khai theo thứ tự sau:

1. **Khảo sát repository và cache:** xác định runtime hệ thống, model snapshot, cấu hình target/DFlash, schema JSONL và các báo cáo GroundSync trước đó.
2. **Đóng băng protocol:** ghi thiết kế E0/E1 vào plan trước khi chạy; quy định metric, bucket, gate, exclusion và các điều kiện dừng.
3. **Xây E0 runner:** đọc JSONL, tokenize, lọc input hợp lệ, chạy target AR và DFlash cho từng (K), thu acceptance, thời gian, peak memory, token ids và trạng thái exact-match.
4. **Xây E0 analyzer:** tính survival (S_L(j)), mean accepted tokens (MAT), bootstrap theo document, bảng bucket và kiểm tra context-drop.
5. **Xây E1 extraction:** lấy target hidden/KV tại các anchor hợp lệ, chuyển tensor trung gian cần thiết về CPU để tránh giữ nhiều graph/GPU tensor.
6. **Xây E1 probe:** dùng chung một decoder head nhỏ, cùng horizon và số bước tối ưu; chỉ thay đổi representation (`hidden`, `hidden_sequence`, `multi_layer_hidden`, `kv` và controls).
7. **Chạy pilot → smoke → confirmation:** kiểm tra trước trên tập nhỏ, sau đó chạy các quy mô đã định; mọi kết quả đều lưu JSONL/JSON và có report máy sinh tự động.
8. **Audit và tổng hợp:** chạy unit test, compile, diff check, artifact audit; sau đó đối chiếu kết quả với gate đã chốt, không điều chỉnh gate theo kết quả quan sát.

## 10. Kiến trúc code, luồng dữ liệu và data contract

### 10.1. Luồng E0

```text
JSONL dataset
    -> load/filter/tokenize
    -> target AR prefill + target continuation
    -> DFlash block proposal (K=4,8,16)
    -> target verification / acceptance accounting
    -> exact-match guardrail + timing/memory capture
    -> per-round JSONL
    -> E0 analyzer: survival, MAT, bootstrap, bucket tables
```

Runner E0 nằm trong `target_kv_experiments.py`. Phần runner không tự suy diễn kết luận; nó chỉ ghi quan sát gốc. `e0_report.py` là lớp tổng hợp, nhờ đó cùng một file raw có thể được phân tích lại với gate khác mà không phải chạy model lần nữa.

Mỗi round E0 lưu tối thiểu:

| Trường | Ý nghĩa |
|---|---|
| `dataset`, `doc_id`, `prompt_tokens` | Định danh và độ dài input |
| `k`, `max_new_tokens` | Cấu hình block và budget sinh |
| `raw_acceptance_length` | Giá trị API trả về, bao gồm fallback |
| `accepted_draft_tokens` | Giá trị đã chuẩn hóa để phân tích |
| `accepted_token_ids`, `target_token_ids` | Audit nội dung và exact-match |
| `target_ms`, `draft_ms`, `verify_ms`, `total_ms` | Cost accounting |
| `peak_memory_mb` | Đỉnh bộ nhớ quan sát được |
| `status`, `error` | Phân biệt row hợp lệ và row lỗi |

### 10.2. Luồng E1

```text
E0/JSONL anchors
    -> target forward tại anchor
    -> hidden state và/hoặc KV state
    -> CPU cache dạng shard
    -> split theo document (train/dev/test)
    -> frozen-size probe decoder
    -> CE/Top-1/Top-5/prefix@8
    -> report theo dataset và representation
```

Extraction và probe được tách thành `target_kv_e1.py` và `e1_representation_probe.py`. `e1_representation_probe.py` dùng cùng projection/head giữa các variant, cùng số tham số trainable (158,021,760), cùng interface 128×64 và horizon 16. Như vậy chênh lệch không đến từ việc KV variant có decoder lớn hơn.

Các variant được định nghĩa như sau:

| Variant | Thông tin được phép dùng | Vai trò |
|---|---|---|
| `hidden` | Hidden state tại anchor | Baseline token-wise |
| `hidden_sequence` | Chuỗi hidden gần anchor | Baseline mạnh hơn, có local history |
| `multi_layer_hidden` | Hidden từ nhiều layer chọn trước | Kiểm tra lợi ích depth |
| `kv` | KV của target tại anchor/layers đã chọn | Hypothesis chính |
| `kv_shuffled` | KV bị xáo trộn giữa sample/anchor | Control phá thông tin đúng |
| `kv_recent` | Chỉ phần KV gần đây | Control về locality |
| `kv_wrong_document` | KV lấy từ document khác | Negative control |

`kv_shuffled`, `kv_recent` và `kv_wrong_document` là controls chẩn đoán, không phải các baseline serving hoàn chỉnh. Chúng giúp phát hiện một probe có thể đang học artifact về kích thước, vị trí hoặc document identity thay vì nội dung.

## 11. Chuẩn bị dữ liệu, coverage và tính hợp lệ

### 11.1. Nguồn dữ liệu và lọc

Các thí nghiệm dùng JSONL local trong `data/`, giữ nguyên trường document id và nội dung nguồn. Trước mỗi run, pipeline:

- bỏ record thiếu input hoặc có tokenization lỗi;
- tính độ dài bằng tokenizer của target;
- đưa record vào bucket theo độ dài token;
- không pad/truncate để đạt bucket;
- giữ document id cho bootstrap và split E1;
- loại các row không đạt exact-match khỏi diễn giải speed/quality, nhưng vẫn ghi nhận trong audit.

Với E1, anchor chỉ được giữ khi còn đủ target suffix cho horizon 16 và đủ context để tạo representation. Split theo document, không split ngẫu nhiên theo anchor, để tránh leakage giữa các anchor của cùng một tài liệu.

### 11.2. Quy mô đã thực sự chạy

| Run | Documents | K rows | Round rows | Bucket chính |
|---|---:|---:|---:|---|
| GovReport pilot | 13 | 39 | 312 | 2–4K: 3; 4–8K: 10 |
| Multi-News pilot | 49 | 147 | 1,176 | 0–2K: 27; 2–4K: 15; 4–8K: 7 |
| CNN/DM pilot | 30 | 90 | 720 | 0–2K: 30 |
| GovReport confirmation | 5 | 15 | 480 | max-new 32 |
| Multi-News confirmation | 5 | 15 | 480 | max-new 32 |

Tổng cộng E0 có 92 documents pilot, 276 cấu hình K ở pilot và 3,168 round rows nếu tính cả confirmation (306 cấu hình K nếu tính cả confirmation). Con số này đủ để kiểm định feasibility trên các bucket quan sát được, nhưng **chưa đủ để tuyên bố hiệu năng production hoặc kết luận về context 8–16K+** vì GovReport dài chưa phủ được bucket dài tương ứng và CNN/DM chủ yếu là short-context control.

### 11.3. Quy mô E1

- GovReport: 17 documents, 34 anchors sau lọc; 20/6/8 rows cho train/dev/test.
- Multi-News: 49 documents, 98 anchors sau lọc; 58/18/22 rows cho train/dev/test.
- 7 representation variants trên mỗi dataset.
- Mọi variant có 158,021,760 trainable parameters, interface 128×64 và horizon 16.

E1 được dùng như **representation probe**, không phải benchmark latency. Do số anchor còn nhỏ, các metric CE/accuracy được dùng để quyết định gate định hướng; chưa xem chúng là estimate tổng quát cho mọi dữ liệu.

## 12. Nhật ký tiến hành và các vấn đề kỹ thuật đã xử lý

### 12.1. Preflight môi trường

Máy thực nghiệm là `tuantb@teslaT4`, dùng Python hệ thống Miniconda theo yêu cầu, không dùng `.venv` cho run GPU. Các snapshot Qwen3-4B target và Qwen3-4B-DFlash-b16 đã có trong cache local. Preflight xác nhận GPU, CUDA, torch, model config, tokenizer và khả năng load model trước khi chạy dataset.

Thông tin phần cứng/runtime chi tiết nằm ở mục 3; điểm cần nhấn mạnh là T4 có 15,360 MiB VRAM và target chạy FP16. Đây là giới hạn thực tế chi phối cả quy mô context và số anchor có thể giữ trên GPU.

### 12.2. E0 pilot và kiểm tra accounting

Run đầu tiên được dùng để kiểm tra toàn bộ đường đi từ JSONL đến raw output. Một sample chính thức của DFlash cho `raw_acceptance_length=[1,...]` được đối chiếu trực tiếp; vì vậy metric phân tích dùng `accepted_draft_tokens = raw - 1`, không đếm fallback target token như token được draft chấp nhận.

Sau khi accounting ổn định, E0 chạy từng (K\in\{4,8,16\}), thu per-round records và tổng hợp theo bucket độ dài. Các run không dừng ở một mean duy nhất: survival tại từng vị trí (j), MAT, số positive rounds, exact-match và peak memory đều được ghi.

### 12.3. Context-drop feasibility và giới hạn T4

Một run context-drop được thử trên input 11,052, 16,384 và 28,156 tokens với `max_new_tokens=1`, dùng 8-bit loading để đo khả năng chứa context dài. Peak VRAM lần lượt khoảng 8.08, 9.50 và 12.65 GiB; cả ba run đều có raw acceptance bằng 1, tức accepted draft bằng 0. Input 41,651 token bị loại vì vượt `max_position_embeddings=40,960` của target.

Kết quả này có hai diễn giải riêng:

- T4 có thể chứa một số context dài khi dùng 8-bit, nhưng đó không phải cấu hình chính FP16 và không chứng minh được acceptance dài.
- Việc không có bucket tự nhiên 8–16K+ trong data không được biến thành bằng chứng phủ định long-context hypothesis. Vì thế context-drop gate được ghi là **INCONCLUSIVE**, không phải PASS hay FAIL.

### 12.4. E0 confirmation với max-new 32

Để kiểm tra có phải `max_new_tokens=1` làm mất cơ hội hình thành block hay không, GovReport và Multi-News được chạy lại với `max_new_tokens=32`, vẫn giữ cùng cặp target/drafter và (K\). Kết quả vẫn không có round nào có accepted draft token dương. Đây là confirmation cho short/mid-context feasibility failure trong scope đã quan sát.

### 12.5. E1 extraction/probe và lỗi OOM

Extraction ban đầu giữ quá nhiều hidden/KV tensor trên GPU trong một lượt xử lý. Trên T4 điều này gây tăng VRAM và không phù hợp với số layer của Qwen3-4B. Pipeline được sửa theo hướng:

- chạy extraction trong `inference_mode`;
- chỉ giữ tensor cần cho representation hiện tại;
- chuyển hidden/KV chunks về CPU ngay sau mỗi sample;
- lưu shard nhỏ, đọc lại theo split;
- probe đọc CPU cache và chỉ đưa mini-batch cần thiết lên device.

Sau sửa, E1 hoàn thành cho GovReport và Multi-News với cùng budget giữa các variant. Đây là sửa về quản lý bộ nhớ, không thay đổi công thức label hoặc kiến trúc probe.

### 12.6. Exact-match guardrail

E0 kiểm tra token ids của target continuation và token ids được dùng trong verification. Kết quả exact-match: GovReport pilot 36/39, Multi-News 147/147, CNN/DM 90/90, GovReport confirmation 12/15 và Multi-News confirmation 15/15. Ba row GovReport pilot không khớp đều thuộc cùng một document; bảy token đầu trùng, token cuối khác (8397 so với 29340).

Các row mismatch không được dùng để kết luận speed/quality. Chúng vẫn được giữ trong report để minh bạch audit. Nguyên nhân chính xác của mismatch cuối token chưa được khẳng định; do đó báo cáo không gán nó cho một lỗi cụ thể của model hay CUDA.

## 13. Kết quả chi tiết, diễn giải và decision gate

### 13.1. E0 — acceptance survival

Trong GovReport pilot (312 rounds), Multi-News pilot (1,176 rounds), CNN/DM pilot (720 rounds), GovReport confirmation (480 rounds) và Multi-News confirmation (480 rounds), mọi (K\in\{4,8,16\}) đều có:

\[
\mathrm{MAT}=0,\qquad S_L(j)=0\quad\text{cho mọi }j\ge1.
\]

Nói cách khác, trong các proposal hợp lệ đã chạy, DFlash không chấp nhận token draft nào trước token fallback. Vì vậy không thể ước lượng một survival curve giảm dần, không thể chứng minh acceptance burst, và không có cơ sở để đánh giá speedup của Target-KV block drafting trên các row này.

Kết luận E0 được phân tầng:

| Claim | Trạng thái | Lý do |
|---|---|---|
| Có acceptance dương ở short/mid context trên cặp model hiện tại | **FAIL** | 0/2,208 pilot rounds dương và confirmation cũng 0 |
| Tăng max-new từ 1 lên 32 khôi phục acceptance | **FAIL** | Hai dataset confirmation vẫn toàn 0 |
| Acceptance sống sót ở natural long context 8–16K+ | **INCONCLUSIVE** | Không có đủ bucket tự nhiên; context-drop không thay thế được sample tự nhiên |
| Dùng E0 hiện tại để claim serving speedup | **FAIL / không được phép** | Không có accepted draft token để tạo speedup |

### 13.2. E1 — chất lượng representation probe

Các số liệu chính đã ghi ở mục 6. Mẫu hình nhất quán trên cả hai dataset:

- `hidden_sequence` là baseline tốt nhất hoặc gần tốt nhất theo CE/Top-5;
- `kv` không vượt `hidden_sequence`: GovReport CE 11.9097 so với 11.7173, Multi-News CE 11.8899 so với 11.6988;
- `kv_shuffled`, `kv_recent` và `kv_wrong_document` không tạo ra mẫu hình cho thấy một tín hiệu KV ổn định, vượt trội;
- `prefix@8` bằng 0 cho tất cả variant trong các run đã thực hiện, tức probe không đạt yêu cầu dự đoán chính xác cả prefix 8 token.

Gate E1 được đánh giá như sau:

| Gate | Quan sát | Quyết định |
|---|---|---|
| KV thắng token-wise hidden sequence trên ít nhất 2 regime | Không; KV thua trên GovReport và Multi-News | **FAIL** |
| KV có lợi ích nhất quán qua controls | Không có bằng chứng nhất quán | **FAIL** |
| Probe chứng minh được exact block drafting | Prefix@8 đều 0 | **FAIL** |
| Có thể chuyển thẳng E1 thành latency claim | Không; E1 không đo serving | **Không được suy luận** |

Điểm cần thận trọng: `kv_wrong_document` có accuracy một số nơi không thấp hơn tất cả variant không phải bằng chứng KV có ích. Đó là dấu hiệu probe nhỏ và tập dữ liệu nhỏ chưa đủ mạnh để phân giải mọi hiệu ứng; gate chính vẫn là so sánh công bằng `kv` với baseline mạnh nhất `hidden_sequence`.

### 13.3. Bảng quyết định tổng hợp

| Nhánh | Điều kiện cần | Kết quả | Hành động |
|---|---|---|---|
| Tiếp tục Target-KV | E0 có acceptance dương và E1 KV có signal | Không đạt cả hai gate trong scope | Dừng hướng hiện tại ở phase này |
| Mở rộng block drafting | E0 MAT dương, có survival theo (j) | MAT=0 | Không triển khai controller/serving |
| Tối ưu KV adapter | E1 KV thắng hidden sequence | KV thua trên hai dataset | Không đầu tư adapter KV-specific |
| Chạy E2 benchmark | E0/E1 pass | E0/E1 fail | E2 chưa chạy |
| Chạy E3 strong drafter | Có tín hiệu hiện tượng cần replication | Chưa có tín hiệu để replicate | Chưa chạy |

Đây là quyết định “dừng có điều kiện trong scope” chứ không phải khẳng định toán học rằng mọi target-KV drafting đều bất khả thi. Một replication dài-context FP16 với cặp model/weight tương thích có thể mở lại câu hỏi, nhưng phải được xem là một protocol mới.

## 14. Những gì kết quả cho phép và không cho phép kết luận

### 14.1. Kết luận được phép

1. Trên môi trường T4, cặp Qwen3-4B target + Qwen3-4B-DFlash-b16 và các dữ liệu/độ dài đã chạy, không quan sát được accepted draft token ở (K=4,8,16).
2. Việc tăng `max_new_tokens` từ 1 lên 32 không làm thay đổi kết quả zero-acceptance trên GovReport và Multi-News confirmation.
3. Trong representation probe hiện tại, target KV không chứng minh được lợi ích dự đoán suffix so với hidden sequence trong cả GovReport và Multi-News.
4. Với bằng chứng hiện có, không nên xây tiếp controller Target-KV, KV adapter-specific hoặc E2 serving benchmark.

### 14.2. Kết luận chưa được phép

- Không được nói “long-context Target-KV hypothesis đã bị bác bỏ ở mọi context”, vì long natural-context bucket chưa đủ.
- Không được quy đổi CE/accuracy của E1 thành tokens/ms hoặc speedup.
- Không được gọi E0 là benchmark production: quy mô và profile context chưa đạt yêu cầu.
- Không được kết luận DFlash nói chung không hoạt động; kết luận chỉ áp dụng cho cặp snapshot, protocol và môi trường đã ghi.
- Không được dùng zero acceptance để khẳng định nguyên nhân là target–drafter misalignment, tokenizer, weight hoặc kernel nếu chưa có thí nghiệm nguyên nhân riêng.

### 14.3. Điều kiện để mở lại hướng nghiên cứu

Chỉ nên mở lại sau khi đồng thời có:

1. snapshot target/drafter đã xác nhận aligned bằng một sanity test acceptance dương ở context ngắn;
2. ít nhất một bucket natural context dài được phủ đủ documents, không tạo bằng padding/truncation;
3. E0 có MAT dương và survival curve có ý nghĩa;
4. E1 được chạy lại với split lớn hơn, bootstrap document-level và probe KV thắng baseline hidden sequence;
5. sau đó mới thực hiện E2 latency trên cùng target/drafter và cost accounting thật.

## 15. Phụ lục tái lập: artifact, lệnh chạy và kiểm chứng

### 15.1. Artifact chính

Toàn bộ code và kết quả liên quan được đặt cùng thư mục `src/analyze/groundsync/` theo yêu cầu:

| Artifact | Nội dung |
|---|---|
| `target_kv_experiments.py` | E0 runner, model loading, block sizes, metrics raw, timing/memory |
| `e0_dflash_failure_map.py` | Chạy/tổng hợp failure map và context-drop feasibility |
| `e0_report.py` | Phân tích E0 từ JSONL: bucket, survival, MAT, bootstrap |
| `target_kv_e1.py` | Extraction representation/target suffix và lưu shard |
| `e1_representation_probe.py` | Probe công bằng cho hidden/KV variants |
| `e1_report.py` | Tổng hợp CE, Top-1, Top-5, prefix@8 |
| `tests/test_target_kv_experiments.py` | Test contract E0, acceptance accounting, bucket và exact guardrail |
| `tests/test_target_kv_e1.py` | Test extraction, split, variant shape và probe |
| `tests/test_target_kv_reports.py` | Test report/aggregation và bootstrap |
| `results/tkv-*` | Raw JSONL, summary và report cho từng run |
| `target_kv_decision_report_2026-09-03.md` | Báo cáo tổng hợp hiện tại |

Các model snapshot được dùng:

```text
/home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
/home/tuantb/.cache/huggingface/hub/models--z-lab--Qwen3-4B-DFlash-b16/snapshots/61ab4992e5b5ec5913c7f8a9618367b4309533a3
```

### 15.2. Các lệnh E0 đã dùng

Các lệnh dưới đây mô tả protocol đã chạy; đường dẫn output có thể thay đổi theo run id nhưng không thay đổi tham số thực nghiệm:

```bash
cd /home/tuantb/fast_infer_text_sum
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TARGET_MODEL=/home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
export DFLASH_MODEL=/home/tuantb/.cache/huggingface/hub/models--z-lab--Qwen3-4B-DFlash-b16/snapshots/61ab4992e5b5ec5913c7f8a9618367b4309533a3

python3 src/analyze/groundsync/target_kv_experiments.py \
  --target-model "$TARGET_MODEL" \
  --drafter-model "$DFLASH_MODEL" \
  --data-file data/govreport/test.jsonl \
  --output-dir src/analyze/groundsync/results/tkv-e0-pilot-gov-20260903 \
  --k-values 4 8 16 --max-new-tokens 32

python3 src/analyze/groundsync/target_kv_experiments.py \
  --target-model "$TARGET_MODEL" \
  --drafter-model "$DFLASH_MODEL" \
  --data-file data/multi_news/test.jsonl \
  --output-dir src/analyze/groundsync/results/tkv-e0-pilot-multinews-20260903 \
  --k-values 4 8 16 --max-new-tokens 32

python3 src/analyze/groundsync/e0_report.py \
  --input src/analyze/groundsync/results/tkv-e0-pilot-gov-20260903 \
  --output src/analyze/groundsync/results/tkv-e0-pilot-gov-20260903/report.json
```

CNN/DM được chạy cùng runner như một short-context control. Confirmation dùng đúng runner nhưng giới hạn 5 documents/dataset và `max_new_tokens=32`; đây là run kiểm tra protocol, không phải claim tăng quy mô thống kê.

### 15.3. Các lệnh E1 đã dùng

```bash
python3 src/analyze/groundsync/target_kv_e1.py \
  --target-model "$TARGET_MODEL" \
  --data-file data/govreport/test.jsonl \
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

Multi-News dùng cùng extraction/probe, với split theo document và 98 anchors. Các lệnh là cách tái lập logic; nếu cache/model hoặc tên file dữ liệu thay đổi, cần thay đường dẫn nhưng không được thay đổi định nghĩa metric khi so sánh.

### 15.4. Kiểm chứng code và artifact

Tại thời điểm hoàn thiện báo cáo:

```text
python3 -m pytest -q src/analyze/groundsync/tests   -> 94 passed
python3 -m compileall -q src/analyze/groundsync   -> exit 0
git diff --check -- src/analyze/groundsync         -> exit 0
artifact audit                                    -> PASS
```

Toàn bộ test riêng cho Target-KV đã pass 28 test trong targeted audit trước khi chạy full test suite của thư mục. Full test suite của `src/analyze/groundsync/tests` pass 94 test. Lệnh pytest toàn repository có một failure cũ ngoài phạm vi Target-KV tại `tests/test_runtime_compat_fixes.py::test_llama31_rope_theta_is_recovered_from_modern_config` (giá trị `rope_theta` hiện trả 10000 thay vì test kỳ vọng 500000); failure này không được sửa trong task vì không liên quan đến code thực nghiệm.

## 16. Kết luận cuối của phase này

Phase Target-KV đã hoàn tất phần **thiết kế, triển khai, chạy thực nghiệm trong khả năng của T4, lưu artifact, phân tích và kiểm chứng code**. Kết quả quyết định là:

\[
\boxed{\text{E0 short/mid-context: FAIL}}\qquad
\boxed{\text{E0 natural long-context: INCONCLUSIVE}}\qquad
\boxed{\text{E1 KV-specific representation: FAIL}}
\]

Do đó quyết định hiện tại là **không tiếp tục đầu tư Target-KV/KV-specific adapter, block controller hoặc E2 serving benchmark trên cùng setup**. E2 và E3 được ghi nhận là chưa chạy một cách có chủ đích, vì các gate tiền đề chưa đạt; đây không phải thiếu artifact hay bỏ sót một bước bắt buộc.

Kết luận thực tế nhất cho nhóm nghiên cứu là: trước mắt quay lại kiểm tra alignment/compatibility của target–drafter bằng một sanity protocol độc lập. Chỉ khi sanity acceptance dương và có natural long-context coverage mới đáng mở lại E0; nếu E0 vẫn zero, hướng Target-KV nên được đóng. Nếu E0 sống lại nhưng E1 KV vẫn thua hidden sequence, nên bỏ claim KV-specific và chỉ xem xét các representation/policy khác có bằng chứng trực tiếp.

Báo cáo này vì vậy phân biệt rõ ba trạng thái: **đã kiểm định và thất bại**, **đã kiểm định nhưng chưa đủ dữ liệu để kết luận**, và **chưa chạy vì gate trước đó thất bại**. Đây là cơ sở để quyết định bước tiếp theo mà không phóng đại bằng chứng.
```
