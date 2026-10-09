# Chuẩn bị dữ liệu huấn luyện AMR-DFlash từ corpus regenerate

Ngày cập nhật: **09/10/2026**. Mục tiêu trước mắt là tạo một pilot có tín hiệu
supervision thật, từ corpus ShareGPT + ArXiv đã regenerate trên B200.

## Dữ liệu cần tạo

AMR giữ nguyên target và pretrained DFlash-5L. Selector học từ chênh lệch
acceptance giữa các support cùng state/budget; compressor học từ logits của
target verifier. Vì vậy dữ liệu train gồm:

| Artifact | Nguồn tạo | Mục đích |
|---|---|---|
| Manifest prompt, source ID, split | Corpus regenerate hiện có | Đầu vào capture |
| Projected features và state snapshots | `capture` với snapshot target/drafter đã khóa | Context trước anchor |
| Candidate supports | `candidates` | So các tập raw token cùng budget |
| Acceptance preferences | `label` với full target verifier | Huấn luyện selector |
| Teacher logits và proposal history | Cùng lượt `label` | Huấn luyện compressor |

Corpus regenerate đủ làm nguồn prompt. Câu trả lời regenerate cuối cùng được
loại khỏi prompt; những lượt assistant trước đó vẫn thuộc lịch sử hội thoại.
AMR tạo trajectory greedy mới và supervision riêng. Tóm tắt gốc ArXiv ở
`metadata.reference_summary` được giữ làm reference; câu trả lời regenerate
không được dùng làm reference ROUGE.

Cache hidden của MR-DFlash chưa có importer trực tiếp vào AMR. Nếu đây là
cache của run 50k ngày 28/09 được ghi trong
[audit MR-DFlash](mr_dflash_gpu_experiments.md), hidden recompute ngày 05/10
đã fail 32/32 mẫu train và 32/32 mẫu val. Quy trình dưới đây đọc JSONL
regenerate và capture feature mới với hợp đồng AMR, không cần dùng cache hidden đó.

## Pilot đề nghị

Chọn tối đa **200 document train + 50 document validation**. Giữ nguyên split
nguồn; phân tầng ShareGPT/ArXiv bằng seed cố định. Chọn prompt sau chat template
dài ít nhất **4.097 token**, rồi giới hạn input ở **16.384 token** theo cùng
head/tail truncation với capture. Với raw budget 4.096, prompt quá ngắn thường
làm các candidate trùng nhau và không có preference để học.

Nếu ShareGPT không có đủ mẫu dài, report sẽ cho thấy mất cân bằng còn lại;
không tăng chiều dài bằng padding hay đưa response regenerate vào prompt.
Hai limit là trần số mẫu, không đảm bảo corpus có đủ mẫu đạt bộ lọc. Nếu budget
khác 4.096, đổi ngưỡng tối thiểu tương ứng và khóa config trước capture.

## 1. Tạo manifest bằng tokenizer local

Chạy trong repository trên server; production dùng `python3` hệ thống thông
qua shared runtime. Master phải trỏ tới đúng snapshot target và pretrained
DFlash **5 layer, block size 16**. Tokenizer dùng khi chuẩn bị phải thuộc
snapshot target sẽ capture. Lệnh `prepare-data` chỉ load tokenizer local,
không load weights, không cần GPU và không tải tài nguyên.

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum
export FAST_INFER_MASTER_CONFIG=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env
source scripts/common/config.sh
fast_infer_load_config amr_dflash

# Thay bằng thư mục regenerate thực tế nếu corpus đang nằm ở run khác.
MR_REGENERATED_DIR=/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_phase1_50k_50_50_10k_b200_gpu0_ctx16k_out1k_20260928T132928Z/regenerated_full
AMR_PILOT_ID=amr_regenerated_pilot_20261009
AMR_PREPARED_DIR="$PWD/outputs/amr_dflash/prepared/$AMR_PILOT_ID"
AMR_PILOT_ROOT="$PWD/outputs/amr_dflash/$AMR_PILOT_ID"

bash scripts/run.sh amr_dflash prepare-data \
  --train-input "$MR_REGENERATED_DIR/train.jsonl" \
  --validation-input "$MR_REGENERATED_DIR/val.jsonl" \
  --tokenizer-path "$TARGET_MODEL" \
  --output-dir "$AMR_PREPARED_DIR" \
  --train-samples 200 --validation-samples 50 \
  --min-input-tokens 4097 --max-input-tokens 16384 --seed 17

export AMR_DATA_MANIFEST="$AMR_PREPARED_DIR/manifest.jsonl"
python3 -m json.tool "$AMR_PREPARED_DIR/report.json"
```

`--train-input`/`--validation-input` nhận nhiều lần nếu hai nguồn nằm trong
file riêng. Input phải là JSONL regenerate có `id`, `source` bằng `sharegpt`
hoặc `arxiv`, và `conversations` kết thúc bằng câu trả lời assistant không rỗng.
Role `human`/`gpt` cũng được chuẩn hóa. Record lỗi báo rõ file, dòng, sample ID;
không âm thầm đưa record lỗi vào manifest.

Đầu ra gồm `manifest.jsonl`, `report.json`, `preparation_contract.json` và
`token_lengths.sqlite3`. Report ghi số đọc/đủ điều kiện/được chọn theo
split và source, số bỏ vì ngắn/trùng, token-length statistics, số reference,
hash input/tokenizer và seed. `training_labels_ready=false` là đúng ở giai đoạn
này: manifest chưa có supervision AMR.

Tool quét toàn bộ ID và nội dung prompt trước khi chốt subset, để phát hiện
overlap nằm ngoài limit. Nội dung được so trên toàn lịch sử, gồm role và các
lượt assistant cũ. Nếu nguồn có `source_document_id`, `document_id` hoặc
`metadata.original_id`, các cửa sổ của cùng document cũng không được nằm ở
hai split. Tool bỏ duplicate trong cùng split, giữ nguyên JSONL đầu vào và
giới hạn số prompt giữ trong RAM theo limit/source.

Không ghi đè thư mục có sẵn. Nếu bị gián đoạn, chạy lại **cùng lệnh** thêm
`--resume`: tiếp tục quét nhưng dùng lại token lengths đã checkpoint. Resume
đòi cùng input bytes, tokenizer, seed và các limit. Muốn đổi subset/config,
dùng ID/thư mục mới. Progress audit hiện ở terminal.

## 2. Capture state và features mới trên B200

Sau khi review report và spot-check vài hội thoại trong manifest, dùng GPU
đã được cấp trong master/allocation hiện tại. Preflight kiểm tra đúng model,
tokenizer, input và B200; không chạy trên host T4 local.

```bash
bash scripts/run.sh amr_dflash preflight --require-b200

bash scripts/run.sh amr_dflash capture \
  --input "$AMR_DATA_MANIFEST" --run-root "$AMR_PILOT_ROOT" \
  --max-input-tokens 16384 --max-new-tokens 256 \
  --max-states-per-document 3
```

Không thêm `--max-samples 200` ở bước này: manifest đã giới hạn số mẫu theo
split và đặt train trước validation; cắt toàn manifest có thể làm mất validation.
Giới hạn ba state/document là mức bắt đầu để kiểm tra tín hiệu/chi phí. Có thể
đổi lên sáu trong một run mới sau khi pilot cho thấy cần thêm state.

Capture lưu `documents.jsonl`, `states.jsonl`, feature bundles và
`manifest.json` khóa fingerprint target/drafter, dtype/backend, memory config,
input/output caps, prompt policy, seed và split. Snapshot model phải giữ
nguyên qua label/train. `capture --resume` yêu cầu cùng các giá trị này.
Prompt policy hiện hỗ trợ structured messages version 2; capture cũ có policy
version 1 không được resume bằng hợp đồng mới, cần run root mới khi recapture.

## 3. Tạo labels huấn luyện

```bash
bash scripts/run.sh amr_dflash candidates --run-root "$AMR_PILOT_ROOT"
bash scripts/run.sh amr_dflash label --run-root "$AMR_PILOT_ROOT"
python3 -m json.tool "$AMR_PILOT_ROOT/training_signal.json"
```

Kiểm tra summary của capture/label để xác nhận job hoàn tất, số documents/states
theo split đúng và không hết adaptation budget giữa chừng. Supervision nằm ở
`preferences.jsonl`, `candidate_labels.jsonl`, `teacher/*.pt`; states và
features được tham chiếu từ cùng capture run.

Trong `training_signal.json`:

- `train_selector_signal_present=true`: train có preference không hòa do
  verifier tạo; fixture tổng hợp không được đếm trong báo cáo tín hiệu.
- `train_compressor_signal_present=true`: train có teacher logits từ candidate
  thành công, chưa censor.
- `by_split` ghi số states có preference, số pairs, teacher rows và censor.
  Validation cần đủ states có tín hiệu để đánh giá; không lấy validation
  pairs để bù thiếu train pairs.

Hai cờ trên chỉ xác nhận có supervision, chưa chứng minh số mẫu đủ hoặc model
học tốt. Nếu train preferences bằng 0, xem candidate có bị degenerate do
context ≤ raw budget, reward có hòa hết, hay quá nhiều state cuối bị censor.
Chọn thêm document dài/đa dạng hoặc điều chỉnh state/candidate/config rồi
tạo run mới. Không gán reward giả và không phá tie thành preference.

## 4. Kiểm chứng train ngắn trước khi mở rộng

Khi có tín hiệu thật và artifact hoàn tất, chạy 20 bước mỗi pha:

```bash
AMR_PILOT_CHECKPOINT_DIR="$PWD/checkpoints/amr_dflash/$AMR_PILOT_ID"
bash scripts/run.sh amr_dflash train-selector \
  --run-root "$AMR_PILOT_ROOT" --steps 20 \
  --checkpoint-out "$AMR_PILOT_CHECKPOINT_DIR/selector_20.pt"

bash scripts/run.sh amr_dflash train-compressor \
  --run-root "$AMR_PILOT_ROOT" --steps 20 \
  --checkpoint-in "$AMR_PILOT_CHECKPOINT_DIR/selector_20.pt" \
  --checkpoint-out "$AMR_PILOT_CHECKPOINT_DIR/compressor_20.pt"

bash scripts/run.sh amr_dflash evaluate-fixed \
  --run-root "$AMR_PILOT_ROOT" --split validation --mode amr \
  --checkpoint "$AMR_PILOT_CHECKPOINT_DIR/compressor_20.pt" \
  --output "$AMR_PILOT_ROOT/evaluation/amr_validation_20.jsonl"
```

Kiểm tra loss/gradient hữu hạn, cập nhật weights, save/load checkpoint và
same-state acceptance trên validation, so với dense/selection. Dùng cùng
max-input cap khi chạy rollout trên manifest này. Với pilot đạt yêu cầu, mới
chọn một run lớn hơn, chẳng hạn 500–1.000 document train, và tính tiếp theo
RAM, disk, GPU-hour ledger và tỷ lệ state có preference. Train V0 giữ feature
bundles/teachers trong cache RAM; số document lớn cần xem lại loader/cache trước
khi dùng hết 50k.

Giữ holdout riêng theo source ID/nội dung cho đánh giá cuối. Có thể cung cấp
`--holdout-input` và `--holdout-samples` nếu đã có file test riêng; không đổi
validation thành holdout sau khi đã dùng để chọn config/checkpoint.

Kiểm chứng local ngày 09/10/2026: **68 tests passed** cho converter, AMR contracts,
regressions và launcher. Sample tổng hợp 120 dòng chọn 20 train + 10 validation,
mỗi split cân bằng hai source, giữ 15 reference gốc ArXiv và không đưa response
regenerate cuối vào manifest. Test với Qwen/DFlash nhỏ thật trên CPU xác nhận
manifest nhiều lượt đi qua capture/candidates/verifier labeling và tạo teacher.
Các regression train dùng preferences có kiểm soát để kiểm tra gradient; không
xem đó là bằng chứng preference tự nhiên của corpus thật.

Corpus 50k và runtime B200 thực chưa được đọc/chạy từ workspace này; report
pilot trên server mới là bằng chứng về phân phối, số preference tự nhiên và
chi phí thực tế của dữ liệu hiện có.
