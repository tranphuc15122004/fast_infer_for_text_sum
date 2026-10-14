# Chạy campaign A1–A3 trên B200

## Chỉ cần hai lệnh

Chạy từ thư mục repo trên server B200:

```bash
bash scripts/run_dflash_context_A_b200.sh check
bash scripts/run_dflash_context_A_b200.sh start
# Nếu tiến trình vẫn thoát sau lần tự retry:
bash scripts/run_dflash_context_A_b200.sh resume RUN_ID
```

`check` kiểm tra Python/packages, GPU/FA4, models, dữ liệu, hash corpus,
exposure audit và số source groups. Lệnh này không chạy generation.

`start` treo campaign dưới nền. Nó tự chạy preflight → prepare → parity/validate
→ pilot → dev → lock → confirm → report. Campaign dùng GPU 0, FA4, seed 42,
và cấu hình B200 đã đăng ký trong kế hoạch. Master config được đọc qua
`config/master.path`; không cần truyền run ID, đường dẫn manifest hay config.
Các phase GPU tự retry một lần từ checkpoint khi gặp lỗi tiến trình. Nếu lần retry
vẫn thất bại, `resume RUN_ID` chạy lại preflight/prepare rồi nối tiếp từ
cell/prefix đã lưu và ghi tiếp vào
`campaign.log`; lệnh từ chối chạy song song cùng RUN_ID. Resume yêu cầu giữ nguyên
code, runtime, model và dữ liệu để chữ ký checkpoint còn khớp. `flock` giữ khóa
trong suốt campaign để hai lệnh `start/resume` đồng thời không ghi chung checkpoint.

Gate correctness yêu cầu full adapter khớp native full DFlash. Khác biệt token
giữa DFlash và AR được ghi làm diagnostic, không chặn A; A3 báo ROUGE theo cặp
với native full DFlash và acceptance. Vì chưa đăng ký biên ROUGE, không dùng
ROUGE để tuyên bố chất lượng được bảo toàn. Protocol amendment này được ghi
nhận ngày 13/10/2026, trước khi chạy ma trận A1/A2/A3 và mở heldout.

Round index và accepted-candidate count được ghi trên trajectory của full
adapter, nên prefix replay cũng phải dựng từ `adapter_ids`; trace chỉ được replay
khi `native_ids == adapter_ids` và parity status là `pass`. `ar_ids` chỉ dùng
để chẩn đoán first divergence, không được dùng để dựng prefix hay pending anchor.

Mỗi prefix được kiểm tra số accepted candidates so với trace trước khi chạy
A1/A2/A3. Nếu full DFlash replay của một prefix lỗi hoặc lệch acceptance, runner
ghi checkpoint `prefix_diagnostics.jsonl`, đánh dấu prefix đó bị loại, rồi tiếp
tục các prefix còn lại. Báo cáo giữ lại chẩn đoán, hạ kết luận A1/A2/A3 replay về
`inconclusive`, và không tính prefix lỗi vào các ước lượng. Gate native/full
adapter parity và gate chọn cấu hình trên dev vẫn dừng campaign khi không đạt.

Replay của round trước chỉ tạo attention ranking cho `previous_dflash`; nếu số
accepted ở round phụ lệch nhưng attention map đủ cho mọi layer, runner tiếp tục
dùng map và ghi cả hai số cùng trạng thái `acceptance_mismatch` vào
`previous_replay_check`. Nếu replay phụ lỗi hoặc thiếu attention layer, chỉ các
cell `previous_dflash` được đánh dấu `not_eligible`; selector khác và các prefix
còn lại vẫn chạy.

Checkpoint cell có `status=error` được thử lại khi resume; cell thành công không
bị ghi đè. Prefix marker chỉ bỏ qua prefix khi các cell đã lưu còn đủ và không
có cell lỗi. Nếu máy hoặc tiến trình dừng đột ngột, dùng `resume RUN_ID` với
cùng code, runtime, model và dữ liệu.

Exception cục bộ trong reference, dựng/replay prefix hoặc đo A3 được thử lại một
lần sau khi giải phóng bộ nhớ. Nếu vẫn lỗi, runner ghi record lỗi rồi chạy đơn
vị tiếp theo. Reference thiếu, prefix bị loại, timing/acceptance không hợp lệ
đều được giữ trong coverage; các số NaN/Inf trong record lỗi được lưu thành
`null` kèm tên trường lỗi để checkpoint vẫn là JSON hợp lệ. Cache prefix được
giải phóng trước khi dựng prefix kế tiếp, tránh giữ hai cache lớn cùng lúc.

Chỉ các trường hợp nghiêm trọng dừng ngay: native/full-adapter mất parity,
invariant KV cache sai, CUDA device-side assert/illegal memory access, thiếu
dependency hoặc lỗi filesystem. Ba đơn vị tính toán liên tiếp thất bại sau
retry được coi là lỗi hệ thống và cũng dừng; với A3, ngưỡng này tính trên ba
mẫu liên tiếp không có bộ arm hợp lệ ở bất kỳ cấu hình nào. Lỗi liên tiếp chỉ
ở các refresh period phụ không kích hoạt ngưỡng dừng toàn campaign. Ctrl-C không bị nuốt. Dev lock
vẫn yêu cầu đầy đủ các arm/repetition của cấu hình được chọn và đạt các gate
acceptance/speedup trước khi mở heldout.

Bốn arm A3 của một lượt đo được lưu chung trong một checkpoint nguyên tử tại
`<phase>/a3_rollout_pairs/cells/`; file `a3_rollout.jsonl` vẫn có từng arm và
summary theo schema cũ. Retry đo lại cả bộ bốn arm, không trộn arm từ các lượt
đo khác nhau. Lỗi ở cấu hình refresh phụ chỉ hạ kết luận của cấu hình đó; không
chặn cấu hình được chọn nếu cấu hình được chọn vẫn có đầy đủ dữ liệu hợp lệ.

Trace được tách theo phase dưới `traces/<phase>/`. Pilot tái sử dụng trace đã
validate nếu chữ ký đầy đủ khớp, còn dev/confirm có trace riêng theo horizon
của phase. Chữ ký model bỏ qua metadata download, notebook checkpoint và
bytecode; thay đổi weights/config/tokenizer thực vẫn làm chữ ký đổi.

Sau khi `start` trả về, theo dõi log bằng lệnh `tail -f` mà nó in ra. Kết quả
nằm ở `outputs/dflash_context_interventions/<run_id>/`; báo cáo cuối là
`reports/confirm_decision.md` và `reports/confirm_coverage.json`.

## Exposure và quota

Probe và pilot cũ đã được tổng hợp trong
`data/dflash_context_A_audit/combined_exposure_manifest.json`; bằng chứng đối
soát nằm trong `exposure_audit.json`. `prepare` tự tạo split mới trong output,
không cần file split bên ngoài. Lệnh `check` so hash ba dataset với corpus đã
audit và dừng nếu dữ liệu khác.

Với corpus `longbench_100_14k` hiện tại, audit để lại 37 GovReport, 33 Multi-News
và **3 QMSum** groups trong test; kế hoạch ban đầu cần 18/18/12. Vì vậy runner
dùng toàn bộ nhóm chưa xem còn lại: confirm thực tế là **18/18/3 = 39 groups**.
Báo cáo đánh dấu confirm là partial và giới hạn kết luận; không được diễn giải
như đã đạt quota 48. Đây là thu hẹp theo quy tắc exposure trong kế hoạch, tránh
đưa dữ liệu đã xem vào heldout.

Nếu parity hoặc gate lock không đạt, runner dừng ở gate đó và không mở confirm.
Khi ấy vẫn giữ report của phase đã hoàn tất trong cùng thư mục run. Không có
Modal job hoặc package mới trong quy trình này.
