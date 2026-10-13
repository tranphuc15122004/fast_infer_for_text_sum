# Chạy campaign A1–A3 trên B200

## Chỉ cần hai lệnh

Chạy từ thư mục repo trên server B200:

```bash
bash scripts/run_dflash_context_A_b200.sh check
bash scripts/run_dflash_context_A_b200.sh start
```

`check` kiểm tra Python/packages, GPU/FA4, models, dữ liệu, hash corpus,
exposure audit và số source groups. Lệnh này không chạy generation.

`start` treo campaign dưới nền. Nó tự chạy preflight → prepare → parity/validate
→ pilot → dev → lock → confirm → report. Campaign dùng GPU 0, FA4, seed 42,
và cấu hình B200 đã đăng ký trong kế hoạch. Master config được đọc qua
`config/master.path`; không cần truyền run ID, đường dẫn manifest hay config.

Gate correctness yêu cầu full adapter khớp native full DFlash. Khác biệt token
giữa DFlash và AR được ghi làm diagnostic, không chặn A; A3 báo ROUGE theo cặp
với native full DFlash và acceptance. Vì chưa đăng ký biên ROUGE, không dùng
ROUGE để tuyên bố chất lượng được bảo toàn. Protocol amendment này được ghi
nhận ngày 13/10/2026, trước khi chạy ma trận A1/A2/A3 và mở heldout.

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
