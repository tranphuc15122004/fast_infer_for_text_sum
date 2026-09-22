# Baseline runners

Thư mục này chứa các launcher nhỏ `run_*.sh` cho từng baseline và các smoke
runner chuyên biệt. Entry point thông thường vẫn là:

```bash
bash scripts/run.sh <baseline>
```

`run.sh` chuyển tiếp tới wrapper tương ứng trong thư mục này. Các entrypoint
điều phối chính như LongBench, representative benchmark và B200 smoke vẫn nằm
trực tiếp dưới `scripts/` để dễ tìm và dùng thường xuyên.
