"""MR-DFlash — bản tự đóng gói (self-contained) của quy trình train DFlash.

Port từ SpecForge (``externals/SpecForge``), dùng làm gốc cho thí nghiệm
MR-DFlash: model draft + wrapper OnlineDFlashModel (block-parallel loss),
capture feature target bằng HF, dataset/collator offline, trainer spine +
checkpoint. Toàn bộ chạy độc lập với torch (+ transformers khi nạp target).
"""

__version__ = "0.1.0"
