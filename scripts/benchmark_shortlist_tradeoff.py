#!/usr/bin/env python3
"""Entrypoint tương thích cho đánh giá Phase A có metadata.

Dùng --target-dir và --output-dir mới; xem docs/_INSIGHT/MASTER_PHASE_A_COMPREHENSIVE_REPORT.md.
Các kết quả cũ giữ nguyên trong outputs; không tái sinh bảng từ vector/query hoặc layer thiếu.
"""
from run_candidate_benchmarks import main

if __name__ == '__main__':
    main()
