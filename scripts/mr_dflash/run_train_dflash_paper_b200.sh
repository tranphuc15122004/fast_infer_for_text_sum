#!/usr/bin/env bash
# ==============================================================================
# Launcher: Train DFlash (Qwen3-4B 5-Layer Draft) trên server B200 theo chuẩn Paper
# Dữ liệu: 50K Phase 1 Full Cache (Context 16K, Output 1K)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Cấu hình mặc định
CONFIG_FILE="${REPO_ROOT}/src/MR_DFlash/configs/train_qwen3_4b_dflash_paper.yaml"
TARGET_MODEL="/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B"
RUN_ROOT="/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_phase1_50k_50_50_10k_b200_gpu0_ctx16k_out1k_20260928T132928Z"
OUTPUT_DIR="/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/dflash_qwen3_4b_trained_paper"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
    cat << 'EOF'
Cách sử dụng:
  bash scripts/mr_dflash/run_train_dflash_paper_b200.sh [LỆNH] [OPTIONS]

Các lệnh chính:
  check        : Kiểm tra preflight (Model, Cache manifest, VRAM, Python env)
  train        : Chạy train trực tiếp trên 1 GPU (Foreground)
  train-ddp    : Chạy train song song trên 2 GPU DDP (torchrun)
  bg           : Chạy train ngầm (Nohup background), tự ghi PID & logs
  status       : Xem trạng thái và metrics của job đang chạy
  stop         : Dừng tiến trình train ngầm an toàn

Options:
  --gpu <ID>            : Chọn GPU ID (mặc định: 0)
  --output-dir <PATH>   : Ghi đè thư mục output
  --config <PATH>       : Ghi đè file config YAML
  -h, --help            : Hiển thị trợ giúp này

Ví dụ:
  bash scripts/mr_dflash/run_train_dflash_paper_b200.sh check
  bash scripts/mr_dflash/run_train_dflash_paper_b200.sh train --gpu 0
  bash scripts/mr_dflash/run_train_dflash_paper_b200.sh bg --gpu 0
  bash scripts/mr_dflash/run_train_dflash_paper_b200.sh status
EOF
    exit 0
}

COMMAND="${1:-train}"
if [[ "$#" -gt 0 ]]; then
    shift
fi

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Lỗi: Tham số không xác định '$1'" >&2
            usage
            ;;
    esac
done

cd "${REPO_ROOT}"

check_preflight() {
    echo "================================================================="
    echo "[PREFLIGHT CHECK] Kiểm tra tài nguyên và cấu hình DFlash"
    echo "================================================================="
    echo "Config file : ${CONFIG_FILE}"
    echo "Target model: ${TARGET_MODEL}"
    echo "Feature root: ${RUN_ROOT}"
    echo "Output root : ${OUTPUT_DIR}"
    echo "GPU ID      : ${GPU_ID}"
    echo "-----------------------------------------------------------------"

    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "[ERROR] Không tìm thấy file config: ${CONFIG_FILE}" >&2
        exit 1
    fi

    if [[ ! -d "${TARGET_MODEL}" ]]; then
        echo "[WARNING] Thư mục Target Model chưa tồn tại tại: ${TARGET_MODEL}"
        echo "          (Bỏ qua nếu đang chạy trên máy local không gắn ổ /workspace)"
    else
        echo "[OK] Target Model tồn tại."
    fi

    TRAIN_CACHE="${RUN_ROOT}/target_features_qwen3_4b_full/train"
    if [[ ! -d "${TRAIN_CACHE}" ]]; then
        echo "[WARNING] Thư mục Feature Cache chưa tồn tại tại: ${TRAIN_CACHE}"
    else
        echo "[OK] Feature Cache Train tồn tại."
        if [[ -f "${TRAIN_CACHE}/manifest.json" ]]; then
            echo "[OK] Manifest Cache Train hợp lệ."
        fi
    fi

    echo "[INFO] Kiểm tra Python config schema..."
    PYTHONPATH=src python3 -c "from MR_DFlash.run_train import load_run_config; cfg = load_run_config('${CONFIG_FILE}'); print('[OK] Config load thành công:', cfg.run_id, 'Strategy:', cfg.training.strategy)"
    echo "================================================================="
    echo "[PREFLIGHT CHECK] HOÀN TẤT - Sẵn sàng khởi chạy."
    echo "================================================================="
}

run_train_single() {
    mkdir -p "${OUTPUT_DIR}"
    echo "[START] Khởi chạy DFlash Training trên GPU ${GPU_ID}..."
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export FI_OFFLINE=1
    export PYTHONPATH=src

    python3 -m MR_DFlash.run_train \
        --config "${CONFIG_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --device cuda
}

run_train_ddp() {
    mkdir -p "${OUTPUT_DIR}"
    echo "[START] Khởi chạy DFlash Training DDP trên 2 GPU..."
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export FI_OFFLINE=1
    export PYTHONPATH=src

    torchrun --standalone --nproc_per_node=2 -m MR_DFlash.run_train \
        --config "${CONFIG_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --device cuda \
        --batch-size 2 \
        --accumulation-steps 1
}

run_train_background() {
    mkdir -p "${OUTPUT_DIR}/logs"
    LOG_FILE="${OUTPUT_DIR}/logs/train.log"
    PID_FILE="${OUTPUT_DIR}/logs/train.pid"

    if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "[ERROR] Job đang chạy với PID: $(cat "${PID_FILE}"). Dùng 'stop' hoặc 'status'." >&2
        exit 1
    fi

    echo "[START] Khởi chạy DFlash Training ngầm (Background)..."
    echo "Logs sẽ được lưu tại: ${LOG_FILE}"

    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export FI_OFFLINE=1
    export PYTHONPATH=src

    nohup python3 -m MR_DFlash.run_train \
        --config "${CONFIG_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --device cuda \
        > "${LOG_FILE}" 2>&1 &

    TRAIN_PID=$!
    echo "${TRAIN_PID}" > "${PID_FILE}"
    echo "[OK] Đã khởi chạy tiến trình PID: ${TRAIN_PID}"
    echo "Theo dõi logs bằng lệnh: tail -f ${LOG_FILE}"
    echo "Theo dõi metrics bằng lệnh: tail -f ${OUTPUT_DIR}/metrics.jsonl"
}

check_status() {
    PID_FILE="${OUTPUT_DIR}/logs/train.pid"
    LOG_FILE="${OUTPUT_DIR}/logs/train.log"
    METRICS_FILE="${OUTPUT_DIR}/metrics.jsonl"

    echo "================================================================="
    echo "[STATUS] Trạng thái job DFlash Training"
    echo "================================================================="
    if [[ -f "${PID_FILE}" ]]; then
        PID="$(cat "${PID_FILE}")"
        if kill -0 "${PID}" 2>/dev/null; then
            echo "Tiến trình : ĐANG CHẠY (PID: ${PID})"
        else
            echo "Tiến trình : ĐÃ DỪNG / KẾT THÚC (PID cũ: ${PID})"
        fi
    else
        echo "Tiến trình : Không tìm thấy PID file."
    fi

    if [[ -f "${METRICS_FILE}" ]]; then
        echo "-----------------------------------------------------------------"
        echo "5 bước metrics gần nhất:"
        tail -n 5 "${METRICS_FILE}"
    fi

    if [[ -f "${LOG_FILE}" ]]; then
        echo "-----------------------------------------------------------------"
        echo "10 dòng logs gần nhất (${LOG_FILE}):"
        tail -n 10 "${LOG_FILE}"
    fi
    echo "================================================================="
}

stop_train() {
    PID_FILE="${OUTPUT_DIR}/logs/train.pid"
    if [[ -f "${PID_FILE}" ]]; then
        PID="$(cat "${PID_FILE}")"
        if kill -0 "${PID}" 2>/dev/null; then
            echo "[INFO] Đang dừng tiến trình PID: ${PID}..."
            kill -TERM "${PID}"
            sleep 2
            if kill -0 "${PID}" 2>/dev/null; then
                kill -9 "${PID}"
            fi
            echo "[OK] Đã dừng thành công."
            rm -f "${PID_FILE}"
        else
            echo "[INFO] Tiến trình PID: ${PID} không còn chạy."
            rm -f "${PID_FILE}"
        fi
    else
        echo "[INFO] Không tìm thấy PID file."
    fi
}

case "${COMMAND}" in
    check)
        check_preflight
        ;;
    train)
        check_preflight
        run_train_single
        ;;
    train-ddp)
        check_preflight
        run_train_ddp
        ;;
    bg)
        check_preflight
        run_train_background
        ;;
    status)
        check_status
        ;;
    stop)
        stop_train
        ;;
    *)
        echo "Lỗi: Lệnh '${COMMAND}' không hợp lệ." >&2
        usage
        ;;
esac
