#!/usr/bin/env bash
# Use to run built docker image
set -euo pipefail

IMAGE_TAG="${1:-sglang_sssd:latest}"

docker run --gpus all \
  # --shm-size 32g \
  --ipc=host \
  -it "${IMAGE_TAG}" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e UPLOAD_RESULTS="${UPLOAD_RESULTS:-false}" \
  -e RUN_HYPERPARAMETER_SEARCH="${RUN_HYPERPARAMETER_SEARCH:-false}" \
