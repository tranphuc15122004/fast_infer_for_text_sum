#!/usr/bin/env bash
set -euo pipefail

# Usage: ./build_docker.sh [CUDA_VERSION]
CUDA="${1:-12.6.1}"          # default 12.6.1

IMAGE="sglang_sssd:latest"

docker build \
  --build-arg CUDA_VERSION="${CUDA}" \
  -t "${IMAGE}" \
  .

echo "Docker image '${IMAGE}' built successfully."
