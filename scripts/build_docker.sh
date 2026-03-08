#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME=${1:-ca-server}
IMAGE_TAG=${2:-latest}

PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}
INSTALL_TORCH=${INSTALL_TORCH:-1}
INSTALL_TORCH_NPU=${INSTALL_TORCH_NPU:-0}
TORCH_VERSION=${TORCH_VERSION:-2.6.0}
TORCH_NPU_VERSION=${TORCH_NPU_VERSION:-2.6.0.post5}
TORCH_EXTRA_INDEX_URL=${TORCH_EXTRA_INDEX_URL:-}
TORCH_FIND_LINKS=${TORCH_FIND_LINKS:-}

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} from ${ROOT_DIR}" >&2

docker build \
  --file "${ROOT_DIR}/Dockerfile" \
  --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}" \
  --build-arg "INSTALL_TORCH=${INSTALL_TORCH}" \
  --build-arg "INSTALL_TORCH_NPU=${INSTALL_TORCH_NPU}" \
  --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
  --build-arg "TORCH_NPU_VERSION=${TORCH_NPU_VERSION}" \
  --build-arg "TORCH_EXTRA_INDEX_URL=${TORCH_EXTRA_INDEX_URL}" \
  --build-arg "TORCH_FIND_LINKS=${TORCH_FIND_LINKS}" \
  --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
  "${ROOT_DIR}"

echo "Docker image ${IMAGE_NAME}:${IMAGE_TAG} built successfully" >&2
