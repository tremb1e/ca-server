#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME=${1:-ca-server}
IMAGE_TAG=${2:-latest}
DOCKERFILE_PATH=${CA_SERVER_DOCKERFILE:-Dockerfile}
EXPORT_DIR=${EXPORT_DIR:-"${ROOT_DIR}/dist/docker-images"}
SAFE_IMAGE_NAME="$(printf '%s' "${IMAGE_NAME}" | tr '/:' '__')"
EXPORT_PATH=${EXPORT_PATH:-"${EXPORT_DIR}/${SAFE_IMAGE_NAME}_${IMAGE_TAG}.tar"}

PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}
INSTALL_TORCH=${INSTALL_TORCH:-1}
INSTALL_TORCH_NPU=${INSTALL_TORCH_NPU:-1}
TORCH_VERSION=${TORCH_VERSION:-2.6.0}
TORCH_NPU_VERSION=${TORCH_NPU_VERSION:-2.6.0.post5}
TORCH_EXTRA_INDEX_URL=${TORCH_EXTRA_INDEX_URL:-}
TORCH_FIND_LINKS=${TORCH_FIND_LINKS:-}
BUNDLE_TOOL=${BUNDLE_TOOL:-pyinstaller}
PYINSTALLER_VERSION=${PYINSTALLER_VERSION:-6.16.0}
NUITKA_VERSION=${NUITKA_VERSION:-2.6.8}
GRPC_HEALTH_PROBE_VERSION=${GRPC_HEALTH_PROBE_VERSION:-v0.4.38}
PYTHON_BASE_IMAGE=${PYTHON_BASE_IMAGE:-python:3.12-slim-bookworm}

mkdir -p "${EXPORT_DIR}"

if docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" >/dev/null 2>&1; then
  echo "Removing existing image ${IMAGE_NAME}:${IMAGE_TAG}" >&2
  docker image rm -f "${IMAGE_NAME}:${IMAGE_TAG}" >/dev/null
else
  echo "No existing image ${IMAGE_NAME}:${IMAGE_TAG} found; continuing with a clean build" >&2
fi

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} from ${ROOT_DIR} using ${DOCKERFILE_PATH}" >&2

docker build \
  --file "${ROOT_DIR}/${DOCKERFILE_PATH}" \
  --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}" \
  --build-arg "INSTALL_TORCH=${INSTALL_TORCH}" \
  --build-arg "INSTALL_TORCH_NPU=${INSTALL_TORCH_NPU}" \
  --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
  --build-arg "TORCH_NPU_VERSION=${TORCH_NPU_VERSION}" \
  --build-arg "TORCH_EXTRA_INDEX_URL=${TORCH_EXTRA_INDEX_URL}" \
  --build-arg "TORCH_FIND_LINKS=${TORCH_FIND_LINKS}" \
  --build-arg "BUNDLE_TOOL=${BUNDLE_TOOL}" \
  --build-arg "PYINSTALLER_VERSION=${PYINSTALLER_VERSION}" \
  --build-arg "NUITKA_VERSION=${NUITKA_VERSION}" \
  --build-arg "GRPC_HEALTH_PROBE_VERSION=${GRPC_HEALTH_PROBE_VERSION}" \
  --build-arg "PYTHON_BASE_IMAGE=${PYTHON_BASE_IMAGE}" \
  --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
  "${ROOT_DIR}"

echo "Exporting ${IMAGE_NAME}:${IMAGE_TAG} to ${EXPORT_PATH}" >&2
rm -f "${EXPORT_PATH}"
docker save --output "${EXPORT_PATH}" "${IMAGE_NAME}:${IMAGE_TAG}"

echo "Docker image ${IMAGE_NAME}:${IMAGE_TAG} built and exported successfully" >&2
