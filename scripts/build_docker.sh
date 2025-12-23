#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME=${1:-continuous-auth-server-lite}
IMAGE_TAG=${2:-latest}

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} from ${ROOT_DIR}" >&2

docker build \
  --file "${ROOT_DIR}/Dockerfile" \
  --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
  "${ROOT_DIR}"

echo "Docker image ${IMAGE_NAME}:${IMAGE_TAG} built successfully" >&2
