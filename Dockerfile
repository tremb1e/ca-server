FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG INSTALL_TORCH=1
ARG INSTALL_TORCH_NPU=0
ARG TORCH_VERSION=2.6.0
ARG TORCH_NPU_VERSION=2.6.0.post5
ARG TORCH_EXTRA_INDEX_URL=
ARG TORCH_FIND_LINKS=

COPY requirements.txt .
COPY requirements-ml.txt .
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --retries 5 -r requirements.txt -r requirements-ml.txt

# Install torch by default (required by server import path).
RUN if [ "${INSTALL_TORCH}" = "1" ] || [ "${INSTALL_TORCH_NPU}" = "1" ]; then \
      if [ -n "${TORCH_FIND_LINKS}" ]; then \
        python -m pip install --retries 5 --find-links "${TORCH_FIND_LINKS}" "torch==${TORCH_VERSION}"; \
      elif [ -n "${TORCH_EXTRA_INDEX_URL}" ]; then \
        python -m pip install --retries 5 --extra-index-url "${TORCH_EXTRA_INDEX_URL}" "torch==${TORCH_VERSION}"; \
      else \
        python -m pip install --retries 5 "torch==${TORCH_VERSION}"; \
      fi; \
    fi

# Optional: install torch_npu when wheel source/index is available.
RUN if [ "${INSTALL_TORCH_NPU}" = "1" ]; then \
      if [ -n "${TORCH_FIND_LINKS}" ]; then \
        python -m pip install --retries 5 --find-links "${TORCH_FIND_LINKS}" "torch-npu==${TORCH_NPU_VERSION}"; \
      elif [ -n "${TORCH_EXTRA_INDEX_URL}" ]; then \
        python -m pip install --retries 5 --extra-index-url "${TORCH_EXTRA_INDEX_URL}" "torch-npu==${TORCH_NPU_VERSION}"; \
      else \
        python -m pip install --retries 5 "torch-npu==${TORCH_NPU_VERSION}"; \
      fi; \
    fi

COPY src/ ./src/
COPY ca_train/ ./ca_train/
COPY ca_config.toml ./ca_config.toml
COPY deploy/entrypoint.sh /usr/local/bin/ca-entrypoint.sh
RUN chmod +x /usr/local/bin/ca-entrypoint.sh

RUN mkdir -p \
    /app/data_storage/raw_data \
    /app/data_storage/processed_data \
    /app/data_storage/inference \
    /app/data_storage/models \
    /app/data_storage/hmog_preprocessed \
    /app/ca_train/cached_windows \
    /app/ca_train/token_caches \
    /app/results \
    /app/kernel_meta \
    /app/ascend_logs \
    /app/logs \
    /app/certs

ENV PYTHONPATH=/app \
    HOST=0.0.0.0 \
    PORT=8000 \
    HTTP_ENABLED=false \
    GRPC_HOST=0.0.0.0 \
    GRPC_PORT=8000 \
    DATA_STORAGE_PATH=/app/data_storage/raw_data \
    PROCESSED_DATA_PATH=/app/data_storage/processed_data \
    INFERENCE_STORAGE_PATH=/app/data_storage/inference \
    HMOG_DATA_PATH=/app/data_storage/hmog_preprocessed \
    LOG_PATH=/app/logs \
    ASCEND_PROCESS_LOG_PATH=/app/ascend_logs \
    ASCEND_VISIBLE_DEVICES=0 \
    ASCEND_RT_VISIBLE_DEVICES=0

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/ca-entrypoint.sh"]
CMD ["python", "-m", "src.main"]
