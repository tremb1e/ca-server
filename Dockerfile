ARG PYTHON_BASE_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_BASE_IMAGE} AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    libglib2.0-0 \
    libgomp1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG INSTALL_TORCH=1
ARG INSTALL_TORCH_NPU=1
ARG TORCH_VERSION=2.6.0
ARG TORCH_NPU_VERSION=2.6.0.post5
ARG TORCH_EXTRA_INDEX_URL=
ARG TORCH_FIND_LINKS=
ARG BUNDLE_TOOL=pyinstaller
ARG PYINSTALLER_VERSION=6.16.0
ARG NUITKA_VERSION=2.6.8
ARG GRPC_HEALTH_PROBE_VERSION=v0.4.38

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

RUN python -m venv "${VIRTUAL_ENV}"

COPY requirements.txt requirements-ml.txt ./

RUN python -m pip install --retries 5 setuptools wheel

RUN python -m pip install --retries 5 -r requirements.txt -r requirements-ml.txt

RUN if [ "${INSTALL_TORCH}" = "1" ] || [ "${INSTALL_TORCH_NPU}" = "1" ]; then \
      if [ -n "${TORCH_FIND_LINKS}" ]; then \
        python -m pip install --retries 5 --find-links "${TORCH_FIND_LINKS}" "torch==${TORCH_VERSION}"; \
      elif [ -n "${TORCH_EXTRA_INDEX_URL}" ]; then \
        python -m pip install --retries 5 --extra-index-url "${TORCH_EXTRA_INDEX_URL}" "torch==${TORCH_VERSION}"; \
      else \
        python -m pip install --retries 5 "torch==${TORCH_VERSION}"; \
      fi; \
    fi

RUN if [ "${INSTALL_TORCH_NPU}" = "1" ]; then \
      if [ -n "${TORCH_FIND_LINKS}" ]; then \
        python -m pip install --retries 5 --find-links "${TORCH_FIND_LINKS}" "torch-npu==${TORCH_NPU_VERSION}"; \
      elif [ -n "${TORCH_EXTRA_INDEX_URL}" ]; then \
        python -m pip install --retries 5 --extra-index-url "${TORCH_EXTRA_INDEX_URL}" "torch-npu==${TORCH_NPU_VERSION}"; \
      else \
        python -m pip install --retries 5 "torch-npu==${TORCH_NPU_VERSION}"; \
      fi; \
    fi

RUN arch="$(dpkg --print-architecture)" && \
    case "${arch}" in \
      arm64) probe_arch='linux-arm64' ;; \
      amd64) probe_arch='linux-amd64' ;; \
      *) echo "Unsupported architecture: ${arch}" >&2; exit 1 ;; \
    esac && \
    curl --retry 5 --retry-all-errors --connect-timeout 30 --max-time 300 -fsSL -o /tmp/grpc_health_probe \
      "https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-${probe_arch}" && \
    chmod +x /tmp/grpc_health_probe


FROM ${PYTHON_BASE_IMAGE} AS runtime

WORKDIR /app

# Runtime intentionally consumes the host Ascend driver/toolkit via bind mounts.
# Do not install Ascend driver packages into the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    libglib2.0-0 \
    libgomp1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /tmp/grpc_health_probe /usr/local/bin/grpc_health_probe
COPY src/ /app/src/
COPY ca_train/ /app/ca_train/
COPY ca_config.toml /app/ca_config.toml
COPY deploy/entrypoint.sh /usr/local/bin/ca-entrypoint.sh

RUN mkdir -p /usr/local/Ascend/driver /usr/local/Ascend/ascend-toolkit && \
    chmod +x /usr/local/bin/ca-entrypoint.sh /usr/local/bin/grpc_health_probe && \
    mkdir -p \
      /app/data_storage/raw_data \
      /app/data_storage/processed_data \
      /app/data_storage/inference \
      /app/data_storage/models \
      /app/data_storage/hmog_preprocessed \
      /app/runtime/ca_train/cached_windows \
      /app/runtime/ca_train/token_caches \
      /app/results \
      /app/kernel_meta \
      /app/ascend_logs \
      /app/logs \
      /app/certs

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:${PATH}
ENV APP_ROOT=/app \
    CA_CONFIG_PATH=/app/ca_config.toml \
    CA_SERVER_RUNTIME=python \
    HOST=0.0.0.0 \
    PORT=18000 \
    HTTP_ENABLED=false \
    GRPC_HOST=0.0.0.0 \
    GRPC_PORT=8000 \
    DATA_STORAGE_PATH=/app/data_storage/raw_data \
    PROCESSED_DATA_PATH=/app/data_storage/processed_data \
    INFERENCE_STORAGE_PATH=/app/data_storage/inference \
    HMOG_DATA_PATH=/app/data_storage/hmog_preprocessed \
    CA_TRAIN_WINDOW_CACHE_DIR=/app/runtime/ca_train/cached_windows \
    CA_TRAIN_TOKEN_CACHE_DIR=/app/runtime/ca_train/token_caches \
    CA_RESULTS_PATH=/app/results \
    LOG_PATH=/app/logs \
    ASCEND_DRIVER_HOME=/usr/local/Ascend/driver \
    ASCEND_INSTALL_INFO=/etc/ascend_install.info \
    ASCEND_TOOLKIT_ROOT=/usr/local/Ascend/ascend-toolkit \
    ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp \
    ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    TOOLCHAIN_HOME=/usr/local/Ascend/ascend-toolkit/latest/toolkit \
    ASCEND_PROCESS_LOG_PATH=/app/ascend_logs \
    ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV PATH=/opt/venv/bin:/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/tools/ccec_compiler/bin:/usr/local/sbin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
ENV PYTHONPATH=/app:/app/ca_train:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe

EXPOSE 8000 18000

ENTRYPOINT ["/usr/local/bin/ca-entrypoint.sh"]
CMD ["server"]
