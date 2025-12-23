FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX_URL=https://pypi.org/simple

COPY requirements.txt .
COPY requirements-ml.txt .
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_INDEX_URL=${PIP_INDEX_URL}
RUN python -m pip install --retries 5 -r requirements.txt -r requirements-ml.txt

COPY src/ ./src/
COPY ca_train/ ./ca_train/
COPY ca_config.toml ./ca_config.toml

RUN mkdir -p /app/data_storage/raw_data /app/data_storage/processed_data /app/data_storage/inference /app/data_storage/models /app/logs

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "src.main"]
