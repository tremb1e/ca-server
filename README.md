# Continuous Authentication Server

A lightweight server for receiving and storing encrypted sensor data from Android devices.

## Features

- AES-256-GCM decryption
- LZ4 decompression
- Data validation with Pydantic
- File-based storage using JSONL format
- Comprehensive JSON logging
- Docker support
- FastAPI async framework

## Quick Start

### Using Docker Compose

```bash
docker compose up -d --build
```

### Manual Installation

```bash
pip install -r requirements.txt
# 默认同时启动：
# - gRPC 服务 (SensorDataService + gRPC health): 0.0.0.0:8000 (h2c by default)
# - 若需要 HTTP/JSON API，请将 PORT 调整为与 GRPC_PORT 不同的端口，否则会被自动禁用以保证单端口模式
python -m src.main

# 配置证书后 gRPC/HTTP 自动切换到 TLS 1.2/1.3 (ALPN h2)
TLS_CERTFILE=./certs/server.crt TLS_KEYFILE=./certs/server.key python -m src.main
# 若偏好 hypercorn CLI 仅跑 HTTP，可传递 --certfile/--keyfile 覆盖
```

## API Endpoints

- `POST /sensor-data/{device_id_hash}/{session_id}/{packet_sequence}` - Submit sensor data
- `GET /health` - Health check with storage statistics
- `GET /` - Server status
- Note: HTTP endpoints run only when `PORT` differs from `GRPC_PORT` (single-port mode keeps only gRPC).

## Configuration

Configure via environment variables (or a `.env` file) as needed:
- `HOST`, `PORT`, `HTTP_ENABLED`, `DATA_STORAGE_PATH`, `LOG_PATH`, `LOG_LEVEL`, `LOG_FORMAT`
- TLS (optional): `TLS_CERTFILE`, `TLS_KEYFILE`, `TLS_CA_CERTS`, `TLS_KEYFILE_PASSWORD`
- gRPC: `GRPC_HOST` (default `0.0.0.0`), `GRPC_PORT` (default `8000`), `GRPC_MAX_MESSAGE_SIZE`
- Concurrency: `GRPC_MAX_CONCURRENT_RPCS`, `AUTH_MAX_CONCURRENT`, `AUTH_MAX_CACHED_MODELS`, `TRAINING_MAX_CONCURRENT`
- 单端口模式：当 `GRPC_PORT == PORT` 时，HTTP/JSON API 会被自动关闭，仅保留 gRPC；如需同时提供 HTTP，请改用不同的 `PORT`。

## Testing

```bash
pytest
```

## Docker Packaging (ARM + Ascend, No CUDA)

### Build Image

- **Using helper script (default: install torch):** `scripts/build_docker.sh [image_name] [tag]`
- **Manual build (default: install torch):** `docker build -t ca-server:latest .`
- **Manual build (install torch + torch-npu in image):**
  ```bash
  docker build -t ca-server:latest \
    --build-arg INSTALL_TORCH=1 \
    --build-arg INSTALL_TORCH_NPU=1 \
    --build-arg TORCH_VERSION=2.6.0 \
    --build-arg TORCH_NPU_VERSION=2.6.0.post5 \
    --build-arg TORCH_EXTRA_INDEX_URL=<your_torch_npu_index> \
    .
  ```

### Run Container

- **docker compose（推荐）:** `docker compose up -d --build`
- **首次部署建议：**
  ```bash
  mkdir -p \
    deploy/data/raw_data \
    deploy/data/processed_data \
    deploy/data/inference \
    deploy/data/models \
    deploy/data/hmog_preprocessed \
    deploy/data/ca_train_cached_windows \
    deploy/data/ca_train_token_caches \
    deploy/data/results \
    deploy/data/ascend/kernel_meta \
    deploy/data/ascend/log \
    deploy/logs \
    deploy/certs
  ```

### Host Path Mapping

`docker-compose.yml` 已将以下路径映射到宿主机（均在项目根目录下）：

- `./deploy/config/ca_config.toml` -> `/app/ca_config.toml`
- `./deploy/config/server.env` -> `/app/.env`
- `./deploy/data/raw_data` -> `/app/data_storage/raw_data`
- `./deploy/data/processed_data` -> `/app/data_storage/processed_data`
- `./deploy/data/inference` -> `/app/data_storage/inference`
- `./deploy/data/models` -> `/app/data_storage/models`
- `./deploy/data/hmog_preprocessed` -> `/app/data_storage/hmog_preprocessed`
- `./deploy/data/ca_train_cached_windows` -> `/app/ca_train/cached_windows`
- `./deploy/data/ca_train_token_caches` -> `/app/ca_train/token_caches`
- `./deploy/data/results` -> `/app/results`
- `./deploy/data/ascend/kernel_meta` -> `/app/kernel_meta`
- `./deploy/data/ascend/log` -> `/app/ascend_logs`
- `./deploy/logs` -> `/app/logs`
- `./deploy/certs` -> `/app/certs`（可选）

此外会映射宿主机 Ascend 运行时目录（只读）：
- `/usr/local/Ascend/driver` -> `/usr/local/Ascend/driver`
- `/usr/local/Ascend/ascend-toolkit` -> `/usr/local/Ascend/ascend-toolkit`

### Verify HTTP/2 / gRPC Support

- Health check (HTTP/1.1, when HTTP enabled): `curl http://localhost:10500/health`
- HTTP/2 (h2c) probe (when HTTP enabled): `curl --http2-prior-knowledge http://localhost:10500/health`
- TLS + HTTP/2 probe (if certs provided and HTTP enabled): `curl -vk --http2 https://localhost:10500/health`
- gRPC (h2c): `grpcurl -plaintext localhost:10500 list com.continuousauth.proto.SensorDataService`
- gRPC (TLS): `grpcurl -insecure localhost:10500 list com.continuousauth.proto.SensorDataService`
- gRPC 健康检查 (h2c): `grpcurl -plaintext localhost:10500 grpc.health.v1.Health/Check`

### Configuration Tips

- 主要业务参数建议修改 `deploy/config/server.env`。
- 预处理/滑窗/认证策略建议修改 `deploy/config/ca_config.toml`。
- 若启用 TLS，请把证书放到 `deploy/certs` 并在 `deploy/config/server.env` 中启用 `TLS_CERTFILE/TLS_KEYFILE`。
- 可通过 `ASCEND_VISIBLE_DEVICES` / `ASCEND_RT_VISIBLE_DEVICES` 限定容器使用的 NPU 卡号。

## TLS / HTTP/2 Behavior

- 默认：未提供证书时以 h2c（HTTP/2 over cleartext TCP）运行，适配 gRPC 的降级策略。
- TLS：同时提供 `TLS_CERTFILE` 与 `TLS_KEYFILE` 时自动开启 TLS 1.2/1.3（可选 `TLS_CA_CERTS`、`TLS_KEYFILE_PASSWORD`）。若文件缺失会记录警告并继续使用 h2c。
- 仅启用 TLS 1.2/1.3 的密码套件；证书/密钥无法加载或配置不完整时会记录原因并自动回退到 h2c，避免阻塞调试环境。
- 推荐：使用 `curl --http2-prior-knowledge http://host:port/health` 验证 h2c，或 `curl -vk --http2 https://host:port/health` 验证 TLS。
- gRPC 与 HTTP 共享同一套证书配置：gRPC 优先 TLS，若未配置证书则自动退回明文 h2c。客户端会先探测 TLS，再自动降级到 h2c。默认端口：gRPC 8000（HTTP 若需要请使用不同端口或显式开启）。

## Data Storage

Data is stored in JSONL format:
```
data_storage/
  └── {device_id_hash}/
      └── session_{session_id}.jsonl
```

## Dataset Processing Pipeline

The server now bundles an offline pipeline to turn raw JSONL sessions into aligned datasets, merge HMOG attackers, normalize, and create sliding windows.

Key knobs live in `ca_config.toml` (中文注释，便于后期调整)：
- raw 数据触发阈值（默认 100MB）与并发进程数（默认 5）
- 需要生成/训练/认证的窗口列表（默认 0.1–1.0s）
- 连续拒绝 K 与“允许打断真实用户比例”等认证策略阈值

Run the pipeline:
```bash
python -m src.processing.cli            # process all users that have >=100MB of raw sessions
python -m src.processing.cli --user <device_hash>
```

What it does:
- Detect users with ≥100MB raw data, sort sessions by filename timestamp (if present) or mtime, then take the earliest sessions (prefix) whose combined size reaches ~100MB and resample acc/gyr/mag to 100Hz with linear interpolation (uses up to 5 processes).
- Split per-user data into ~75% train / 12.5% val / 12.5% test, then add HMOG attackers (#1 → val, #2 → test) after column/unit alignment.
- Compute Z-Score stats from train only, store `scaler.json`, and write normalized splits under `data_storage/processed_data/z-score/<user>/`.
- Generate sliding-window datasets for 0.1s–1.0s (stride=window/2, no cross-session) using up to 5 processes via `ProcessPoolExecutor`, saving to `data_storage/processed_data/window/{t}/{user}/`.

## Model Training (VQGAN-only)

The reference training code is vendored under `server/ca_train`. The server provides a thin wrapper to train per-window models from the generated window CSVs and store artifacts under `data_storage/models/<user>/`.

```bash
python -m src.training.cli --user <device_hash> --device auto
# 仅跑一个窗口尺寸（用于快速 smoke test）：
python -m src.training.cli --user <device_hash> --device cpu --window-sizes 0.1 --vqgan-epochs 1 --lm-epochs 1
```

Outputs:
- Checkpoints: `data_storage/models/<user>/checkpoints/`
- Per-window logs: `data_storage/models/<user>/logs/ws_<t>/`
- Aggregated summary: `data_storage/models/<user>/training_summary.json`
- Best lock policy: `data_storage/models/<user>/best_lock_policy.json`

## Offline Policy Search (No Retraining)

Run an offline grid search over vote policies `(N, M, target_window_frr, w, overlap)` on cached per-window scores:

```bash
python -m src.policy_search.cli --user <device_hash> --device auto                 # default: vqgan-only
python -m src.policy_search.cli --user <device_hash> --device auto --auth-method both
```

Outputs (under `data_storage/models/<user>/policy_search/`):
- `vqgan-only`: `grid_results_vqgan_only.csv` + `pareto_frontier_vqgan_only.csv` + `best_lock_policy_vqgan_only.json`
- `vqgan+transformer`: `grid_results.csv` + `pareto_frontier.csv` (and optionally overwrites `best_lock_policy.json`)
- Per-combo logs (default enabled): `per_combo/<auth_method>/t_<t>/N_<N>_M_<M>.csv` (each file groups different `target_window_frr` candidates).

## Authentication Inference (Offline)

Once `best_lock_policy.json` exists, you can run inference on any window CSV (e.g. the generated `test.csv`) and emit per-window scores/accept/reject decisions:

```bash
python -m src.authentication.cli \
  --user <device_hash> \
  --csv-path data_storage/processed_data/window/0.1/<device_hash>/test.csv \
  --device auto
```

Outputs:
- `data_storage/models/<user>/inference/infer_ws_<t>.csv`
