# API 功能测试报告

测试时间：2026-04-28 10:08 CST  
提交：本报告随本地 `添加api调用功能` 提交提交，最终哈希以 `git rev-parse HEAD` 为准  
镜像：`ca-server-api:latest` (`sha256:477e5abc99fb0631bb91cca1e9532256b557791bceef70a90cba2c6ee9cce94a`)

## 测试环境

| 项目 | 结果 |
| --- | --- |
| 宿主机 | Linux aarch64 |
| Docker | `28.2.2` |
| Compose | 本机未安装 `docker compose` / `docker-compose`，本次用等价 `docker build` + `docker run --network host` 验证 |
| 构建代理 | `HTTP_PROXY=http://127.0.0.1:9999`、`HTTPS_PROXY=http://127.0.0.1:9999`，并设置 `--network=host` |
| 容器 | `ca-server` (`07e4a809c3e6`)，镜像 `ca-server-api:latest` |
| 端口 | gRPC `10500`，HTTP/管理 API `18000` |
| 管理 API key | `ca-server-api-test-key` |

## 本地测试

| 项目 | 结果 |
| --- | --- |
| `python3 -m compileall -q src tests` | 通过 |
| `bash -n deploy/entrypoint.sh scripts/build_docker.sh scripts/build_frozen_bundle.sh` | 通过 |
| `docker-compose.yml` YAML 解析 | 通过 |
| Docker 内挂载源码执行 `python -m pytest -q` | 通过，`82 passed` |

## 镜像构建

| 项目 | 结果 |
| --- | --- |
| 构建命令 | `docker build --network=host --build-arg HTTP_PROXY=http://127.0.0.1:9999 --build-arg HTTPS_PROXY=http://127.0.0.1:9999 --build-arg NO_PROXY=localhost,127.0.0.1 -t ca-server-api:latest .` |
| 构建结果 | 通过 |
| 镜像 ID | `sha256:477e5abc99fb0631bb91cca1e9532256b557791bceef70a90cba2c6ee9cce94a` |
| 镜像大小 | `1225827884` bytes，约 `1.23 GB` |

## 容器部署

| 项目 | 结果 |
| --- | --- |
| 停止并移除旧 `ca-server` 容器 | 通过 |
| 使用新 `ca-server-api:latest` 启动 `ca-server` | 通过 |
| 网络模式 | `host` |
| 关键挂载 | raw、processed、inference、models、hmog、训练缓存、results、Ascend 日志、配置、证书、宿主 Ascend driver/toolkit 均已挂载 |
| 容器重启次数 | `0` |
| gRPC 监听 | `0.0.0.0:10500` |
| HTTP 监听 | `0.0.0.0:18000`，由 `MANAGEMENT_API_ENABLED=true` 自动启用 |

## 功能测试

| 功能 | 接口/命令 | 结果 |
| --- | --- | --- |
| gRPC 健康检查 | `docker exec ca-server grpc_health_probe -addr=127.0.0.1:10500` | `SERVING` |
| HTTP 健康检查 | `GET /health` | `200 healthy` |
| 管理 API 无 key | `GET /api/v1/management/summary` | `401 missing_management_api_key` |
| 管理 API 错 key | `GET /api/v1/management/summary` | `403 invalid_management_api_key` |
| 管理 API 正确 key | `GET /api/v1/management/summary` | `200` |
| 运行态查询 | `GET /api/v1/management/runtime` | `200`，HTTP enabled=true，backend=`npu` |
| OpenAPI | `GET /openapi.json` | 包含管理 API、`/api/v1/sensor-data` 和 `APIKeyHeader` |
| 路径安全 | `GET /api/v1/management/devices/bad%5Cdevice` | `400` |
| HTTP 加密上传 | string 设备/session ID，gzip + AES-GCM，`POST /api/v1/sensor-data` | `200 {"status":"ok"}` |
| HTTP 解压上限 | 11 MiB 明文 gzip 后加密上传 | `400 {"reason":"decompression_failed"}` |
| HTTP raw 查询 | `GET /api/v1/management/devices/api-string-device-2/raw-sessions` | `packet_count=1` |
| gRPC 策略接口 | `GetInitialPolicy` | `policy_id=default`，`max_payload_size_bytes=4194304`，与 gRPC message limit 一致 |
| gRPC 心跳 | `SendHeartbeat` | `client_timestamp_echo=123` |
| gRPC 指标上报 | `ReportMetrics` | `accepted=True` |
| gRPC 流式上传 | `StreamSensorData` 上传合法 `SerializedSensorBatch` | Ack `success=True` |
| gRPC raw 查询 | `GET /api/v1/management/devices/grpc-stream-device-2/raw-sessions` | `packet_count=1` |
| 客户端指标 API | `GET /api/v1/management/client-metrics` | 可读出 gRPC 上报指标，含 `received_at` |
| 运行指标 | `GET /api/v1/management/runtime` | counters 包含 `packets_received=3`、`packets_stored=2`、`packets_not_stored=1`、`client_metric_reports=1` |
| Ascend 可见性 | `npu-smi info` | 8 张 `910B2` 均 `OK` |
| 后端探测 | `detect_backend("auto")` | `npu` |

## 结论

新增和修正后的 API 调用链路通过验证：HTTP 兼容上传、gRPC 策略/心跳/指标/流式上传、管理 API 查询、OpenAPI、路径安全、解压上限、运行指标、容器挂载和 NPU 运行环境均正常。当前宿主机上的 `ca-server` 容器正在运行新镜像 `ca-server-api:latest`。

## 已知说明

- 本机没有 Docker Compose CLI，`docker-compose.yml` 已做 YAML 解析检查，但未执行 `docker compose up`。
- 当前服务未配置 TLS 证书，gRPC 和 HTTP 均按项目默认策略以明文监听本机端口；生产暴露前应配置 TLS/mTLS 或通过受控反向代理访问。
