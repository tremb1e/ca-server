# Continuous Authentication Server 外部 API 接口说明

本文档面向外部管理程序和集成方，说明本服务对外可调用的 HTTP 管理 API、现有 HTTP 数据接收接口，以及 gRPC 业务接口。

## 1. 基础信息

### 1.1 服务端口

默认部署方式以 gRPC 为主：

| 类型 | 默认端口 | 说明 |
| --- | ---: | --- |
| gRPC | `10500` | App 主交互链路，包含策略、认证会话、传感器数据流、心跳、客户端指标上报 |
| HTTP | `18000` | 当 `HTTP_ENABLED=true` 或 `MANAGEMENT_API_ENABLED=true`，且 `PORT != GRPC_PORT` 时监听 |

管理 API 是 HTTP API。直接运行 Python 服务时使用 `PORT`；通过本项目 `docker-compose.yml` 启动时，外层变量 `CA_SERVER_HTTP_PORT` 会映射到容器内的 `PORT`。设置 `MANAGEMENT_API_ENABLED=true` 时服务会自动启用 HTTP；如果 HTTP 与 gRPC 配置为同一个端口，服务会拒绝启动并提示拆分端口。

直接运行服务示例：

```bash
PORT=18000
MANAGEMENT_API_ENABLED=true
MANAGEMENT_API_KEY=<long-random-token>
```

Docker Compose 示例：

```bash
CA_SERVER_HTTP_PORT=18000 \
MANAGEMENT_API_ENABLED=true \
MANAGEMENT_API_KEY='replace-with-a-long-random-token' \
docker compose up -d --build
```

### 1.2 管理 API 鉴权

所有 `/api/v1/management/*` 接口都需要请求头：

```http
X-Management-API-Key: <MANAGEMENT_API_KEY>
```

鉴权错误：

| HTTP 状态码 | 场景 |
| ---: | --- |
| `404` | `MANAGEMENT_API_ENABLED=false`，管理路由未注册 |
| `401` | 未提供 `X-Management-API-Key` |
| `403` | API key 不正确 |
| `503` | 管理 API 已启用但未配置 `MANAGEMENT_API_KEY` |

### 1.3 OpenAPI 文档

HTTP 服务开启后，可访问：

```text
GET /docs
GET /openapi.json
```

`/docs` 和 `/openapi.json` 用于查看接口定义。未启用管理 API 时 schema 中不包含 `/api/v1/management/*` 路由；实际调用管理接口仍需要 `X-Management-API-Key`。

### 1.4 数据安全边界

管理 API 只返回运行态、状态、统计和模型策略摘要，不返回：

- 原始传感器样本明细；
- 加密 payload；
- scaler 均值/方差明细；
- 模型权重内容。

模型大小通过 checkpoint/config/policy 文件大小统计，不会默认加载模型权重。

## 2. 管理 API

管理 API Base URL：

```text
http://<host>:<http_port>/api/v1/management
```

调用示例公共请求头：

```bash
BASE_URL=http://localhost:18000
API_KEY=replace-with-a-long-random-token

curl -H "X-Management-API-Key: ${API_KEY}" \
  "${BASE_URL}/api/v1/management/summary"
```

### 2.1 获取全局概览

```http
GET /api/v1/management/summary
```

用途：外部管理程序首页概览，查看服务是否运行、设备数量、模型数量、活跃认证会话数量、训练任务数量、最近错误。

响应字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `status` | string | 固定为 `ok` |
| `runtime` | object | 服务名、版本、启动时间、运行秒数 |
| `storage.raw` | object | raw 数据根目录、设备数、session 数、总大小 |
| `storage.processed_root` | string | 预处理数据根目录 |
| `storage.inference_root` | string | 认证推理结果根目录 |
| `storage.models_root` | string | 模型目录 |
| `counts.devices` | int | 已发现设备数 |
| `counts.models` | int | 已发现模型策略数 |
| `counts.active_auth_sessions` | int | 当前内存中的活跃认证会话数 |
| `counts.active_training_tasks` | int | 当前训练任务数 |
| `recent_errors` | array | 最近包处理/存储错误 |

示例：

```bash
curl -H "X-Management-API-Key: ${API_KEY}" \
  "${BASE_URL}/api/v1/management/summary"
```

示例响应：

```json
{
  "status": "ok",
  "runtime": {
    "app_name": "Continuous Authentication Server",
    "version": "1.0.0",
    "started_at": "2026-04-28T02:00:00+00:00",
    "uptime_seconds": 3600.5
  },
  "storage": {
    "raw": {
      "base_path": "/app/data_storage/raw_data",
      "total_devices": 4,
      "total_sessions": 80,
      "total_size_bytes": 524288000,
      "total_size_mb": 500.0
    },
    "processed_root": "/app/data_storage/processed_data",
    "inference_root": "/app/data_storage/inference",
    "models_root": "/app/data_storage/models"
  },
  "counts": {
    "devices": 4,
    "models": 3,
    "active_auth_sessions": 1,
    "active_training_tasks": 0
  },
  "recent_errors": []
}
```

### 2.2 获取服务运行态

```http
GET /api/v1/management/runtime
```

用途：查看服务配置、后端设备探测、累计计数、训练任务和模型缓存。

响应字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `app_name` | string | 服务名 |
| `version` | string | 服务版本 |
| `started_at` | string | 启动时间，ISO8601 UTC |
| `uptime_seconds` | float | 运行秒数 |
| `http` | object | HTTP 是否启用、host、port |
| `grpc` | object | gRPC host、port、消息大小和并发配置 |
| `tls` | object | TLS 是否启用、证书配置、禁用原因 |
| `backend` | object | 当前推理/训练后端探测结果，如 `npu`、`cuda`、`cpu` |
| `metrics.counters` | object | 运行计数器 |
| `metrics.inference_latency` | object | 认证推理平均/最大耗时 |
| `training_tasks` | object | 训练任务快照 |
| `model_cache` | object | 已加载模型缓存快照 |

示例：

```bash
curl -H "X-Management-API-Key: ${API_KEY}" \
  "${BASE_URL}/api/v1/management/runtime"
```

常见计数器：

| 计数器 | 说明 |
| --- | --- |
| `packets_received` | 已接收 HTTP/gRPC 入站包数 |
| `packets_parsed` | 成功解析为 `SerializedSensorBatch` 的包数 |
| `packets_stored` | 成功落盘包数 |
| `packet_status_parsed_sensor_batch` | 解析成功包数 |
| `packet_status_decrypt_failed` | 解密失败包数 |
| `packet_status_decompress_failed` | 解压失败包数 |
| `packet_status_parse_failed` | protobuf 解析失败包数 |
| `packet_status_validation_failed` | HTTP JSON 兼容链路字段校验失败包数 |
| `packet_status_invalid_identifier` | 设备或 session ID 包含非法路径字符 |
| `packet_status_request_too_large` | HTTP 请求体超限 |
| `packet_status_no_data` | HTTP 请求体为空 |
| `packets_not_stored` | 未成功落盘包数，包含校验失败、解密失败、空请求等未写文件场景 |
| `packet_storage_failed` | 已进入写文件流程但文件写入失败的包数 |
| `auth_results` | 已产生认证结果数 |
| `auth_accepts` | 认证接受数 |
| `auth_rejects` | 认证拒绝数 |
| `auth_interrupts` | 触发打断数 |
| `client_metric_reports` | 客户端指标上报次数 |

### 2.3 获取设备列表

```http
GET /api/v1/management/devices
```

用途：列出服务当前从 raw 数据、processed 数据、模型目录、inference 目录中发现的所有设备。

响应字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `devices` | array | 设备摘要列表 |
| `devices[].device_id_hash` | string | 设备哈希 ID |
| `devices[].raw` | object | raw session 数、大小、最后活动时间 |
| `devices[].processed` | object | 预处理文件、窗口文件、scaler 文件摘要 |
| `devices[].model` | object | 模型是否就绪、模型版本、policy 是否存在 |
| `devices[].inference` | object | inference session 数和结果文件摘要 |
| `devices[].training` | object | 训练状态摘要 |
| `total` | int | 设备总数 |

示例：

```bash
curl -H "X-Management-API-Key: ${API_KEY}" \
  "${BASE_URL}/api/v1/management/devices"
```

### 2.4 获取单设备详情

```http
GET /api/v1/management/devices/{device_id}
```

路径参数：

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `device_id` | string | 是 | 设备哈希 ID |

用途：查看某个设备的 raw 数据、预处理数据、训练状态、模型策略、推理结果目录和活跃认证会话。

路径参数中的 `device_id`、以及认证结果查询中的 `session_id` 只能作为单个路径段使用，长度不能超过 255 字符，不能包含 `/`、`\`、NUL 字符、空字符串、`.` 或 `..`。

示例：

```bash
DEVICE_ID='ZpasvncxwsqVF1U847rDG_CjLvW0trsdrgHpsdicwYw='

curl -H "X-Management-API-Key: ${API_KEY}" \
  "${BASE_URL}/api/v1/management/devices/${DEVICE_ID}"
```

核心响应字段：

| 字段 | 说明 |
| --- | --- |
| `raw.sessions` | raw session 文件数量 |
| `raw.total_size_bytes` | raw 数据总字节数 |
| `processed.splits.train/val/test` | z-score 后 CSV 文件信息 |
| `processed.windows` | 各窗口尺寸下的 train/val/test 文件信息 |
| `inference.sessions` | inference 目录下 session 数 |
| `training.status` | `pending`、`in_progress`、`completed`、`failed` |
| `model.ready` | policy、checkpoint、config 是否齐全 |
| `active_auth_sessions` | 当前内存中的活跃认证会话 |

### 2.5 获取 raw session 文件列表

```http
GET /api/v1/management/devices/{device_id}/raw-sessions
```

查询参数：

| 参数 | 类型 | 默认值 | 范围 | 说明 |
| --- | --- | ---: | --- | --- |
| `limit` | int | `100` | `1..1000` | 返回最近修改的 session 文件数量 |

示例：

```bash
curl -H "X-Management-API-Key: ${API_KEY}" \
  "${BASE_URL}/api/v1/management/devices/${DEVICE_ID}/raw-sessions?limit=50"
```

响应字段：

| 字段 | 说明 |
| --- | --- |
| `device_id_hash` | 设备哈希 ID |
| `sessions[].session_id` | session ID |
| `sessions[].packet_count` | 文件内 JSONL 行数 |
| `sessions[].size_bytes` | 文件大小 |
| `sessions[].modified_at` | 文件最后修改时间 |
| `total` | 该设备 session 文件总数 |

### 2.6 获取训练状态

```http
GET /api/v1/management/devices/{device_id}/training
```

用途：外部管理程序判断该设备是否达到训练条件、训练是否完成、失败原因。

响应字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `status` | string | `pending`、`in_progress`、`completed`、`failed` |
| `total_bytes` | int | 当前 raw 数据总量 |
| `total_mb` | float | 当前 raw 数据 MB |
| `min_bytes` | int | 触发训练所需最小字节数 |
| `min_mb` | float | 触发训练所需 MB |
| `has_enough_data` | bool | raw 数据是否达到阈值 |
| `is_ready` | bool | 训练完成且数据达到阈值 |
| `last_trained_bytes` | int | 上次训练时的数据量 |
| `last_error` | string | 最近训练错误 |
| `updated_at` | string | `training_state.json` 更新时间 |
| `task` | object/null | 当前内存任务快照 |

示例：

```bash
curl -H "X-Management-API-Key: ${API_KEY}" \
  "${BASE_URL}/api/v1/management/devices/${DEVICE_ID}/training"
```

### 2.7 获取模型和策略信息

```http
GET /api/v1/management/devices/{device_id}/models
```

用途：查询模型是否可用于认证、模型大小、认证阈值、窗口大小、K 连拒/投票策略、训练评估指标。

响应字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `ready` | bool | 模型是否就绪 |
| `policy.window_size` | float | 认证窗口秒数 |
| `policy.overlap` | float | 认证窗口重叠比例 |
| `policy.target_width` | int | 模型输入时间轴宽度 |
| `policy.threshold` | float | 判定阈值 |
| `policy.interrupt_rule` | string | `k`、`vote` 或 `none` |
| `policy.k_rejects` | int | 连续拒绝 K 次触发打断 |
| `policy.vote_window_size` | int | 投票窗口 N |
| `policy.vote_min_rejects` | int | 投票窗口中至少 M 次拒绝 |
| `policy.model_version` | string | 模型版本 |
| `policy.vqgan_checkpoint` | string | VQGAN checkpoint 解析后的路径 |
| `policy.vqgan_config` | string | VQGAN config 解析后的路径 |
| `files.policy` | object | `best_lock_policy.json` 文件信息 |
| `files.vqgan_checkpoint` | object | VQGAN checkpoint 文件信息 |
| `files.vqgan_config` | object | VQGAN config 文件信息 |
| `files.lm_checkpoint` | object | 如果策略中配置语言模型 checkpoint，则返回该文件信息 |
| `training_summary` | array | 训练汇总指标，如 AUC/FAR/FRR/EER/F1 |
| `policy_search` | object | 策略搜索输出文件 |
| `error` | string/null | policy 读取错误 |

示例：

```bash
curl -H "X-Management-API-Key: ${API_KEY}" \
  "${BASE_URL}/api/v1/management/devices/${DEVICE_ID}/models"
```

示例响应片段：

```json
{
  "device_id_hash": "device123",
  "ready": true,
  "policy": {
    "user": "device123",
    "window_size": 0.2,
    "target_width": 20,
    "overlap": 0.5,
    "threshold": -0.17253071069709663,
    "interrupt_rule": "k",
    "k_rejects": 20,
    "vote_window_size": 0,
    "vote_min_rejects": 0,
    "model_version": "vqgan_user_device123_ws_0.2.pt"
  },
  "files": {
    "vqgan_checkpoint": {
      "exists": true,
      "size_bytes": 65648895,
      "size_mb": 62.608
    }
  },
  "error": null
}
```

### 2.8 获取活跃认证会话

```http
GET /api/v1/management/devices/{device_id}/auth/sessions
```

用途：查看当前服务内存中该设备正在进行的在线认证会话。服务重启后内存态会话会消失，历史结果应查询认证结果接口。

响应字段：

| 字段 | 说明 |
| --- | --- |
| `sessions[].session_id` | 认证 session ID |
| `sessions[].user_id` | 设备/用户索引 ID |
| `sessions[].created_at` | 会话创建时间 |
| `sessions[].last_activity` | 最近收到数据时间 |
| `sessions[].idle_seconds` | 空闲秒数 |
| `sessions[].window_index` | 已处理窗口计数 |
| `sessions[].policy` | 当前会话使用的策略 |
| `sessions[].tail_records` | 内存中各传感器尾部缓存数量 |
| `sessions[].consecutive_rejects` | K 连拒状态 |
| `sessions[].vote_rejects` | 投票拒绝状态 |

示例：

```bash
curl -H "X-Management-API-Key: ${API_KEY}" \
  "${BASE_URL}/api/v1/management/devices/${DEVICE_ID}/auth/sessions"
```

### 2.9 获取单设备认证结果

```http
GET /api/v1/management/devices/{device_id}/auth/results
```

查询参数：

| 参数 | 类型 | 默认值 | 范围 | 说明 |
| --- | --- | ---: | --- | --- |
| `session_id` | string | 空 | - | 指定认证 session；不传则查询该设备所有 session |
| `limit` | int | `100` | `1..1000` | 返回最近结果数量 |

用途：从 `data_storage/inference/<device_id>/<session_id>/results.jsonl` 查询历史认证结果。

示例：

```bash
curl -H "X-Management-API-Key: ${API_KEY}" \
  "${BASE_URL}/api/v1/management/devices/${DEVICE_ID}/auth/results?limit=100"

curl -H "X-Management-API-Key: ${API_KEY}" \
  "${BASE_URL}/api/v1/management/devices/${DEVICE_ID}/auth/results?session_id=auth-session-a&limit=20"
```

结果字段：

| 字段 | 说明 |
| --- | --- |
| `window_id` | 认证窗口 ID |
| `score` | 原始模型分数或决策分数 |
| `threshold` | 判定阈值 |
| `accept` | 是否接受 |
| `interrupt` | 是否触发打断 |
| `normalized_score` | 归一化分数 |
| `k_rejects` | 连拒阈值 |
| `vote_recent_windows` | 最近投票窗口数量 |
| `vote_recent_rejects` | 最近投票拒绝数量 |
| `window_size` | 认证窗口秒数 |
| `model_version` | 模型版本 |
| `server_written_timestamp` | 结果写入时间 |
| `device_id_hash` | 管理 API 补充的设备 ID |
| `session_id` | 管理 API 补充的 session ID |

### 2.10 获取全局最近认证结果

```http
GET /api/v1/management/auth/results/latest
```

查询参数：

| 参数 | 类型 | 默认值 | 范围 | 说明 |
| --- | --- | ---: | --- | --- |
| `limit` | int | `100` | `1..1000` | 返回最近结果数量 |

示例：

```bash
curl -H "X-Management-API-Key: ${API_KEY}" \
  "${BASE_URL}/api/v1/management/auth/results/latest?limit=100"
```

### 2.11 获取客户端上报指标

```http
GET /api/v1/management/client-metrics
```

用途：查询 App 通过 gRPC `ReportMetrics` 上报的聚合指标。

响应字段：

| 字段 | 说明 |
| --- | --- |
| `latest_by_device` | 每个设备最近一次上报 |
| `recent` | 最近上报列表，按接收时间倒序 |
| `received_at` | 服务端追加到每条上报记录上的接收时间 |

客户端指标内容来自 gRPC `MetricsReport`：

| 字段 | 说明 |
| --- | --- |
| `device_id_hash` | 设备哈希 ID |
| `timestamp_ms` | 客户端上报时间 |
| `reporting_period_ms` | 指标统计周期 |
| `batches_processed` | 客户端处理批次数 |
| `uploads_success` | 上传成功数 |
| `uploads_failed` | 上传失败数 |
| `sensor_samples_collected` | 采集样本数 |
| `anomalies_detected` | 客户端异常数 |
| `avg_upload_latency_ms` | 平均上传延迟 |
| `avg_cpu_usage_percent` | 平均 CPU 使用率 |
| `peak_memory_usage_mb` | 峰值内存 |
| `upload_success_rate` | 上传成功率 |

示例：

```bash
curl -H "X-Management-API-Key: ${API_KEY}" \
  "${BASE_URL}/api/v1/management/client-metrics"
```

## 3. 基础 HTTP 接口

这些接口不属于管理 API。

### 3.1 健康检查

```http
GET /health
```

响应字段：

| 字段 | 说明 |
| --- | --- |
| `status` | `healthy` 或 `unhealthy` |
| `app_name` | 服务名 |
| `version` | 服务版本 |
| `timestamp` | 服务端时间 |
| `storage_stats` | raw 数据存储统计 |

示例：

```bash
curl "${BASE_URL}/health"
```

### 3.2 根路径

```http
GET /
```

示例响应：

```json
{
  "app_name": "Continuous Authentication Server",
  "version": "1.0.0",
  "status": "running"
}
```

### 3.3 传感器数据接收接口（HTTP 兼容链路）

```http
POST /api/v1/sensor-data
Content-Type: application/octet-stream
X-Device-ID-Hash: <device_id_hash>
X-Session-ID: <session_id>
X-Packet-Sequence: <packet_seq_no>
```

用途：接收旧版 HTTP/JSON 风格加密压缩传感器包。当前主链路是 gRPC，外部管理程序通常不需要调用此接口。

请求体：

- 二进制 AES-256-GCM 加密数据；
- 服务端按“先解密、后解压”处理；
- 解压后内容为 JSON；
- 支持 LZ4/GZIP 自动识别；
- 服务端会校验请求头中的设备、session、packet 序号与 JSON 内容一致。
- 解压后的旧版 JSON 中 `device_id_hash`、`session_id`、`packet_seq_no` 会与请求头逐项校验；`device_id_hash` 和 `session_id` 可以是数字或字符串，服务端统一按字符串比较。
- 写入 raw JSONL 时会追加 `server_received_timestamp` 字段。
- 成功写入后会记录运行态计数，并按训练阈值尝试触发训练检查。
- 解压后载荷大小受 `MAX_DECOMPRESSED_SIZE` 限制，默认 `10 MiB`。

成功响应：

```json
{ "status": "ok" }
```

错误响应：

| HTTP 状态码 | `reason` | 说明 |
| ---: | --- | --- |
| `400` | `no_data` | 请求体为空 |
| `400` | `invalid_identifier` | header 中设备或 session ID 含非法路径字符 |
| `400` | `decryption_failed` | 解密失败 |
| `400` | `decompression_failed` | 解压失败 |
| `400` | `invalid_json` | 解压后不是合法 JSON |
| `400` | `validation_failed` | 字段校验失败 |
| `422` | FastAPI validation error | 缺少必填 header，或 `X-Packet-Sequence` 不是整数 |
| `413` | `request_too_large` | 请求体超过 `MAX_REQUEST_SIZE` |
| `500` | `storage_failed` | 文件写入失败 |
| `500` | `internal_error` | 未预期错误 |

## 4. gRPC 业务接口

gRPC proto 文件：

```text
protos/sensor_data.proto
```

服务名：

```text
com.continuousauth.proto.SensorDataService
```

默认地址：

```text
localhost:10500
```

健康检查：

```bash
grpcurl -plaintext localhost:10500 grpc.health.v1.Health/Check
```

### 4.1 GetInitialPolicy

```text
rpc GetInitialPolicy(PolicyRequest) returns (PolicyUpdate)
```

用途：App 获取初始采集/上传策略。

请求字段：

| 字段 | 说明 |
| --- | --- |
| `device_id_hash` | 设备哈希 ID |
| `app_version` | App 版本 |
| `android_api_level` | Android API Level |
| `current_policy_id` | 当前策略 ID |

响应字段：

| 字段 | 当前实现 |
| --- | --- |
| `policy_id` | `default` |
| `policy_version` | 服务版本 |
| `batch_interval_ms` | `1000` |
| `max_payload_size_bytes` | `min(settings.max_request_size, settings.grpc_max_message_size)` |
| `upload_rate_limit` | `50.0` |
| `compression_algorithm` | `LZ4` |
| `batch_size_threshold` | `50` |
| `sensor_sampling_rates` | ACCELEROMETER/GYROSCOPE/MAGNETOMETER 均为 100Hz |

### 4.2 StartAuthentication

```text
rpc StartAuthentication(AuthSessionRequest) returns (AuthSessionResponse)
```

用途：App 请求开始在线认证会话。

请求字段：

| 字段 | 说明 |
| --- | --- |
| `device_id_hash` | 设备哈希 ID |
| `session_id` | 认证 session ID；为空时服务端生成 |
| `app_version` | App 版本 |
| `android_api_level` | Android API Level |

成功响应：

| 字段 | 说明 |
| --- | --- |
| `accepted` | `true` |
| `session_id` | 认证 session ID |
| `message` | `ok` |
| `model_version` | 使用的模型版本 |
| `window_size_sec` | 认证窗口秒数 |
| `decision_time_sec` | 最大决策时间配置 |

拒绝响应：

| `message` | 说明 |
| --- | --- |
| `data_insufficient: <current>MB/<required>MB` | 数据不足，不能训练/认证 |
| `training_in_progress` | 数据已足够，服务端已触发或正在训练 |
| `model_not_ready: ...` | policy/checkpoint/config 缺失或不可用 |
| `invalid_identifier: ...` | 设备 ID 或 session ID 不符合路径安全规则 |

### 4.3 StreamSensorData

```text
rpc StreamSensorData(stream DataPacket) returns (stream ServerDirective)
```

用途：App 持续上传加密压缩后的传感器批次；服务端返回 Ack 和可能的认证结果。

请求 `DataPacket` 核心字段：

| 字段 | 说明 |
| --- | --- |
| `packet_id` | 包唯一 ID |
| `device_id_hash` | 设备哈希 ID |
| `base_wall_ms` | 批次创建 wall clock |
| `device_uptime_ns` | 批次创建 elapsedRealtimeNanos |
| `ntp_offset_ms` | NTP 偏移 |
| `encrypted_sensor_payload` | 加密压缩 payload |
| `encrypted_dek` | 协议字段，当前实现未使用 |
| `dek_key_id` | 协议字段，当前实现未使用 |
| `metadata.compression` | 压缩提示，如 `lz4` |
| `packet_seq_no` | 包序号 |
| `sha256` | 协议字段，当前实现保存但不校验 |

payload 解密解压后应为 `SerializedSensorBatch` protobuf：

| 字段 | 说明 |
| --- | --- |
| `samples` | 传感器样本列表 |
| `user_id_hash` | 用户哈希 ID；当前在线流程主要按 `device_id_hash` 运行 |
| `session_id` | 业务 session ID |

当前实现要求 `SerializedSensorBatch.user_id_hash` 为空或与外层 `DataPacket.device_id_hash` 一致；不一致时 raw 记录会落盘，但状态为 `validation_failed`，不会进入在线认证。

服务端响应 `ServerDirective`：

| oneof 字段 | 说明 |
| --- | --- |
| `ack` | 包接收/存储结果 |
| `auth_result` | 在线认证结果 |
| `policy` | 协议支持，当前流处理中未主动下发 |
| `key_rotation` | 协议支持，当前未实现 |
| `emergency` | 协议支持，当前未实现 |

`Ack` 字段：

| 字段 | 说明 |
| --- | --- |
| `packet_id` | 对应请求包 ID |
| `creation_server_ts` | 服务端接收时间 |
| `success` | 当前表示是否成功写入 raw session 文件 |
| `error_code` | 非法设备 ID 时为 `INVALID_IDENTIFIER`；存储失败或未捕获处理异常时为 `SERVER_ERROR` |
| `retry_after_ms` | 当前未设置 |

解密、解压、protobuf 解析或 batch 用户校验失败时，服务会保存一条带 `decryption_status` / `decryption_error` 的 raw 记录；只要 raw 文件写入成功，`Ack.success` 仍为 `true`，但不会产生认证结果。

`AuthResult` 字段：

| 字段 | 说明 |
| --- | --- |
| `device_id_hash` | 设备哈希 ID |
| `session_id` | 认证 session ID |
| `server_timestamp_ms` | 服务端结果时间 |
| `score` | 决策分数 |
| `threshold` | 决策阈值 |
| `accept` | 是否接受当前窗口/投票结果 |
| `interrupt` | 是否触发打断 |
| `window_size_sec` | 窗口秒数 |
| `window_id` | 窗口 ID |
| `normalized_score` | 归一化分数 |
| `k_rejects` | K 连拒阈值 |
| `model_version` | 模型版本 |
| `message` | 投票状态说明 |

### 4.4 SendHeartbeat

```text
rpc SendHeartbeat(Heartbeat) returns (HeartbeatAck)
```

用途：客户端心跳和时间回显。

请求字段：

| 字段 | 说明 |
| --- | --- |
| `client_timestamp` | 客户端时间 |
| `pending_packets` | 客户端待发送包数 |
| `last_packet_seq_no` | 最近包序号 |

响应字段：

| 字段 | 说明 |
| --- | --- |
| `server_timestamp` | 服务端当前时间 |
| `client_timestamp_echo` | 回显客户端时间 |

### 4.5 ReportMetrics

```text
rpc ReportMetrics(MetricsReport) returns (MetricsResponse)
```

用途：客户端上报聚合运行指标。管理 API 的 `/client-metrics` 会读取最近上报内容。

请求字段见本文 `2.11 获取客户端上报指标`。

响应：

```json
{
  "accepted": true,
  "message": "ok"
}
```

## 5. 对接建议

### 5.1 外部管理程序推荐调用顺序

1. 调用 `GET /api/v1/management/summary` 判断服务整体状态；
2. 调用 `GET /api/v1/management/devices` 展示设备列表；
3. 对单设备调用 `GET /devices/{device_id}/training` 判断训练状态；
4. 对单设备调用 `GET /devices/{device_id}/models` 获取模型大小、模型版本和认证阈值；
5. 对在线会话调用 `GET /devices/{device_id}/auth/sessions`；
6. 对历史结果调用 `GET /devices/{device_id}/auth/results` 或 `GET /auth/results/latest`；
7. 如需 App 侧上传质量，调用 `GET /client-metrics`。

### 5.2 轮询频率建议

| 接口 | 建议频率 |
| --- | --- |
| `/summary` | 5-30 秒 |
| `/runtime` | 10-60 秒 |
| `/devices` | 30-120 秒 |
| `/devices/{device_id}/training` | 10-60 秒，训练中可缩短 |
| `/devices/{device_id}/auth/results` | 1-10 秒，按业务需要 |
| `/client-metrics` | 10-60 秒 |

### 5.3 常见状态判断

模型是否可认证：

```text
GET /api/v1/management/devices/{device_id}/models
ready == true
```

训练是否完成：

```text
GET /api/v1/management/devices/{device_id}/training
status == "completed" && is_ready == true
```

数据是否不足：

```text
has_enough_data == false
```

是否发生认证风险：

```text
GET /api/v1/management/devices/{device_id}/auth/results
results[].accept == false 或 results[].interrupt == true
```

### 5.4 兼容性说明

- 设备 ID 当前以 `device_id_hash` 为主索引；
- `user_id_hash` 存在于 protobuf 中；当前在线训练/认证路径主要按设备 ID 管理，并要求 batch 中的 `user_id_hash` 为空或与设备 ID 一致；
- gRPC `Ack.success` 当前表示 raw 文件是否写入成功，不等价于“已成功训练/推理”；
- `encrypted_dek`、`dek_key_id`、`sha256` 当前会保存或透传部分信息，但主流程仍使用固定对称密钥解密，没有完整实现信封加密和哈希校验；
- 管理 API 读取运行时内存态和文件系统状态。服务重启后，活跃认证会话内存态会清空，但历史 `results.jsonl` 仍可查询。
