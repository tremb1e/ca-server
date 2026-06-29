# Continuous Authentication Server

持续身份认证服务，默认以 gRPC 为主；当 `HTTP_ENABLED=true` 或 `MANAGEMENT_API_ENABLED=true`，且 `PORT != GRPC_PORT` 时额外开启 HTTP。

## 容器化方案

本仓库当前保留两条容器构建路径：
- 默认 `Dockerfile` 是源码型运行镜像，容器内通过 `python -m src.cli` 运行服务和子命令，便于在目标机器上直接构建验证。
- `Dockerfile.prebuilt` 用于预编译产物 `dist/ca-server/`，适合需要隐藏源码或交付冻结二进制的场景。
- 预处理、训练、策略搜索、离线认证统一复用 `deploy/entrypoint.sh` 的多子命令入口。
- `src/training/runner.py` 在冻结运行时会通过 `sys.executable ca-train-vqgan ...` 调起内置训练 helper；源码型镜像则直接使用 `/app/src` 和 `/app/ca_train`。

当前默认打包方案是 **PyInstaller onedir**。
原因：在 Ascend / `torch` / 可选 `torch_npu` 依赖存在时，PyInstaller 对动态库与 Python 扩展的兼容性更稳；`BUNDLE_TOOL=nuitka` 保留为实验构建路径，但未作为默认发布方案。

## 构建镜像

### Docker build

```bash
docker build -t ca-server:latest .
```

若构建环境需要通过宿主机 `9999` 端口代理访问外网：

```bash
docker build \
  --build-arg HTTP_PROXY=http://host.docker.internal:9999 \
  --build-arg HTTPS_PROXY=http://host.docker.internal:9999 \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  -t ca-server-api:latest .
```

在 Linux 上若 `host.docker.internal` 不可用，可改用宿主机网关 IP 或 Docker build 的 `--network=host` 配合 `http://127.0.0.1:9999`。

### 脚本构建

```bash
./scripts/build_docker.sh ca-server latest
```

脚本现在会按顺序执行：
- 若存在旧的 `ca-server:latest`（或你传入的 tag），先删除旧镜像
- 重新执行 `docker build`
- 自动导出镜像到 `dist/docker-images/<image>_<tag>.tar`

可选构建参数：
- `BUNDLE_TOOL=pyinstaller`：默认、已验证
- `BUNDLE_TOOL=nuitka`：实验路径，若与 `torch` / `torch_npu` / Ascend 动态库不兼容可能失败
- `INSTALL_TORCH_NPU=1`：在可用私有源或本地 wheel 时安装 `torch_npu`
- `TORCH_FIND_LINKS` / `TORCH_EXTRA_INDEX_URL`：指定 `torch` / `torch_npu` wheel 来源
- `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`：传递 Docker 构建阶段代理
- `DOCKER_BUILD_NETWORK=host`：Linux 本机代理如 `http://127.0.0.1:9999` 场景可用
- `DOCKER_BUILD_EXTRA_ARGS='--progress=plain'`：向 `docker build` 追加额外参数

说明：
- `docker compose` 当前默认会以 `INSTALL_TORCH_NPU=1` 构建镜像，避免训练镜像遗漏 `torch_npu`
- 运行时镜像不会安装 Ascend 驱动和 toolkit，而是直接挂载宿主机上的 `/usr/local/Ascend/*`
- 若你明确只需要 CPU/CUDA 版本，可在构建时显式设置 `INSTALL_TORCH_NPU=0`

## 启动服务

### Docker Compose

```bash
docker compose up -d --build
```

默认端口：
- gRPC：宿主机 `10500`
- HTTP：宿主机 `18000`（仅当 `HTTP_ENABLED=true` 或 `MANAGEMENT_API_ENABLED=true` 时真正监听）

当前 compose 采用 `network_mode: host`。
原因：在线上“远端 OpenResty -> 域名/VIP -> 本机”链路下，需要让容器监听方式与 `python -m src.main` 直跑一致，避免 Docker bridge/端口映射在特定 VIP 路径上表现不一致。

### 启用独立 HTTP 端口

```bash
HTTP_ENABLED=true \
CA_SERVER_HTTP_PORT=18000 \
docker compose up -d --build
```

注意：
- 若 `HTTP_ENABLED=true` 但 `PORT == GRPC_PORT`，程序会自动关闭 HTTP，仅保留 gRPC
- 默认 compose 已把 `PORT` 与 `GRPC_PORT` 分开设置，是否开启 HTTP 取决于 `HTTP_ENABLED` 或 `MANAGEMENT_API_ENABLED`

### 启用管理 API

管理 API 是只读接口，用于外部管理程序查询运行态、训练状态、模型策略、认证结果和客户端指标。默认关闭，启用时必须设置 API key：

```bash
cp deploy/config/server.env.example deploy/config/server.env
# 编辑 deploy/config/server.env：MANAGEMENT_API_ENABLED=true，并设置 MANAGEMENT_API_KEY
CA_SERVER_ENV_FILE=server.env \
CA_SERVER_HTTP_PORT=18000 \
docker compose up -d --build
```

也可以不创建 `server.env`，直接从 shell 传入 `MANAGEMENT_API_ENABLED` / `MANAGEMENT_API_KEY`；未设置时 compose 会挂载只含默认值的 `deploy/config/server.env.example`。

示例：

```bash
curl -H "X-Management-API-Key: replace-with-a-long-random-token" \
  http://localhost:18000/api/v1/management/summary

curl -H "X-Management-API-Key: replace-with-a-long-random-token" \
  http://localhost:18000/api/v1/management/devices/<device_id_hash>/models

curl -H "X-Management-API-Key: replace-with-a-long-random-token" \
  "http://localhost:18000/api/v1/management/devices/<device_id_hash>/auth/results?limit=100"
```

设置 `MANAGEMENT_API_ENABLED=true` 时，服务会自动启用 HTTP；若 HTTP 端口与 gRPC 端口相同，启动会失败并提示拆分端口。OpenAPI 由 FastAPI 自动生成，只要 HTTP 服务开启即可访问 `/docs` 或 `/openapi.json`；未启用管理 API 时 schema 不包含管理路由。管理 API 不返回原始传感器样本、密文 payload 或 scaler 明细。

## 容器内子命令

容器入口脚本 `deploy/entrypoint.sh` 已做命令分发：

```bash
docker compose run --rm ca-server training --user <device_id_hash> --device cpu
docker compose run --rm ca-server processing --user <device_id_hash>
docker compose run --rm ca-server policy-search --user <device_id_hash> --device cpu
docker compose run --rm ca-server auth --user <device_id_hash> --csv-path /app/data_storage/processed_data/window/0.2/<device_id_hash>/test.csv --device cpu
```

## 当前认证链路关键参数

主要阈值集中在 `ca_config.toml` 和 `deploy/config/ca_config.toml`：

- 数据触发阈值默认 `300MB`。若记为 `100x`，预处理会按 `train=75x`、`val=12.5x`、`test=12.5x` 动态切分。
- HMOG 只填充 `val/test`，默认让真实用户与 HMOG 攻击者数据量约为 `1:1`；val 使用排序靠前 HMOG 用户，test 使用排序靠后 HMOG 用户，每侧至少 10 个用户。
- VQGAN 输入已改为 `(batch, 1, 6, T)`：6 行 = 加速度计 + 陀螺仪，移除磁力计；模型 config 的 `input_height` 由 9 改为 6。旧 9 轴 / 12 轴模型会被就绪检查拒绝，必须重新生成窗口数据并重训。
- 训练默认最多 50 epoch，验证集性能连续 3 次不提升早停；训练完成后默认运行 `policy_search`。
- 训练阶段先写兜底策略 `training_fallback_policy.json`；`policy_search` 成功后“原子写入”正式 `best_lock_policy.json`（标记 `policy_status="ready"`、`policy_search_completed=true`，并落盘 `genuine_score_stats`），线上推理直接消费该文件。
- 认证启动默认只接受正式策略（`policy_status="ready"` 且 `policy_search_completed=true`）。新增配置开关 `auth.allow_training_fallback_policy`（默认 `false`），置 `true` 时缺少正式策略才允许回退到 `training_fallback_policy.json` 降级运行。
- 认证决策的聚合策略 / 投票窗口以 `ca_config.toml [auth]` 为运行时权威来源，覆盖 per-user `best_lock_policy.json` 中的同名字段（per-user 策略仅提供该用户“单窗口原始分数阈值 `threshold`”与模型工件路径）。当前部署为 `decision_strategy="vote"`、`vote_window_size=50`、`vote_min_rejects=30`，与 `result_delay_sec=5.0` 折算的 50 窗对齐，故 App 端 `AuthResult.message` 展示 “30 of 50”；`ema`、`k` 仍可通过配置选择。仍是单一 primary 阶段，不引入二次迟滞/二次投票。
- 新增 `auth.result_delay_sec` 认证结果发布延迟（默认 `0`，部署配置示例 `5.0`）：服务端按该延迟折算窗口数（`stride=window×(1-overlap)`，默认 0.2s 窗 + 0.5 overlap => 每秒约 10 窗，故 5s≈50 窗）累积后，每个周期才向 App 发布一次按 `[auth]` 配置聚合策略得到的 `AuthResult`，期间数据包只回 `Ack`；全量窗口仍逐窗落盘到 `inference` 结果。`StartAuthentication.decision_time_sec` 回报 `max(max_decision_time_sec, result_delay_sec)`。`0` 表示不延迟（每个数据包都回结果，兼容旧行为）。vote 聚合窗口数应与发布窗口数一致（均为 50）。
- 模型就绪检查 `check_trained_model` 同时要求 policy、checkpoint、config 和 `processed_data/z-score/<user>/scaler.json` 完整可读，并校验阈值有限、策略含必要字段、`input_height == 6`，校验详情写入 `inference/<device>/<session>/model_validation.json`。
- 认证结果按尺度拆分写出 `raw_*` / `ema_*` / `display_*`，并带 `result_stage="primary"`；`score` / `threshold` / `normalized_score` 保留为兼容字段。
- 在 Ascend 910B 8 卡机器上，`device=auto` 会优先使用 NPU，训练管理器会把并行用户任务分配到可见 NPU 设备池。
- 并发能力（按“支持 100 并发用户”配置）：服务端为 asyncio + `grpc.aio`。训练对用户“无限接纳 + 队列”，真正同时在 NPU 上执行的训练数由 `TRAINING_MAX_CONCURRENT`（默认 `8`，每卡约 1 个）限制，其余在 `asyncio.Semaphore` 上排队——既支持 100 并发用户又不 OOM；可在 `server.env` 调大（前提是单卡显存能容纳 `ceil(值/卡数)` 个训练）。推理由 `AUTH_MAX_CONCURRENT`（默认 `100`，NPU 前向信号量）+ per-user 模型 LRU 缓存 `AUTH_MAX_CACHED_MODELS`（默认 `100`）+ `GRPC_MAX_CONCURRENT_RPCS`（默认 `512`，每条 `StreamSensorData` 长流计 1 个 RPC）共同约束。注意：`ca_config.toml [training] max_parallel_train` 只用于离线 / CLI 多用户 sweep，不是 server 的并发开关；推理模型缓存当前都在 `npu:0`，与 round-robin 占用 `npu:0..7` 的训练在 `npu:0` 上共享，属可接受的已知现象。

## 数据字段说明

- gRPC `SerializedSensorBatch.samples[]` 的前台应用字段为 `foreground_app_name`，字段号为 8，内容是 Android 当前前台应用的明文包名。
- HTTP JSON 兼容链路使用同名 `foreground_app_name` 可选字段。
- 旧字段 `foreground_app_hash` / `foreground_package_name` 已不作为当前 app/server 数据契约使用。

## 宿主机目录映射

Compose 保留并补全了以下目录映射：

- `deploy/config/ca_config.toml` -> `/app/ca_config.toml`
- `deploy/config/${CA_SERVER_ENV_FILE:-server.env.example}` -> `/app/.env`
- `deploy/data/raw_data` -> `/app/data_storage/raw_data`
- `deploy/data/processed_data` -> `/app/data_storage/processed_data`
- `deploy/data/inference` -> `/app/data_storage/inference`
- `deploy/data/models` -> `/app/data_storage/models`
- `deploy/data/hmog_preprocessed` -> `/app/data_storage/hmog_preprocessed`
- `deploy/data/ca_train_cached_windows` -> `/app/runtime/ca_train/cached_windows`
- `deploy/data/ca_train_token_caches` -> `/app/runtime/ca_train/token_caches`
- `deploy/data/results` -> `/app/results`
- `deploy/data/ascend/kernel_meta` -> `/app/kernel_meta`
- `deploy/data/ascend/log` -> `/app/ascend_logs`
- `deploy/logs` -> `/app/logs`
- `deploy/certs` -> `/app/certs`

其中：
- `HMOG_DATA_PATH` 默认改为容器内可配置目录 `/app/data_storage/hmog_preprocessed`
- 训练缓存目录已从原先 `/app/ca_train/*` 收敛到 `/app/runtime/ca_train/*`，避免运行时镜像出现源码目录假象

此外，compose 现在会显式映射 Ascend 设备节点：
- `/dev/davinci_manager`
- `/dev/devmm_svm`
- `/dev/hisi_hdc`
- `/dev/davinci0` ... `/dev/davinci7`

这一步是为了解决容器内 `torch_npu` / `torch.npu.is_available()` 看不到宿主机 NPU 的问题；若目标机器的 NPU 数量不是 8 张，请按实际设备节点增删对应条目。

## 健康检查与验收

### Compose 健康检查

当前 compose 健康检查已切换为 `grpc_health_probe`：

```bash
docker inspect --format '{{json .State.Health}}' ca-server | jq
```

### gRPC 健康检查

```bash
grpcurl -plaintext localhost:10500 grpc.health.v1.Health/Check
```

或：

```bash
docker exec ca-server grpc_health_probe -addr=127.0.0.1:10500
```

### OpenResty 443 验收

如果通过 OpenResty 暴露 `https://<domain>:443` 给 Android app，先确认外部证书与 app 配置域名匹配，并且 ALPN 能协商 `h2`：

```bash
openssl s_client -connect <domain>:443 \
  -servername <domain> \
  -verify_hostname <domain> \
  -alpn h2 -brief </dev/null
```

通过后再用 TLS gRPC 客户端检查 443；不要用 `-insecure` 作为正式验收条件。后端默认 `10500` 是 h2c，OpenResty 回源应使用 `grpc://127.0.0.1:10500`，详见 `docs/openresty_grpc_proxy.md`。Android app 的公网入口应配置为 `https://ca.macrz.com:443`；80 只返回 301，gRPC 客户端不依赖跳转，当前 app 会把 `https` + 80 归一化为 443。

### HTTP 健康检查

仅在 HTTP 服务开启且使用独立端口时执行：

```bash
curl --http2-prior-knowledge http://localhost:18000/health
```

### 验收建议

- 镜像构建成功：`docker build -t ca-server:latest .`
- 容器启动成功：`docker compose up -d`
- gRPC health 成功：`docker exec ca-server grpc_health_probe -addr=127.0.0.1:10500`
- HTTP 模式验收：`curl http://localhost:18000/health`
- 日志落盘：检查 `deploy/logs`
- 业务数据落盘：检查 `deploy/data/raw_data`、`deploy/data/inference`、`deploy/data/models`
- 管理 API 验收：`curl -H "X-Management-API-Key: $MANAGEMENT_API_KEY" http://localhost:18000/api/v1/management/summary`
- 预编译镜像验收：使用 `Dockerfile.prebuilt` 时再检查 `/app/src`、`/app/ca_train` 不存在，以及主进程为冻结二进制

## Ascend 依赖说明

Compose 保留了 Ascend 910B2 所需环境变量与只读挂载：
- `/etc/ascend_install.info:/etc/ascend_install.info:ro`
- `/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro`
- `/usr/local/Ascend/ascend-toolkit:/usr/local/Ascend/ascend-toolkit:ro`
- `/usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro`
- `/usr/bin/msnpureport:/usr/bin/msnpureport:ro`
- `/dev/davinci_manager:/dev/davinci_manager`
- `/dev/devmm_svm:/dev/devmm_svm`
- `/dev/hisi_hdc:/dev/hisi_hdc`
- `/dev/davinci0` ... `/dev/davinci7`
- `ASCEND_VISIBLE_DEVICES`
- `ASCEND_RT_VISIBLE_DEVICES`
- `ASCEND_DRIVER_HOME`
- `ASCEND_INSTALL_INFO`
- `ASCEND_TOOLKIT_ROOT`
- `ASCEND_TOOLKIT_HOME`
- `ASCEND_OPP_PATH`
- `LD_LIBRARY_PATH`
- `PYTHONPATH`（源码镜像包含 `/app`、`/app/ca_train` 和 Ascend Python site-packages；预编译镜像只保留 Ascend Python site-packages）

当前约束是：未来运行环境需与本机一致，即宿主机继续提供上述 Ascend 驱动、toolkit、设备节点和工具文件；容器只消费挂载，不在镜像内重复安装驱动/toolkit。

当前仍保留 `privileged: true`，原因是：
- Ascend 910B2 设备访问通常依赖驱动栈、设备节点与 runtime 行为的组合
- 在只列设备节点而不启用 `privileged` 的场景下，训练/推理链路常出现不稳定或初始化失败
- 现阶段优先保证与现网 Ascend 宿主机的兼容性；若后续要继续收紧权限，应基于目标机器逐项裁剪 `devices`、`cap_add` 与驱动映射

## 已知限制

- `BUNDLE_TOOL=nuitka` 仍是实验路径，未作为默认交付物
- 若未安装 `torch_npu`，服务会回退为 CPU / CUDA / NPU 自动探测逻辑中的可用后端
- 在线训练/策略搜索对数据质量、窗口 CSV 完整性与双类别验证集有要求；空目录或极小样本只能做 smoke test，不能代表完整训练性能
- 使用 `Dockerfile.prebuilt` 构建的预编译镜像不包含业务源码；模型权重、日志、配置和挂载数据仍属于敏感资产，需结合文件权限与宿主机安全策略管理

## 源码保护边界

将 Python 程序编译为二进制只能**提高源码获取门槛**，并不能提供绝对防逆向能力：
- 二进制、符号、字符串、模型结构、协议与运行时行为仍可能被分析
- 若要进一步提高保护强度，仍需结合最小暴露面、镜像分层控制、访问控制、权重加密、远程密钥管理与宿主机安全加固

## 本地源码运行

如需直接从源码运行：

```bash
pip install -r requirements.txt -r requirements-ml.txt
python -m src.main
```

默认仍以 gRPC 为主；显式设置 `HTTP_ENABLED=true` 或 `MANAGEMENT_API_ENABLED=true`，且 `PORT != GRPC_PORT` 时会同时开启 HTTP。
