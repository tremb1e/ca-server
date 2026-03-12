# Continuous Authentication Server

持续身份认证服务，默认以 gRPC 为主；只有在 `HTTP_ENABLED=true` 且 `PORT != GRPC_PORT` 时才额外开启 HTTP。

## 容器化方案

本仓库已收敛为**方案 A：单镜像、单编译产物、多子命令**：
- 运行时主进程为编译后的 `ca-server` 二进制，不再使用 `python -m src.main`
- 预处理、训练、策略搜索、离线认证统一复用同一个二进制子命令
- `src/training/runner.py` 在冻结运行时会通过 `sys.executable ca-train-vqgan ...` 调起内置训练 helper，不再依赖运行时源码目录
- 运行时镜像不保留 `/app/src`、`/app/ca_train`、`/app/tests`、`/app/docs`

当前默认打包方案是 **PyInstaller onedir**。
原因：在 Ascend / `torch` / 可选 `torch_npu` 依赖存在时，PyInstaller 对动态库与 Python 扩展的兼容性更稳；`BUNDLE_TOOL=nuitka` 保留为实验构建路径，但未作为默认发布方案。

## 构建镜像

### Docker build

```bash
docker build -t ca-server:latest .
```

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
- HTTP：宿主机 `18000`（仅当 `HTTP_ENABLED=true` 时真正监听）

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
- 默认 compose 已把 `PORT` 与 `GRPC_PORT` 分开设置，是否开启 HTTP 仅取决于 `HTTP_ENABLED`

## 容器内子命令

容器入口脚本 `deploy/entrypoint.sh` 已做命令分发：

```bash
docker compose run --rm ca-server training --user <user_id> --device cpu
docker compose run --rm ca-server processing --user <user_id>
docker compose run --rm ca-server policy-search --user <user_id> --device cpu
docker compose run --rm ca-server auth --user <user_id> --csv-path /app/data_storage/processed_data/window/0.2/<user_id>/test.csv --device cpu
```

## 宿主机目录映射

Compose 保留并补全了以下目录映射：

- `deploy/config/ca_config.toml` -> `/app/ca_config.toml`
- `deploy/config/server.env` -> `/app/.env`
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

### HTTP 健康检查

仅在 `HTTP_ENABLED=true` 且 HTTP 使用独立端口时执行：

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
- 运行时镜像无源码目录：`docker exec ca-server sh -lc 'test ! -d /app/src && test ! -d /app/ca_train && test ! -d /app/tests && test ! -d /app/docs'`
- 主进程为编译二进制：`docker exec ca-server ps -o pid,comm,args -p 1`

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
- `PYTHONPATH`（仅保留 Ascend Python site-packages，不再注入 `/app` 源码路径）

当前约束是：未来运行环境需与本机一致，即宿主机继续提供上述 Ascend 驱动、toolkit、设备节点和工具文件；容器只消费挂载，不在镜像内重复安装驱动/toolkit。

当前仍保留 `privileged: true`，原因是：
- Ascend 910B2 设备访问通常依赖驱动栈、设备节点与 runtime 行为的组合
- 在只列设备节点而不启用 `privileged` 的场景下，训练/推理链路常出现不稳定或初始化失败
- 现阶段优先保证与现网 Ascend 宿主机的兼容性；若后续要继续收紧权限，应基于目标机器逐项裁剪 `devices`、`cap_add` 与驱动映射

## 已知限制

- `BUNDLE_TOOL=nuitka` 仍是实验路径，未作为默认交付物
- 若未安装 `torch_npu`，服务会回退为 CPU / CUDA / NPU 自动探测逻辑中的可用后端
- 在线训练/策略搜索对数据质量、窗口 CSV 完整性与双类别验证集有要求；空目录或极小样本只能做 smoke test，不能代表完整训练性能
- 运行时镜像不包含业务源码，但模型权重、日志、配置和挂载数据仍属于敏感资产，需结合文件权限与宿主机安全策略管理

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

默认仍以 gRPC 为主；只有在显式设置 `HTTP_ENABLED=true` 且 `PORT != GRPC_PORT` 时才会同时开启 HTTP。
