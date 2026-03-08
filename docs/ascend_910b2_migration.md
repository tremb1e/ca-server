# CA-Server 从 NVIDIA CUDA 迁移到 Ascend 910B2（torch_npu）技术方案

## 1. 目标与范围

本方案目标：让 `ca-server` 在华为 Ascend 910B2 上跑通完整主链路：

1. 数据处理（`src.processing`）
2. VQGAN 训练（`src.training` -> `ca_train/hmog_vqgan_experiment.py`）
3. 离线策略搜索（`src.policy_search`）
4. 离线认证推理（`src.authentication`）
5. 在线会话认证（`AuthSessionManager`）

本次迁移优先保障：**同一份代码同时兼容 `npu/cuda/cpu`**，避免硬编码单一后端。

## 2. 迁移依据（来自已有资料）

依据你提供的资料目录 `/data/huawei_docs`，关键迁移点包括：

1. `html/PT_LMTMOG_0070.html`：明确 CUDA 与 NPU 设备接口替换关系（`torch.cuda.*` -> `torch_npu.npu.*` / `torch.npu.*`）。
2. `html/pttools_qucikstart_0001.html`：迁移脚本需要引入 Ascend 运行时能力，推荐用 NPU 设备字符串运行。
3. `pytorch/README.zh.md`：Ascend PyTorch 插件 `torch_npu` 的安装与版本匹配规则（`torch` 与 `torch-npu` 需严格配套）。
4. `html/PT_LMTMOG_0078.html`：单进程多卡 NPU 使用示例（`torch.npu.set_device`、`npu:x`）。

## 3. 迁移设计

### 3.1 设备抽象统一

新增统一设备适配层：

- `src/utils/accelerator.py`
- `ca_train/accelerator.py`

核心策略：

1. 默认设备从 `cuda:0` 改为 `auto`。
2. 自动选择优先级：`npu > cuda > cpu`。
3. 若用户显式传入不可用设备（如 `cuda:0` 但无 CUDA），自动降级到可用后端并告警。
4. 推理/训练里的 AMP 使用统一上下文，NPU 优先 `torch.npu.amp.autocast`。

### 3.2 多卡并行调度改造（训练脚本）

文件：`ca_train/hmog_vqgan_experiment.py`

原问题：

1. 多卡枚举只看 `torch.cuda.device_count()`。
2. 子进程绑定设备只写 `CUDA_VISIBLE_DEVICES`。
3. worker 内设备解析依赖 `torch.cuda.is_available()`，在 Ascend 上会错误退回 CPU。

迁移后：

1. 根据 `backend`（npu/cuda/cpu）动态枚举设备。
2. NPU 使用 `ASCEND_RT_VISIBLE_DEVICES` 做子进程设备隔离。
3. worker 里统一用 `resolve_torch_device()` 获得实际设备。
4. DataParallel 仅在 CUDA 启用（NPU 路径默认单卡/多进程并行）。

### 3.3 推理打分路径改造

文件：

- `src/policy_search/runner.py`
- `src/authentication/runner.py`
- `src/authentication/vqgan_inference.py`
- `ca_train/hmog_token_auth_inference.py`
- `ca_train/hmog_tokenizer.py`

改造点：

1. 去除 `torch.cuda.is_available()` 硬分支。
2. 使用统一设备解析后再构建 `torch.device`。
3. AMP 统一通过 NPU/CUDA 兼容上下文执行。

### 3.4 CLI 与文档层

默认参数改为 `--device auto`，并在帮助信息中明确支持：`auto / npu:0 / cuda:0 / cpu`。

## 4. 已实施代码变更（本次提交）

1. 新增 `src/utils/accelerator.py`。
2. 新增 `ca_train/accelerator.py`。
3. 修改 `src/training/cli.py`：设备默认值改为 `auto`。
4. 修改 `src/training/runner.py`：训练前标准化设备。
5. 修改 `src/policy_search/cli.py`：设备默认值改为 `auto`。
6. 修改 `src/policy_search/runner.py`：统一设备解析、NPU/CUDA 兼容 autocast；并修复 `vqgan-only` 模式下对 LM checkpoint 的强依赖。
7. 修改 `src/authentication/cli.py`：设备默认值改为 `auto`。
8. 修改 `src/authentication/runner.py`：统一设备解析。
9. 修改 `src/authentication/manager.py`：模型缓存默认设备改为 `auto`。
10. 修改 `src/authentication/vqgan_inference.py`：统一 autocast。
11. 修改 `ca_train/hmog_vqgan_experiment.py`：训练调度与 worker 全链路 NPU 兼容改造。
12. 修改 `ca_train/hmog_token_auth_inference.py`：设备解析与 autocast 兼容。
13. 修改 `ca_train/hmog_tokenizer.py`：autocast 兼容。
14. 修改 `README.md`：示例命令设备改为 `auto`。

## 5. Ascend 环境依赖与本机落地约束

### 5.1 必须先加载 CANN 环境

每次新 shell 进入项目前，先执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 5.2 `PYTHONPATH` 必须“追加”，不能覆盖

```bash
export PYTHONPATH=/data/code/ca-server:${PYTHONPATH}
```

若直接覆盖 `PYTHONPATH`，可能导致 CANN 的 Python 组件（如 `tbe/te`）不可见，触发 `ModuleNotFoundError`。

### 5.3 无法使用 `venv` 时的安装方式（本机已验证）

本机无 `sudo` 且不保证 `python3-venv` 可用，使用用户目录安装：

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user torch==2.6.0 torch-npu==2.6.0.post5
python3 -m pip install --user -r requirements.txt -r requirements-ml.txt
python3 -m pip install --user decorator psutil cloudpickle ml-dtypes tornado absl-py
```

说明：

1. `torch` 与 `torch-npu` 必须同大版本严格匹配（如 `2.6.0` / `2.6.0.post5`）。
2. `requirements.txt` 中已将 `numpy` 约束为 `<2.0`，避免当前 CANN Python 运行时与 NumPy 2.x 的兼容问题。

### 5.4 HMOG 攻击者数据路径

数据处理阶段默认会读取 `HMOG_DATA_PATH`（默认值为 `/data/code/ca/refer/ContinAuth/src/data/processed/raw_hmog_data`）来合并攻击者样本。

若该目录缺失：

1. 处理链路会退化为仅正样本。
2. 训练阶段会触发数据完整性检查报错（`val/test must contain both classes`）。

本机验证时使用：

```bash
export HMOG_DATA_PATH=/data/code/ca-server/tmp_hmog
```

## 6. 详细测试流程（Ascend 910B2，已实机验证）

### 6.1 硬件与运行时检查

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/data/code/ca-server:${PYTHONPATH}
npu-smi info
python3 - <<'PY'
import torch
import torch_npu
print("torch=", torch.__version__)
print("torch_npu imported ok")
print("npu available=", torch.npu.is_available())
print("npu count=", torch.npu.device_count())
x = torch.randn(2, 2, device='npu:0')
y = torch.randn(2, 2, device='npu:0')
print("matmul shape=", (x @ y).shape)
PY
```

通过标准：`npu available=True` 且矩阵乘法成功。

### 6.2 代码健康检查

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/data/code/ca-server:${PYTHONPATH}
cd /data/code/ca-server
python3 -m compileall src ca_train
pytest -q
```

通过标准：`compileall` 无语法错误，`pytest` 全部通过。

### 6.3 数据处理链路（最小样本）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/data/code/ca-server:${PYTHONPATH}
export HMOG_DATA_PATH=/data/code/ca-server/tmp_hmog
cd /data/code/ca-server

# 如果 raw 数据总量不足 100MB，可用以下方式快速扩展 smoke 样本
src=data_storage/raw_data/u_demo_npu/session_1700000000000.jsonl
dst=data_storage/raw_data/u_demo_npu/session_1700000001000.jsonl
: > "$dst"
for i in $(seq 1 210); do cat "$src" >> "$dst"; done

python3 -m src.processing.cli --user u_demo_npu
```

检查输出：

1. `data_storage/processed_data/z-score/u_demo_npu/scaler.json`
2. `data_storage/processed_data/window/0.2/u_demo_npu/train.csv`
3. `data_storage/processed_data/window/0.2/u_demo_npu/val.csv`
4. `data_storage/processed_data/window/0.2/u_demo_npu/test.csv`

说明：当前 `ca_config.toml` 默认要求 raw 数据达到约 100MB 才会触发处理。若样本不足，该命令会提示 `Skip user ... total < threshold`。

### 6.4 训练链路（NPU smoke）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/data/code/ca-server:${PYTHONPATH}
cd /data/code/ca-server
python3 -m src.training.cli \
  --user u_demo_npu \
  --device npu:0 \
  --window-sizes 0.2 \
  --vqgan-epochs 1 \
  --batch-size 32 \
  --max-train-per-user 512 \
  --max-negative-per-split 512 \
  --max-eval-per-split 512 \
  --no-reuse
```

检查输出：

1. `data_storage/models/u_demo_npu/checkpoints/vqgan_user_u_demo_npu_ws_0.2.pt`
2. `data_storage/models/u_demo_npu/logs/ws_0.2/best_windows.json`
3. `data_storage/models/u_demo_npu/best_lock_policy.json`

### 6.5 策略搜索（vqgan-only）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/data/code/ca-server:${PYTHONPATH}
cd /data/code/ca-server
python3 -m src.policy_search.cli \
  --user u_demo_npu \
  --device npu:0 \
  --auth-method vqgan-only \
  --window-sizes 0.2
```

检查输出：

1. `data_storage/models/u_demo_npu/policy_search/grid_results_vqgan_only.csv`
2. `data_storage/models/u_demo_npu/policy_search/pareto_frontier_vqgan_only.csv`
3. `data_storage/models/u_demo_npu/policy_search/per_combo/vqgan_only/t_0.2/N_7_M_4.csv`（任取存在一个即可）

### 6.6 离线认证推理

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/data/code/ca-server:${PYTHONPATH}
cd /data/code/ca-server
python3 -m src.authentication.cli \
  --user u_demo_npu \
  --csv-path data_storage/processed_data/window/0.2/u_demo_npu/test.csv \
  --device npu:0 \
  --max-windows 256
```

检查输出：

1. `data_storage/models/u_demo_npu/inference/infer_ws_0.2.csv`

### 6.7 在线服务回归（健康检查）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/data/code/ca-server:${PYTHONPATH}
cd /data/code/ca-server
python3 -m src.main
```

另开终端验证：

```bash
python3 - <<'PY'
import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
ch = grpc.insecure_channel("127.0.0.1:10500")
stub = health_pb2_grpc.HealthStub(ch)
resp = stub.Check(health_pb2.HealthCheckRequest(service=""), timeout=8)
print(health_pb2.HealthCheckResponse.ServingStatus.Name(resp.status))
PY
```

通过标准：输出 `SERVING`。

## 7. 本机实测结果（本轮迁移）

1. `python3 -m compileall src ca_train` 通过。
2. `pytest -q` 通过（51 passed）。
3. 处理、训练、策略搜索、认证推理四条离线链路在 `u_demo_npu` 样本上跑通。
4. `python3 -m src.main` 可启动并通过 gRPC health（`127.0.0.1:10500`）检查。
5. 若不设置 `HMOG_DATA_PATH`，训练会因 `val/test` 缺少负样本失败；设置后恢复正常。

## 8. 回滚与兼容说明

1. 设备参数均支持显式 `--device cpu`，可作为紧急回滚路径。
2. CUDA 环境仍可用（`--device cuda:0`），本次改造没有移除 CUDA 能力。
3. 未安装 `torch_npu` 的环境，`auto` 会自动回退到 CUDA/CPU。

## 9. 已知现象与后续优化

1. Ascend 工具链可能输出大量 `SyntaxWarning` / `torch_npu` 权限提示日志，通常不影响功能，可视为噪声。
2. 若需 NPU 多机多卡训练，可将当前路径升级为 `torch.distributed` + `hccl`。
3. 建议新增 `scripts/check_accelerator_env.py` 做启动前环境自检。
