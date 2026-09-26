# 2026-09-27 Ascend 兼容性与数据链路修复记录

## 运行环境与验证范围

本次检查的宿主机为 Kunpeng aarch64、8 张 Ascend 910B2。现有源码运行容器使用 Python 3.12.13、PyTorch `2.6.0+cpu`、`torch_npu 2.6.0.post5`，通过宿主机挂载使用 CANN `8.2.RC1`。容器中 `torch.npu.is_available()` 为 `True`，`torch.npu.device_count()` 为 `8`。

PyTorch 版本中的 `+cpu` 不代表模型只能运行在 CPU：NPU 后端由配套的 `torch_npu` 注册。当前 VQGAN 主链路具备 NPU 训练和推理支持，本次已在真实 `npu:5` 上验证加权重建、反向传播和在线推理使用的工作线程调用。

下述测试结论覆盖本文列出的服务与预处理修改。VQGAN 与 Token-LM 联合训练入口的进一步兼容性验证单独进行，本文不据此宣称该入口已完成端到端测试。

## 问题与修复

### 推理线程、梯度与评分一致性

- 在线认证通过 `asyncio.to_thread` 执行评分。模型加载线程选择的 NPU 不能替代工作线程的设备设置；`score_windows` 现在在实际执行线程中调用 `set_current_device(device)`。
- 评分之前未禁用 autograd，即使模型处于 `eval()` 状态也会构建梯度图。评分函数现在使用 `torch.inference_mode()`，避免认证过程中保存不需要的反向传播状态。
- 在线和离线认证现在传递策略中的 `score_metric`，避免策略采用 L1、运行时仍用默认 MSE 的尺度不一致。
- 离线认证的 `max_windows` 原先多处理一个窗口，现按实际计数限制，并拒绝非正数。
- 在线展示分数使用有界 sigmoid，避免极端重建误差使 `math.exp` 溢出；该转换不改变原始评分阈值的判断。

主要代码：`src/authentication/vqgan_inference.py`、`src/authentication/runner.py`、`src/authentication/manager.py`。

### 模型和策略分数缓存

模型缓存原先只按用户及 checkpoint 路径命中。重新训练覆盖同一路径后，新会话可能继续使用旧权重。现在缓存同时校验 checkpoint 和配置文件的文件标识、大小与时间信息；工件变化后重新加载。

策略搜索的分数缓存现在记录执行设备及 `use_amp`。CPU、NPU 或混合精度设置变化时重新评分，避免直接复用不同精度下生成的阈值依据。Transformer 策略评分也显式关闭梯度记录。

主要代码：`src/authentication/manager.py`、`src/policy_search/runner.py`。

### 中断、重复上传与传感器时钟

原预处理从最早到最晚时间戳构建连续网格，并在缺口内插值。较长的采集中断会因此变成大量合成样本，影响训练、归一化和重放结果，也会消耗大量内存。

现在每个传感器分别按连续采样区间重采样，超过 `200 ms` 的缺口不插值；加速度计和陀螺仪只保留共同覆盖的时间点。磁力计缺失保留为 NaN，交由既有的磁力计质量检测决定模型轴数。六轴模型的有效窗口不要求磁力计存在。

离线和在线分窗均识别不连续时间戳。时间倒退、重复时间点或间隔超过目标采样步长的 1.5 倍时分段；在线还按 session 变化分段，避免删除缺失数据后将前后片段拼成一个窗口。离线处理保留原始文件及片段顺序。

离线采样去重避免重复上传生成额外训练样本。在线按各传感器已接收的最大时间戳过滤旧样本，并在包内去重。仅当批次 wall clock 前进且设备 uptime 降低时，在线认证才确认新的启动域并重置历史尾部、去重状态和决策累计状态。gRPC 将这两项元数据随解析后的批次传入认证管理器。离线处理同样在明确重启时清空去重集合，使新启动域中重复使用的时间戳仍能保留；旧格式缺少元数据时仍对采样倒退安全分段。

主要代码：`src/processing/pipeline.py`、`src/authentication/manager.py`、`src/authentication/vqgan_inference.py`、`src/grpc_server.py`。

## 已完成的测试

| 测试范围 | 实际结果 |
| --- | --- |
| 最终全仓回归 `python3 -m pytest -q` | 212 项通过，23.77 秒；另通过 compileall |
| 服务层相关 CPU 回归：认证、评分、传感器权重、结果发布、策略配置与搜索 | 54 项通过 |
| `VQGAN_TEST_DEVICE=npu:5` 下的重建和推理运行时测试 | 23 项通过，25.26 秒 |
| 追加重启处理后的预处理、gRPC、推理运行时回归 | 36 项通过，5.48 秒 |

NPU 测试包含真实 NPU 上的加权重建误差、梯度、零权重输入屏蔽，以及从 `asyncio` 工作线程调用评分；缓存、参数边界等测试仍通过独立的轻量测试验证。这些记录不替代最终全仓测试结果。

相关测试文件包括 `tests/unit/test_sensor_reconstruction.py`、`tests/unit/test_inference_runtime.py`、`tests/unit/test_vqgan_inference_axes.py`、`tests/unit/test_processing_pipeline.py` 和 `tests/unit/test_grpc_server.py`。

审计环境中的原始日志为 `/data/code/backup/compat_audit_20260927/pipeline_npu5_pytest.log` 和 `pipeline_review_pytest.log`。

## 加载环境与源码镜像构建

宿主机每个新 shell 先加载 CANN 环境，再使用安装了配套 PyTorch 和 `torch_npu` 的 Python：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available(), torch.npu.device_count())'
```

源码镜像使用仓库根目录的 `Dockerfile`。例如，在保存代码提交后按提交号生成镜像标签：

```bash
cd /data/code/backup/ca-server
image_tag="ascend-$(git rev-parse --short HEAD)"
docker build --file Dockerfile \
  --build-arg PYTHON_BASE_IMAGE=python:3.12-slim-bookworm \
  --build-arg INSTALL_TORCH=1 \
  --build-arg INSTALL_TORCH_NPU=1 \
  --build-arg TORCH_VERSION=2.6.0 \
  --build-arg TORCH_NPU_VERSION=2.6.0.post5 \
  --tag "ca-server:${image_tag}" .
```

运行时沿用 `docker-compose.yml` 中的 Ascend 设备节点、驱动和 toolkit 挂载。`deploy/entrypoint.sh` 会加载 CANN 环境，源码运行方式为 `python -m src.cli`。只修改或重建源码镜像不会更新 `dist/ca-server/` 中的冻结产物；选择 `Dockerfile.prebuilt` 时必须另行重新生成并验证该产物。

## 重新训练要求

预处理的有效时间范围和样本去重规则已改变。以前基于长缺口插值生成的模型及阈值不会因镜像更新而自动修复，因此需要从原始数据重新生成处理后的 CSV、归一化参数和窗口，再重新训练并生成策略。

显式训练时使用 `--no-reuse`，避免复用旧的训练摘要和 checkpoint。确认本次模型、相邻配置 JSON、scaler、sensor mode 与策略属于同一次处理和训练结果，再开始新的认证会话。模型缓存的工件失效机制负责重新加载更新文件，但不能替代模型与策略的重新生成。

## 实际 checkpoint 跨设备数值检查

使用同一个真实训练 checkpoint、64 个来自原始包重放的窗口，分别在 CPU FP32、NPU:6 FP32 和 NPU:6 AMP 下推理。所有输出均有限；NPU AMP 相对 CPU FP32 的最大分数差为 `6.842613220214844e-05`，64 窗的原始阈值判定完全一致。此结果是所测 checkpoint 与窗口的实测数值一致性，不代表所有模型跨设备逐比特相同。

脚本与完整结果保存在 `/data/code/backup/compat_audit_20260927/compare_cpu_npu.py`、`cpu_npu_comparison.json`。

## 联合训练入口与原生 AMP

`hmog_vqgan_token_transformer_experiment.py` 原先按 CUDA 是否可用决定设备，在 Ascend 主机上会把显式 NPU 请求退到 CPU。现统一通过 accelerator 解析设备、设置随机种子，并使用 NPU 原生 autocast/GradScaler。若运行时缺少原生 NPU AMP，实现退回全精度；三项新增回归覆盖无 autocast/scaler 时仍可执行反向和优化器更新。

在正确加载 `set_env.sh` 的容器中，NPU:6 的小 VQGAN 前向、反向及实际参数更新通过；Token-LM 的 loss、反向、优化器及评分均通过，分数和损失有限。这是组件级实机验证，并非完整 Token-LM 数据集训练。日志为审计目录下 `vqgan_npu6_smoke.log`、`tokenlm_npu6_smoke.log`。

直接使用 `docker exec ... bash -lc` 可能由登录 shell 重置 PATH，导致 CANN 的 `which ccec` 失败并回退到不存在的旧路径。这不是驱动挂载缺失；使用非登录 shell 并加载 toolkit 环境后上述测试通过。
