# Ascend 910B NPU 占用机制与释放 7 号卡方案

## 1. 结论概要

当前仓库没有把 8 张 Ascend 910B NPU 合并成一个不可分割的逻辑处理单元。代码里没有 NPU 版 HCCL/DDP/tensor parallel 之类的“8 卡组成一个模型实例”的实现；主训练脚本的 NPU 路径是“多进程 + 单进程单卡”，worker 通过 `ASCEND_RT_VISIBLE_DEVICES=<物理卡号>` 隔离后，在进程内统一使用 `npu:0`。

当前看起来“8 卡全占用”的主要原因在部署和默认配置层：

- `docker-compose.yml` 默认把 `ASCEND_VISIBLE_DEVICES`、`ASCEND_RT_VISIBLE_DEVICES` 都设成 `0,1,2,3,4,5,6,7`，并映射 `/dev/davinci0` 到 `/dev/davinci7`。
- `[training].max_parallel_train` 默认是 `8`，训练脚本在缺省 `--gpu-ids` 时会枚举全部可见 NPU 并按任务轮询。
- 在线认证推理并没有多卡均衡，`device=auto` 通常会落到首个可用 NPU，即 `npu:0`。

因此，释放 7 号卡是可行的。最低风险做法是把 ca-server 容器的可见设备集合限制为 `0,1,2,3,4,5,6`，同步把训练并行上限改为 `7`。如果要强隔离 7 号卡，还需要处理当前 `privileged: true` 带来的设备访问边界问题。

另有一个需要注意的实现细节：服务层 `TrainingManager` 有按设备池轮询选择 `npu:K` 的意图，但 `src/training/runner.py` 调用 `ca_train/hmog_vqgan_experiment.py` 时没有传 `--gpu-ids`。在“单用户 + 单窗口”默认场景下，训练脚本内部会把唯一任务分给可见列表中的第 0 张设备。这不是 8 卡合一，而是外层设备选择和内层训练脚本设备分发没有完全对齐。

## 2. 容器层如何让 NPU 可见

`docker-compose.yml` 是当前 NPU 暴露的核心入口：

- `docker-compose.yml:23-24` 默认注入 `ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 和 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。
- `docker-compose.yml:53-60` 挂载宿主机 Ascend 安装信息、driver、toolkit、`npu-smi`、`msnpureport`。
- `docker-compose.yml:90-103` 显式映射 `/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`、`/dev/davinci0` 到 `/dev/davinci7`。
- `docker-compose.yml:104` 启用了 `privileged: true`。

`deploy/entrypoint.sh` 不负责选择哪几张卡。它主要 source CANN 环境并补齐 `PYTHONPATH`、`PATH`、`LD_LIBRARY_PATH`：

- `deploy/entrypoint.sh:18-28` 尝试 source `set_env.sh`。
- `deploy/entrypoint.sh:44-72` 追加 Ascend Python 包、工具链和动态库路径。

镜像默认值和 compose 运行时默认值要分开看：

- 源码镜像 `Dockerfile:149-150` 默认也是 0-7。
- 预编译镜像 `Dockerfile.prebuilt:83-84` 默认只有 0，但通过 compose 启动时会被 `docker-compose.yml:87` 的 environment 覆盖为 0-7。

`deploy/config/server.env.example` 当前只作为 `/app/.env` 挂载，见 `docker-compose.yml:39-40`。应用的 pydantic settings 会读取它，见 `src/config.py:63-67`，但 entrypoint 不会 source 这个文件。因此，把 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6` 写进 `deploy/config/server.env` 本身不会改变进程环境；真正生效的位置是 compose 的 `environment`、compose 项目 `.env`，或启动 `docker compose` 前的宿主机环境变量。

## 3. Python 层如何选择设备

服务侧设备抽象在 `src/utils/accelerator.py`：

- `src/utils/accelerator.py:95-125` 中 `detect_backend()` 的自动选择顺序是 NPU、CUDA、CPU。
- `src/utils/accelerator.py:128-141` 中 `normalize_device()` / `resolve_torch_device()` 会把 `auto` 解析为类似 `npu:0` 的设备并设置当前设备。
- `src/utils/accelerator.py:169-186` 中 `device_pool()` 会基于可见设备数量生成 `["npu:0", ..., "npu:N"]` 的池，`max_devices` 用来限制池大小。

训练脚本侧还有一份类似实现 `ca_train/accelerator.py`。其中最关键的是 worker 级设备隔离：

- `ca_train/accelerator.py:161-169` 中 `set_visible_device("npu", K)` 会设置 `ASCEND_RT_VISIBLE_DEVICES=K`、`ASCEND_DEVICE_ID=0`，并清掉 `CUDA_VISIBLE_DEVICES`。
- 随后 worker 内部使用 `npu:0`，这个 `0` 是隔离后的进程内逻辑号，不再是原始物理卡号。

这说明 NPU 不是被合成一个 8 卡设备，而是通过环境变量把每个 worker 限制到单张卡。

## 4. 训练路径的实际行为

在线服务创建训练管理器的位置：

- `src/management/runtime.py:173-180` 使用 `settings.training_max_concurrent` 和 `settings.auth_max_concurrent` 创建训练和认证管理器。
- `src/config.py:45-49` 默认 `training_max_concurrent=8`、`auth_max_concurrent=16`。
- 但部署示例 `deploy/config/server.env.example:17-19` 把 `TRAINING_MAX_CONCURRENT=1`，所以默认 compose 运行时通常只允许一个用户训练任务同时跑。

训练管理器的设备池逻辑：

- `src/training/manager.py:102-118` 从 `[training].device` 和 `[training].max_parallel_train` 构造设备池，并用 `_device_cursor` 轮询。
- `src/training/manager.py:193-196` 先选出 `train_device`，再把它传给 `run_window_sweep_for_user()`。
- `ca_config.toml:127-132` 和 `deploy/config/ca_config.toml:127-132` 默认 `device="auto"`、`max_parallel_train=8`。

训练 runner 的子进程命令：

- `src/training/runner.py:251` 先 `normalize_device(device)`。
- `src/training/runner.py:276-308` 调用 `hmog_vqgan_experiment.py`，传入 `--device <resolved_device>`、`--max-parallel-train <配置值>`，但没有传 `--gpu-ids`。
- `src/training/runner.py:110-120` 调用 `subprocess.run()` 时没有传自定义 `env`，所以训练脚本继承 server 进程的 `ASCEND_*` 环境。
- `src/training/runner.py:370-381` 训练后会用同一个 `resolved_device` 做 policy search。

训练脚本内部还有一层并行调度：

- `ca_train/hmog_vqgan_experiment.py:657-670` 如果没有 `--gpu-ids`，会用 `device_count(backend)` 枚举全部可见 NPU，并把 `max_workers` 限制在可见设备数量内。
- `ca_train/hmog_vqgan_experiment.py:673-676` 给每个 `(user, window)` 任务轮询分配 `gpu_id`。
- `ca_train/hmog_vqgan_experiment.py:682-695` 用 `ProcessPoolExecutor` 启动 worker。
- `ca_train/hmog_vqgan_experiment.py:608-617` worker 收到 `gpu_id` 后设置可见设备，再使用 `npu:0`。
- `ca_train/hmog_vqgan_experiment.py:398-401` 只在 CUDA 且多卡时启用 `nn.DataParallel`；NPU 路径没有启用 DataParallel。

按当前默认 `[windows].sizes = [0.2]`，见 `ca_config.toml:49-52`，服务侧一次 `run_window_sweep_for_user()` 通常只给训练脚本一个 `(user, 0.2)` 任务。此时如果外层选中 `npu:3`，但命令没有 `--gpu-ids 3`，训练脚本内部仍会先构造 `[0,1,2,3,4,5,6,7]`，唯一任务会拿到 `gpu_id=0`。因此：

- 外层轮询设备池的意图存在。
- 直接多用户/多窗口运行 `hmog_vqgan_experiment.py` 时可以把任务摊到多张可见卡。
- 但当前 server 默认单用户单窗口调用方式下，训练 worker 很可能没有按外层选中的 `npu:K` 落卡。
- 训练完成后的 policy search 会使用 `resolved_device`，所以如果外层选到 `npu:7`，policy search 仍可能使用 7 号卡。

## 5. 推理和策略搜索路径

在线认证推理：

- `src/authentication/manager.py:58-80` 中 `VQGANModelCache` 默认设备是 `auto`，加载模型时调用 `resolve_torch_device()`。
- `src/authentication/manager.py:113-118` 的并发控制只是 semaphore，不是多卡调度。
- `src/authentication/manager.py:375-384` 使用模型所在设备做 `score_windows()`。

这意味着在线推理默认通常集中在 `npu:0`，不会自动把模型分摊到 8 张卡。

离线 policy search：

- `src/policy_search/runner.py:687-710` 会把传入的 `device` 标准化。
- `src/policy_search/runner.py:770-797` 打分 val/test 时使用这个 `resolved_device`。

如果训练管理器选中了 `npu:7` 且容器可见 7 号卡，policy search 有机会实际使用 7 号卡。因此释放 7 号卡时不能只看训练 worker，还要限制整个容器的可见卡集合。

另外，仓库中还保留了一些旧脚本：

- `ca_train/training_vqgan.py` 硬编码 `CUDA_VISIBLE_DEVICES='0, 1'`，默认设备是 `cuda:0`。
- `ca_train/training_vqgan_dataparallel.py` 硬编码 CUDA 0/1 DataParallel。
- `ca_train/model_inference.py` 默认设备是 `cuda`。

这些不是当前 server 主训练路径，但如果有人手工调用，需要单独清理或明确标注为 CUDA legacy，避免绕过 NPU 0-6 的运行约束。

## 6. 修改方案一：配置层释放 7 号卡

这是建议的第一步，目标是让 ca-server 只看见 0-6。

### 6.1 修改 compose 可见卡集合

把 `docker-compose.yml:23-24` 从：

```yaml
- ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
- ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
```

改成：

```yaml
- ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-0,1,2,3,4,5,6}
- ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6}
```

如果不想改 compose 文件，也可以在 compose 项目 `.env` 或启动 shell 里设置：

```bash
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6
```

注意不是写进 `deploy/config/server.env`。

### 6.2 裁剪设备节点

把 `docker-compose.yml:96-103` 中的 `/dev/davinci7:/dev/davinci7` 删除，并更新 `docker-compose.yml:91-92` 的注释。

当前仍有 `privileged: true`，见 `docker-compose.yml:104`。在 Docker 语义下，privileged 会显著放宽设备访问边界，所以“删除 devices 中的 davinci7”未必构成强隔离。若目标是严格把 7 号卡留给其他容器或宿主机任务，需要进一步实机验证去掉 `privileged: true` 后 Ascend 训练/推理是否稳定，或改用更细的 device cgroup / capability 配置。

### 6.3 修改训练并行上限

把两份配置里的 `max_parallel_train` 改为 7：

- `ca_config.toml:131-132`
- `deploy/config/ca_config.toml:131-132`

建议同步把注释从“8 卡环境”改成“0-6 共 7 张 NPU，7 号卡预留”。严格来说，如果容器只可见 7 张卡，`device_pool()` 和训练脚本都会按 `device_count()` 限制到 7；但配置改为 7 能避免文档和运行意图不一致。

不要只改 `max_parallel_train=7`。如果容器仍然可见 0-7，`hmog_vqgan_experiment.py` 的 `available_gpus` 仍会是 `[0,1,2,3,4,5,6,7]`；`max_workers` 会限制同时运行的 worker 数，但排队任务的 `job_specs` 仍可能被分配到 `gpu_id=7`。真正释放 7 号卡必须限制可见设备集合，或显式传 `--gpu-ids 0 1 2 3 4 5 6`。

### 6.4 同步文档

需要更新的说明包括：

- `README.md:143` 中“8 卡机器上”的训练调度描述。
- `README.md:174-180` 和 `README.md:238-258` 中 `/dev/davinci0 ... /dev/davinci7` 的描述。
- `ca_train/HMOG_RUN.md:49-56`、`ca_train/HMOG_RUN.md:86`、`ca_train/HMOG_RUN.md:163-173` 中 8 卡和 0-7 的描述。
- `docs/ascend_npu_containerization_vs_cuda.md` 中关于 0-7 设备节点和 8 卡透传的描述。

## 7. 修改方案二：修正训练脚本落卡语义

该方案不是释放 7 号卡的必要条件，但建议修。它解决的是“外层选中 `npu:K`，内层训练 worker 却从可见列表第 0 张开始分配”的语义偏差。

推荐在 `ca_train/hmog_vqgan_experiment.py` 的 `dispatch_training_tasks()` 中加入逻辑：

1. 如果用户显式传了 `--gpu-ids`，继续使用 `args.gpu_ids`。
2. 如果没有 `--gpu-ids`，但 `args.device` 是显式设备号，例如 `npu:3` 或 `cuda:2`，则 `available_gpus = [3]` 或 `[2]`。
3. 只有 `args.device=auto`、`npu`、`cuda` 这类未指定编号的情况，才枚举全部可见设备。

这样直接运行：

```bash
python3 ca_train/hmog_vqgan_experiment.py --device npu:6 ...
```

就会只使用物理 6 号卡，而不是因为没有 `--gpu-ids` 又回到可见列表第 0 张。

可选地，也可以在 `src/training/runner.py` 里解析 `resolved_device`，当它是 `npu:K` 或 `cuda:K` 时追加：

```bash
--gpu-ids K
```

这能让 server 调用路径更直观。更完整的做法是两处都做：训练脚本保证命令行语义正确，server runner 传参保证意图显式。

如果希望配置更清晰，可以在 `src/ca_config.py` 的 `TrainingConfig` 中增加显式字段，例如 `device_ids = [0,1,2,3,4,5,6]`，再由 `TrainingManager` 和训练 runner 传递给 `--gpu-ids`。这比单独依赖 `max_parallel_train` 更准确，因为 `max_parallel_train` 表达的是并发数，不是可用设备白名单。

建议增加测试：

- `run_window_sweep_for_user(device="npu:6")` 生成的命令包含 `--device npu:6` 和 `--gpu-ids 6`。
- `run_window_sweep_for_user(device="cpu")` 不包含 `--gpu-ids`。
- `hmog_vqgan_experiment.py --device npu:6` 在未传 `--gpu-ids` 时，`dispatch_training_tasks()` 只生成 `gpu_id=6`。
- `hmog_vqgan_experiment.py --device auto` 仍可枚举全部可见 NPU。

## 8. 验证建议

修改后需要重建或重启容器，已运行的 NPU 上下文不会因为配置文件变化自动释放。

先检查 compose 展开结果：

```bash
docker compose config | grep -E "ASCEND_(RT_)?VISIBLE_DEVICES|davinci7" -n
```

容器内检查环境变量和 torch_npu 可见数量：

```bash
docker exec ca-server bash -lc '
echo "ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES}"
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
python - <<'"'"'PY'"'"'
import torch
import torch_npu
print("npu available=", torch.npu.is_available())
print("npu count=", torch.npu.device_count())
print("current=", torch.npu.current_device())
PY
'
```

预期输出中 `ASCEND_RT_VISIBLE_DEVICES` 是 `0,1,2,3,4,5,6`，`torch.npu.device_count()` 是 `7`。

训练 smoke test 建议观察 worker 日志：

```bash
docker compose run --rm ca-server ca-train-vqgan \
  --dataset-path /app/data_storage/processed_data/window \
  --users=<device_id_hash> \
  --window-sizes 0.2 \
  --device auto \
  --gpu-ids 0 1 2 3 4 5 6 \
  --max-parallel-train 7 \
  --sweep-epochs 1 \
  --final-epochs 1
```

日志中的 `[WORKER] ... device_id=<K> device=npu:0` 表示 worker 进程已把物理卡 `<K>` 映射成进程内 `npu:0`。不应出现 `device_id=7`。

最后用宿主机侧 `npu-smi info` 观察 7 号卡。若容器仍然 `privileged: true`，即使业务代码不选择 7 号卡，也建议用实际负载验证 7 号卡没有被 torch_npu 上下文或训练进程占用。

## 9. 最终建议

如果目标是“业务上不再使用 7 号卡”，采用方案一即可：compose 可见集合改成 0-6，配置并行上限改成 7，并重启容器验证 `torch.npu.device_count()==7`。

如果目标是“强隔离，保证容器没有能力访问 7 号卡”，还需要在 Ascend 910B 机器上验证取消 `privileged: true` 后的稳定性；否则仅靠 `devices` 删除 `/dev/davinci7` 不能视作严格边界。

如果目标是“多用户训练准确按外层设备池分发”，建议同时实施方案二，修正 `--device npu:K` 和训练脚本内部 `--gpu-ids` 的语义对齐问题。
