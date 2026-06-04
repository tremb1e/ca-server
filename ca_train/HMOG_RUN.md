# HMOG / Server Window VQGAN 训练与认证说明

本文档描述当前服务端代码的训练、策略搜索和离线认证流程。旧版 HMOG 实验文档中“五个固定 HMOG 用户、12 轴输入、CUDA-only”的说明已经过期；当前主链路以服务端生成的 window CSV 为输入，VQGAN 输入统一为 9 轴。

## 数据来源

服务端预处理输出目录：

```text
<processed_data>/window/<window_size>/<user>/{train,val,test}.csv
<processed_data>/z-score/<user>/scaler.json
```

其中：

- `train.csv` 只包含真实用户数据。
- `val.csv` / `test.csv` 包含真实用户数据和 HMOG 攻击者数据。
- HMOG 填充默认按真实用户 val/test 数据量做约 1:1 平衡。
- val 使用按目录名排序后从前往后的 HMOG 用户，test 使用从后往前的 HMOG 用户，默认每侧至少 10 个用户、每个用户候选 24 个 session。

## 输入形状

当前 VQGAN 输入为：

```text
(batch, 1, 9, T)
```

9 个通道为：

```text
acc_x, acc_y, acc_z,
gyr_x, gyr_y, gyr_z,
mag_x, mag_y, mag_z
```

不再拼接 `acc_magnitude`、`gyr_magnitude`、`mag_magnitude`。因此旧的 12 轴 checkpoint、阈值和策略文件不能继续作为线上模型使用，必须重新训练并重新运行 `policy_search` 标定。

## 推荐主入口

优先使用服务端 CLI，它会读取 `ca_config.toml`：

```bash
python3 -m src.training.cli \
  --user <device_id_hash> \
  --device auto
```

关键默认值来自配置文件：

- `[training].max_epochs = 50`
- `[training].early_stop_patience = 3`
- `[training].batch_size = 128`
- `[training].device = "auto"`
- `[training].max_parallel_train = 8`
- `[training].run_policy_search = true`
- `[windows].sizes = [0.2]`
- `[auth].decision_strategy = "ema"`

训练完成后会自动运行策略搜索，并写入：

```text
<models>/<user>/best_lock_policy.json
<models>/<user>/policy_search/grid_results_vqgan_only.csv
```

在线推理、离线推理和管理接口均以 `best_lock_policy.json` 为该用户的策略来源。

## 直接运行 VQGAN 实验脚本

调试或研究窗口 sweep 时可直接运行：

```bash
python3 ca_train/hmog_vqgan_experiment.py \
  --dataset-path <processed_data>/window \
  --users=<device_id_hash> \
  --window-sizes 0.2 \
  --device auto \
  --sweep-epochs 50 \
  --final-epochs 50 \
  --early-stop-patience 3 \
  --batch-size 128 \
  --max-parallel-train 8
```

`--device auto` 会优先选择 NPU，其次 CUDA，最后 CPU。华为 Kunpeng 920 + Ascend 910B 8 卡环境下，训练管理器会把多个用户训练任务按设备池分配到 `npu:0` 到 `npu:7`，用于并行利用 8 张 NPU。若设备 ID 以 `-` 开头，直接调用该脚本时必须使用 `--users=<device_id_hash>`。

常用 smoke test 参数：

```bash
python3 ca_train/hmog_vqgan_experiment.py \
  --dataset-path /tmp/ca-server-functional/processed/window \
  --users=<device_id_hash> \
  --window-sizes 0.2 \
  --device auto \
  --sweep-epochs 1 \
  --final-epochs 1 \
  --early-stop-patience 3 \
  --batch-size 32 \
  --max-train-per-user 512 \
  --max-negative-per-split 256 \
  --max-eval-per-split 256
```

## 策略搜索

手工运行：

```bash
python3 -m src.policy_search.cli \
  --user <device_id_hash> \
  --device auto \
  --auth-method vqgan-only
```

当前线上链路执行 VQGAN-only 分数，因此默认 `policy_search_auth_method = "vqgan-only"`。搜索会在验证集上选择阈值和连续认证策略参数，并把最终线上策略写入 `models/<user>/best_lock_policy.json`。

支持的决策策略：

- `ema`：默认，对分数做指数滑动平均，降低真实用户短时波动导致的 FRR。
- `vote`：保留 y-of-x 投票机制，最近 `N` 窗中拒绝数 `>= M` 时打断。
- `k`：兼容旧的连续 K 次拒绝机制。

## 离线认证推理

VQGAN-only 离线认证推荐通过服务端认证 runner 或容器子命令运行。Token-LM 实验脚本仍可用于 transformer 分支研究：

```bash
python3 ca_train/hmog_token_auth_inference.py \
  --device auto \
  --csv-path <processed_data>/window/0.2/<device_id_hash>/test.csv \
  --window-size 0.2 \
  --target-width 20 \
  --vqgan-checkpoint <models>/<user>/checkpoints/vqgan_user_<user>_ws_0.2.pt \
  --vqgan-config <models>/<user>/checkpoints/vqgan_user_<user>_ws_0.2.config.json \
  --lm-checkpoint <token_lm_checkpoint> \
  --threshold <threshold> \
  --decision-strategy ema \
  --ema-alpha 0.2 \
  --output-csv /tmp/auth_results.csv
```

输出 CSV 会包含 `score`、`ema_score`、`decision_strategy`、`raw_accept`、`decision_accept`、`accept`、`interrupt` 等字段，便于回放阈值和策略效果。

## 日志与结果

训练脚本会输出：

- `hmog_vqgan.log`：训练过程日志。
- `hmog_metrics.txt`：人类可读指标。
- `hmog_metrics.jsonl`：逐 epoch 机器可读指标。
- `best_windows.json`：窗口 sweep 摘要。
- checkpoint 和 config：包含 `input_height = 9`。

策略搜索会输出：

- `grid_results_vqgan_only.csv`
- `best_lock_policy_vqgan_only.json`
- 根目录 `best_lock_policy.json`

线上认证会在 inference 目录中记录原始输入和结果 JSONL，结果中保留原始窗口分数、聚合后策略分数、阈值、策略名和最终判定，方便后续搜索阈值和策略。

## 容器和 NPU 注意事项

Docker Compose 会挂载 Ascend driver/toolkit、`/dev/davinci0` 到 `/dev/davinci7`、`npu-smi` 等宿主机资源。镜像内不安装驱动，只消费宿主机挂载的 Ascend 运行时。

如只做 CPU smoke test，可显式设置：

```bash
INSTALL_TORCH_NPU=0 docker build -t ca-server:latest .
```

生产环境建议保持 `INSTALL_TORCH_NPU=1`，并确认容器内 `torch_npu` 能看到 8 张 NPU。
