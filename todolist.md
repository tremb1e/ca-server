## Continuous Authentication Server - ToDo List (from `server/prompt.txt`)

### 1) 数据预处理 / 数据集生成
- [x] 按用户 raw session 总量阈值触发（默认 100MB），并按“文件名时间戳→否则 mtime”升序排序后取前 x 个 session 凑够目标体量（默认 100MB）。
- [x] 对 acc/gyr/mag 做 100Hz 重采样 + 线性插值对齐时间戳。
- [x] 生成 per-user `train/val/test.csv`（schema：`subject, session, timestamp, acc_x..mag_z`），并存放于 `data_storage/processed_data/<user>/`。
- [x] 在合并 HMOG 前完成列名对齐与单位转换；并通过配置限制 HMOG 合并体量，避免 val/test 与滑窗规模爆炸。
- [x] 仅用目标用户 train split 计算 Z-Score 统计量（9 mean + 9 std），保存 `scaler.json`；对 val/test（含攻击者）统一用该 scaler 归一化。

### 2) 滑动窗口数据集
- [x] 对 train/val/test 分别做滑窗（0.1–1.0s，步长=window/2，窗口不跨 session），写入 `data_storage/processed_data/window/<t>/<user>/{train,val,test}.csv`。
- [x] 滑窗/预处理 CPU 并发由 `ca_config.toml [processing].workers` 控制，并做 `min(workers, os.cpu_count())` 上限裁剪。

### 3) 训练
- [x] 最大 epoch=10，早停策略：验证集性能连续 3 次不提升则停止（由 CA-train 脚本 `--early-stop-patience=3` 提供）。
- [x] 训练阶段可通过 `subject` 区分目标用户与 HMOG 攻击者（label=1/0）。
- [x] 已跑通一次端到端训练（CPU smoke：`t=0.1`，并带训练/评估子采样参数；若需严格覆盖 `0.1–1.0s` + 更充分 epoch，请在 GPU 环境按 README 命令执行）。

### 4) 离线策略遍历 / 评估输出
- [x] 按 prompt 指定的 N∈[7,20] 与对应的 M 范围做策略 sweep（复用同一 t 下的 per-window scores，不为每个 (N,M) 重新训练）。
- [x] 输出每个 (认证方式, t, N, M) 的详细评估日志与量化指标，并写入独立文件（`policy_search/per_combo/...`）；同时保留 per-window score 缓存（`policy_search/cache/...`）。
- [x] 已在 `t=0.1` 上评估两种认证方式：`vqgan-only` 与 `vqgan+transformer`（CPU smoke）。

### 5) 推理验证 & 生产化评估
- [x] 基于最优策略对测试集做离线推理验证（debug：先跑 `--max-windows 5000`，输出 per-window 决策/得分）。
- [x] 已输出 `analysis_report.md`：汇总 FAR/FRR/EER、平均检出耗时与推理吞吐，并给出生产化缺口清单。
