HMOG VQGAN 训练文档

环境准备
- 激活环境：`conda activate tremb1e`（已包含 CUDA）。
- 项目目录：`/data/code/CA-Model`
- 数据目录：`/data/code/ca/refer/ContinAuth/src/data/processed/Z-score_hmog_data_with_magnitude`，包含用户 100669、151985、171538、180679、186676 的 train/val/test CSV。

核心脚本
- `hmog_vqgan_experiment.py`：加载五个用户数据，按 13 个窗口长度（0.1–2.0 秒，50% 重叠）扫参，记录验证 AUC 最佳窗口并可选复训。
- 主要默认参数（可用命令行覆盖）：
  - `--batch-size 128`
  - `--num-workers 22`、`--cpu-threads 44`（充分使用 22 核 44 线程）
  - `--learning-rate 2.5e-4`、`--latent-dim 256`、`--num-codebook-vectors 512`
  - `--use-amp`：默认开启混合精度；多卡自动 DataParallel（除非 `--no-data-parallel`）。
  - 日志 & Checkpoint：`--log-dir results/experiment_logs`，`--output-dir results`

运行示例
```bash
conda activate tremb1e
cd /data/code/CA-Model
# 全部 5 个用户，13 个窗口，每个窗口 1 epoch 扫参，最佳窗口再训 5 轮
python hmog_vqgan_experiment.py --use-amp --sweep-epochs 1 --final-epochs 5
# 只跑某些用户与窗口
python hmog_vqgan_experiment.py --users 100669 151985 --window-sizes 0.5 1.0 2.0 --sweep-epochs 2 --final-epochs 6
```

日志与结果
- 人类可读日志：`results/experiment_logs/hmog_metrics.txt`（包含 stage/user/window/epoch、AUC、FAR、FRR、EER、F1、阈值、推理延迟，全部小数制）。
- 机器可读：`results/experiment_logs/hmog_metrics.jsonl`。
- 训练日志：`results/experiment_logs/hmog_vqgan.log`（Python logging 输出）。
- 最优窗口摘要：`results/experiment_logs/best_windows.json`。
- 最优权重：`results/checkpoints/vqgan_user_<uid>_ws_<window>.pt`。

调参提示
- FAR/FRR/EER 计算在 `compute_metrics` 中通过 ROC 求阈值；若需降低 FAR，可在该函数对 `eer_threshold` 做平移或改用固定阈值策略。
- 推理延迟 `latency` 为单样本平均耗时（秒），从 `evaluate_model` 里获得。

数据与形状
- 输入窗口形状：`(batch, 1, 12, 50)`，包含 12 个传感器通道，时间轴重采样到 50。
- 量化后 token 网格：`6x6`（36 token），匹配 codebook 大小 512，便于 Transformer 使用。

常见问题
- 单卡运行：自动退回 CPU/GPU 可用设备，但速度会下降。
- 若 DataLoader 报内存不足，可降低 `--batch-size` 或 `--max-negative-per-split`。

---

VQGAN->Token->Transformer(LM) 持续认证（server window 数据）

数据目录
- server 端已生成滑窗：`/data/code/server/data_storage/processed_data/window/<t>/<target_user>/{train,val,test}.csv`
- `train.csv` 仅包含真实用户；`val/test.csv` 含真实用户 + 冒充者（`subject==target_user` 为正类）。

核心脚本
- `hmog_vqgan_token_transformer_experiment.py`：先训练/加载 VQGAN（用于离散化 codebook），再训练/加载 Token-LM（Transformer/GPT）并用 NLL 做认证打分，支持 0.1–1.0 窗口扫参。
- `hmog_token_auth_inference.py`：加载已训练的 VQGAN + Token-LM，对任意同格式 CSV 做连续认证推理（输出每个 window 的 score/accept）。

运行示例（cuda1）
```bash
conda activate tremb1e
cd /data/code/server/ca_train

# 0.1~1.0 窗口扫参（示例：可用性优先，连续拒绝5次才打断）
python hmog_vqgan_token_transformer_experiment.py \
  --device cuda:1 \
  --users GzT_pGKIknsqGuvpvpLThmXMU1gKx1Bur4n_iSFdhIU= \
  --window-sizes 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --target-width 50 \
  --vqgan-epochs 1 --lm-epochs 1 \
  --max-negative-per-split 5000 --max-eval-per-split 10000 \
  --reuse-vqgan --reuse-lm \
  --k-rejects 5 --threshold-strategy k_session_frr --target-session-frr 0.0 \
  --log-dir results/experiment_logs/window_sweep_cuda1
```

推理示例
```bash
python hmog_token_auth_inference.py \
  --device cuda:1 \
  --csv-path /data/code/server/data_storage/processed_data/window/0.8/GzT_pGKIknsqGuvpvpLThmXMU1gKx1Bur4n_iSFdhIU=/test.csv \
  --window-size 0.8 --target-width 50 \
  --vqgan-checkpoint results/checkpoints/vqgan_user_GzT_pGKIknsqGuvpvpLThmXMU1gKx1Bur4n_iSFdhIU=_ws_0.8.pt \
  --lm-checkpoint results/checkpoints/token_gpt_user_GzT_pGKIknsqGuvpvpLThmXMU1gKx1Bur4n_iSFdhIU=_ws_0.8.pt \
  --threshold <VAL_THRESHOLD> \
  --k-rejects 5
```

阈值选择说明
- `hmog_vqgan_token_transformer_experiment.py` 在 `lm-val` 上选阈值，并固定到 `lm-test` 报告 FAR/FRR/F1（AUC/EER 为阈值无关指标）。
- 指标文件里的阶段名：
  - `vqgan-*(val/test)`：仅用 VQGAN 重建误差做异常分数（`score=-MSE` 或 `-L1`），属于 baseline。
  - `lm-*(val/test)`：VQGAN 将窗口离散化为 codebook token 后，用 Transformer(自回归 LM) 的 `score=-NLL(tokens)` 做认证分数（主方案）。
  - `lmseq-*(val/test)`：当设置 `--k-rejects > 0` 时额外输出的“序列/会话级”指标：按时间顺序逐窗口打分，并用“连续拒绝 K 次才打断”规则统计 `session_frr/session_far/session_tpr` 等。
- 如出现 AUC 很好但 F1 很低，通常是阈值过松导致 FAR 偏高（类别不平衡时 F1 对 FP 很敏感）。
- 当前实现里 `label=1` 表示真实用户（genuine），因此日志里的 `F1/precision/recall` 默认是在“接受真实用户”这个正类上计算；同时也会输出 `impostor_f1`（把“拒绝冒充者”视为正类）方便对照。
- 可切换阈值策略：
  - `--threshold-strategy eer`：默认，按 EER 选阈值。
  - `--threshold-strategy f1`：在 val 上选使 F1 最大的阈值。
  - `--threshold-strategy far --target-far 0.05`：在 val 上选满足 FAR≤target 的阈值并尽量提高 TPR（更贴近认证场景的“固定 FAR 工作点”）。
  - `--threshold-strategy frr --target-frr 0.001`：在 val 上选满足 FRR≤target 的阈值并尽量降低 FAR（更偏可用性，减少误报打断）。
    - 注意：当 val 正样本窗口数较少时，过小的 `target-frr` 会等价于“val 上 FRR=0”，阈值会非常宽松，可能在 test 上出现 FAR 接近 1 的退化现象。
  - `--threshold-strategy k_session_frr --k-rejects 5 --target-session-frr 0.0`：更贴近“持续性认证”的可用性目标：在 val 上选择**最严格**(阈值最高)但仍满足 `session_frr<=target-session-frr` 的阈值；其中 `session_frr` 是在“连续拒绝 K 次才打断”规则下统计的“真实用户被打断的会话比例”。该策略会在尽量不打断真实用户的前提下，最大化对冒充者的拒绝。

连续拒绝 K 次才打断（可用性优先）
- `--k-rejects K`：只有连续拒绝 K 个窗口才触发一次 `interrupt`（默认 0=关闭）。
- 推荐与 `--threshold-strategy k_session_frr` 联用来“保可用性”：例如 `K=5` 且 `--target-session-frr 0.0` 表示 val 上不打断真实用户（会牺牲对冒充者的检出率）。
- 重点看 `lmseq-test`：
  - `session_frr`：真实用户会话被打断比例（越低越好；你关注的“误报/打断”）。
  - `session_tpr`：冒充者会话被打断比例（检出率；你允许降低）。
  - `session_far`：冒充者会话未被打断比例（漏报；`session_far = 1 - session_tpr`）。
