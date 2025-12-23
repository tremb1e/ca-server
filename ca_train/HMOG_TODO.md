HMOG VQGAN 运行待办
1. 激活环境：`conda activate tremb1e`，确认 GPU 及 CUDA 可用（`nvidia-smi`）。
2. 进入项目根目录：`/data/code/CA-Model`，确保数据目录 `/data/code/ca/refer/ContinAuth/src/data/processed/Z-score_hmog_data_with_magnitude` 存在五个用户子目录。
3. 运行窗口扫参：`python hmog_vqgan_experiment.py --use-amp`；可通过 `--users 100669 151985 ...` 指定子集，或用默认遍历五个用户。
4. 监控日志：查看 `results/experiment_logs/hmog_vqgan.log`（控制台同样输出）以及 `results/experiment_logs/hmog_metrics.txt/jsonl`，关注 FAR/FRR/EER/AUC/F1/latency。
5. 完成扫参后，若需要更长训练，在日志中选择验证 AUC 最佳的窗口，使用 `--final-epochs` > `--sweep-epochs` 重新训练。
6. 训练结束后，检查 `results/experiment_logs/best_windows.json` 摘要与 `results/checkpoints` 下的最优 VQGAN 权重。
