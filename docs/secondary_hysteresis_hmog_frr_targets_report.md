# HMOG 不同 FRR 目标下的二次迟滞性能对比报告

## 测试方法

- 模型目录：`/data/code/backup/ca-server/deploy/data/models`
- 数据来源：各用户 `policy_search/cache/vqgan_only/ws_0.2/{val,test}.npz`，复用已有 VQGAN-only 分数缓存。
- 阈值选择：每个用户、每个目标 FRR、每个方案都在 `val.npz` 上选择最高 EMA 分数阈值，使该方案的验证集 FRR 不超过目标值。
- 评估方式：把验证集选出的阈值固定后，在 `test.npz` 上计算性能指标；不限制、不惩罚检测出时间，首次拒绝时间只作为观察指标。
- 一次迟滞：VQGAN 分数先按用户 `best_lock_policy.json` 中的 `ema_alpha` 做 EMA，再按 1 秒节拍抽样为一次返回结果。
- 二次迟滞 EMA+EMA：对一次结果 reject(0/1) 序列做二次 EMA，10 个一次结果输出一次，`alpha=0.25`，reject 阈值 `0.8`。
- 二次迟滞 EMA+vote：对最近 10 个一次结果做 `8-of-10` 投票，10 个一次结果输出一次。

## 汇总结果

| 目标 FRR | 方案 | 验证 FRR | 验证 FAR | 测试 FRR | 测试 FAR | 攻击拒绝率 | ERR | 输出数 | 攻击平均首次拒绝(s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | 一次迟滞 EMA | 4.97% | 55.95% | 4.24% | 55.95% | 44.05% | 30.00% | 10157 | 3.92 |
| 5% | 二次迟滞 EMA+EMA | 4.61% | 50.00% | 3.01% | 50.00% | 50.00% | 25.57% | 958 | 18.03 |
| 5% | 二次迟滞 EMA+vote 8/10 | 4.41% | 48.91% | 2.81% | 48.91% | 51.09% | 24.95% | 958 | 19.87 |
| 10% | 一次迟滞 EMA | 9.95% | 45.28% | 8.53% | 45.28% | 54.72% | 26.84% | 10157 | 6.73 |
| 10% | 二次迟滞 EMA+EMA | 9.62% | 27.61% | 7.43% | 27.61% | 72.39% | 17.12% | 958 | 12.97 |
| 10% | 二次迟滞 EMA+vote 8/10 | 9.62% | 22.39% | 8.43% | 22.39% | 77.61% | 15.14% | 958 | 14.28 |
| 15% | 一次迟滞 EMA | 14.98% | 33.28% | 11.95% | 33.28% | 66.72% | 22.58% | 10157 | 3.53 |
| 15% | 二次迟滞 EMA+EMA | 14.63% | 18.70% | 9.44% | 18.70% | 81.30% | 13.88% | 958 | 12.23 |
| 15% | 二次迟滞 EMA+vote 8/10 | 14.63% | 15.65% | 12.25% | 15.65% | 84.35% | 13.88% | 958 | 13.92 |
| 20% | 一次迟滞 EMA | 19.98% | 21.90% | 16.05% | 21.90% | 78.10% | 18.96% | 10157 | 2.48 |
| 20% | 二次迟滞 EMA+EMA | 19.84% | 15.22% | 12.25% | 15.22% | 84.78% | 13.67% | 958 | 12.75 |
| 20% | 二次迟滞 EMA+vote 8/10 | 19.84% | 14.78% | 15.26% | 14.78% | 85.22% | 15.03% | 958 | 14.63 |

## 分析结论

- 目标 5%：一次 EMA 测试 FAR=55.95%，二次 EMA+EMA FAR=50.00%，二次 EMA+vote FAR=48.91%；对应测试 FRR 分别为 4.24%、3.01%、2.81%。
- 目标 10%：一次 EMA 测试 FAR=45.28%，二次 EMA+EMA FAR=27.61%，二次 EMA+vote FAR=22.39%；对应测试 FRR 分别为 8.53%、7.43%、8.43%。
- 目标 15%：一次 EMA 测试 FAR=33.28%，二次 EMA+EMA FAR=18.70%，二次 EMA+vote FAR=15.65%；对应测试 FRR 分别为 11.95%、9.44%、12.25%。
- 目标 20%：一次 EMA 测试 FAR=21.90%，二次 EMA+EMA FAR=15.22%，二次 EMA+vote FAR=14.78%；对应测试 FRR 分别为 16.05%、12.25%、15.26%。
- 由于本次明确不限制检测出时间，阈值选择只受验证集 FRR 约束；测试集 FRR 可能因数据分布差异略高或略低于目标。
- 二次迟滞会显著降低输出频率，因此其 FRR/FAR 是按最终返回给 App 的二次结果统计，不是按每秒一次结果统计。
- 在相同目标 FRR 下，二次迟滞通常需要更激进的一次 EMA 阈值才能用满 FRR 预算；这可能降低 FAR，但也会推迟或减少最终拒绝次数。

## 分用户结果

| 用户 | 目标 FRR | 方案 | 阈值 | 验证 FRR | 测试 FRR | 测试 FAR | 攻击拒绝率 | ERR | 输出数 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3tCcdso4dOYpDcGI3yHBB2bzXorXXukBYJPg_jM8eK4= | 5% | 一次迟滞 EMA | -0.077992 | 4.95% | 6.13% | 25.05% | 74.95% | 15.54% | 3899 |
| 3tCcdso4dOYpDcGI3yHBB2bzXorXXukBYJPg_jM8eK4= | 5% | 二次迟滞 EMA+EMA | -0.077298 | 4.62% | 2.05% | 37.22% | 62.78% | 18.93% | 375 |
| 3tCcdso4dOYpDcGI3yHBB2bzXorXXukBYJPg_jM8eK4= | 5% | 二次迟滞 EMA+vote 8/10 | -0.077486 | 4.62% | 2.05% | 34.44% | 65.56% | 17.60% | 375 |
| 3tCcdso4dOYpDcGI3yHBB2bzXorXXukBYJPg_jM8eK4= | 10% | 一次迟滞 EMA | -0.077469 | 9.96% | 6.18% | 24.48% | 75.52% | 15.29% | 3899 |
| 3tCcdso4dOYpDcGI3yHBB2bzXorXXukBYJPg_jM8eK4= | 10% | 二次迟滞 EMA+EMA | -0.077104 | 9.74% | 2.05% | 36.67% | 63.33% | 18.67% | 375 |
| 3tCcdso4dOYpDcGI3yHBB2bzXorXXukBYJPg_jM8eK4= | 10% | 二次迟滞 EMA+vote 8/10 | -0.077137 | 9.74% | 2.05% | 34.44% | 65.56% | 17.60% | 375 |
| 3tCcdso4dOYpDcGI3yHBB2bzXorXXukBYJPg_jM8eK4= | 15% | 一次迟滞 EMA | -0.077115 | 14.96% | 6.33% | 24.28% | 75.72% | 15.26% | 3899 |
| 3tCcdso4dOYpDcGI3yHBB2bzXorXXukBYJPg_jM8eK4= | 15% | 二次迟滞 EMA+EMA | -0.076773 | 14.87% | 2.05% | 35.56% | 64.44% | 18.13% | 375 |
| 3tCcdso4dOYpDcGI3yHBB2bzXorXXukBYJPg_jM8eK4= | 15% | 二次迟滞 EMA+vote 8/10 | -0.076824 | 14.87% | 2.05% | 34.44% | 65.56% | 17.60% | 375 |
| 3tCcdso4dOYpDcGI3yHBB2bzXorXXukBYJPg_jM8eK4= | 20% | 一次迟滞 EMA | -0.076812 | 19.97% | 6.33% | 24.02% | 75.98% | 15.13% | 3899 |
| 3tCcdso4dOYpDcGI3yHBB2bzXorXXukBYJPg_jM8eK4= | 20% | 二次迟滞 EMA+EMA | -0.076363 | 20.00% | 2.05% | 35.00% | 65.00% | 17.87% | 375 |
| 3tCcdso4dOYpDcGI3yHBB2bzXorXXukBYJPg_jM8eK4= | 20% | 二次迟滞 EMA+vote 8/10 | -0.076483 | 20.00% | 2.05% | 33.89% | 66.11% | 17.33% | 375 |
| O3lLYw8FJ-i2-gMx_oXyjfnPh5UgM7MUObX-sJRiAws= | 5% | 一次迟滞 EMA | -0.213726 | 4.96% | 4.52% | 58.52% | 41.48% | 31.44% | 3530 |
| O3lLYw8FJ-i2-gMx_oXyjfnPh5UgM7MUObX-sJRiAws= | 5% | 二次迟滞 EMA+EMA | -0.174609 | 4.62% | 0.00% | 60.00% | 40.00% | 29.36% | 327 |
| O3lLYw8FJ-i2-gMx_oXyjfnPh5UgM7MUObX-sJRiAws= | 5% | 二次迟滞 EMA+vote 8/10 | -0.174609 | 4.62% | 0.00% | 61.25% | 38.75% | 29.97% | 327 |
| O3lLYw8FJ-i2-gMx_oXyjfnPh5UgM7MUObX-sJRiAws= | 10% | 一次迟滞 EMA | -0.178069 | 9.97% | 5.48% | 52.67% | 47.33% | 29.01% | 3530 |
| O3lLYw8FJ-i2-gMx_oXyjfnPh5UgM7MUObX-sJRiAws= | 10% | 二次迟滞 EMA+EMA | -0.037461 | 9.83% | 4.19% | 16.25% | 83.75% | 10.09% | 327 |
| O3lLYw8FJ-i2-gMx_oXyjfnPh5UgM7MUObX-sJRiAws= | 10% | 二次迟滞 EMA+vote 8/10 | -0.042226 | 9.83% | 1.80% | 17.50% | 82.50% | 9.48% | 327 |
| O3lLYw8FJ-i2-gMx_oXyjfnPh5UgM7MUObX-sJRiAws= | 15% | 一次迟滞 EMA | -0.094708 | 14.99% | 9.89% | 34.94% | 65.06% | 22.38% | 3530 |
| O3lLYw8FJ-i2-gMx_oXyjfnPh5UgM7MUObX-sJRiAws= | 15% | 二次迟滞 EMA+EMA | -0.030230 | 14.45% | 5.99% | 5.62% | 94.38% | 5.81% | 327 |
| O3lLYw8FJ-i2-gMx_oXyjfnPh5UgM7MUObX-sJRiAws= | 15% | 二次迟滞 EMA+vote 8/10 | -0.030338 | 14.45% | 7.19% | 5.62% | 94.38% | 6.42% | 327 |
| O3lLYw8FJ-i2-gMx_oXyjfnPh5UgM7MUObX-sJRiAws= | 20% | 一次迟滞 EMA | -0.055623 | 20.00% | 13.79% | 17.73% | 82.27% | 15.75% | 3530 |
| O3lLYw8FJ-i2-gMx_oXyjfnPh5UgM7MUObX-sJRiAws= | 20% | 二次迟滞 EMA+EMA | -0.019923 | 19.65% | 10.18% | 4.38% | 95.62% | 7.34% | 327 |
| O3lLYw8FJ-i2-gMx_oXyjfnPh5UgM7MUObX-sJRiAws= | 20% | 二次迟滞 EMA+vote 8/10 | -0.020849 | 19.65% | 13.77% | 3.75% | 96.25% | 8.87% | 327 |
| SABwzzvhSEzsTNLmScnCtxJFVc2032FbpeHuHCai4MY= | 5% | 一次迟滞 EMA | -1.842589 | 5.00% | 1.17% | 96.69% | 3.31% | 48.79% | 2728 |
| SABwzzvhSEzsTNLmScnCtxJFVc2032FbpeHuHCai4MY= | 5% | 二次迟滞 EMA+EMA | -0.181146 | 4.58% | 8.09% | 55.83% | 44.17% | 30.47% | 256 |
| SABwzzvhSEzsTNLmScnCtxJFVc2032FbpeHuHCai4MY= | 5% | 二次迟滞 EMA+vote 8/10 | -0.179148 | 3.82% | 7.35% | 54.17% | 45.83% | 29.30% | 256 |
| SABwzzvhSEzsTNLmScnCtxJFVc2032FbpeHuHCai4MY= | 10% | 一次迟滞 EMA | -0.280373 | 9.92% | 15.86% | 65.37% | 34.63% | 40.54% | 2728 |
| SABwzzvhSEzsTNLmScnCtxJFVc2032FbpeHuHCai4MY= | 10% | 二次迟滞 EMA+EMA | -0.079907 | 9.16% | 19.12% | 29.17% | 70.83% | 23.83% | 256 |
| SABwzzvhSEzsTNLmScnCtxJFVc2032FbpeHuHCai4MY= | 10% | 二次迟滞 EMA+vote 8/10 | -0.050716 | 9.16% | 25.74% | 10.83% | 89.17% | 18.75% | 256 |
| SABwzzvhSEzsTNLmScnCtxJFVc2032FbpeHuHCai4MY= | 15% | 一次迟滞 EMA | -0.164435 | 14.99% | 22.66% | 43.97% | 56.03% | 33.28% | 2728 |
| SABwzzvhSEzsTNLmScnCtxJFVc2032FbpeHuHCai4MY= | 15% | 二次迟滞 EMA+EMA | -0.045322 | 14.50% | 24.26% | 10.83% | 89.17% | 17.97% | 256 |
| SABwzzvhSEzsTNLmScnCtxJFVc2032FbpeHuHCai4MY= | 15% | 二次迟滞 EMA+vote 8/10 | -0.038585 | 14.50% | 33.09% | 0.83% | 99.17% | 17.97% | 256 |
| SABwzzvhSEzsTNLmScnCtxJFVc2032FbpeHuHCai4MY= | 20% | 一次迟滞 EMA | -0.092282 | 19.99% | 32.89% | 24.26% | 75.74% | 28.59% | 2728 |
| SABwzzvhSEzsTNLmScnCtxJFVc2032FbpeHuHCai4MY= | 20% | 二次迟滞 EMA+EMA | -0.035887 | 19.85% | 29.41% | 0.00% | 100.00% | 15.62% | 256 |
| SABwzzvhSEzsTNLmScnCtxJFVc2032FbpeHuHCai4MY= | 20% | 二次迟滞 EMA+vote 8/10 | -0.035887 | 19.85% | 36.03% | 0.83% | 99.17% | 19.53% | 256 |

## 明细

完整明细见同目录 CSV：`secondary_hysteresis_hmog_frr_targets_metrics.csv`。
