# 二次迟滞决策 HMOG 性能对比报告

## 测试方法

- 模型目录：`/data/code/backup/ca-server/deploy/data/models`
- 数据来源：各用户 `policy_search/cache/vqgan_only/ws_0.2/test.npz`，复用已有 VQGAN-only 测试集分数。
- 一次迟滞：按各用户 `best_lock_policy.json` 中的 EMA alpha 与 threshold 复现，然后按 1 秒节拍抽样为一次返回给 App 的认证结果。
- 二次迟滞 EMA+EMA：对一次结果的 reject(0/1) 序列做 EMA，默认 10 秒输出一次，`alpha=0.25`，reject 阈值 `0.8`。
- 二次迟滞 EMA+vote：对最近 10 个一次结果做 8-of-10 投票，10 秒输出一次。

## 汇总结果

| 方案 | 输出数 | 误报率 FRR | 漏报率 FAR | 攻击拒绝率 | ERR | 攻击平均首次拒绝(s) | 真实平均首次误拒(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 一次迟滞 EMA | 10157 | 0.96% | 91.50% | 8.50% | 46.07% | 9.16 | 209.70 |
| 二次迟滞 EMA+EMA | 958 | 0.00% | 93.91% | 6.09% | 45.09% | 18.75 | 0.00 |
| 二次迟滞 EMA+vote 8/10 | 958 | 0.00% | 93.91% | 6.09% | 45.09% | 20.00 | 0.00 |

## 结论

- 二次迟滞显著降低了输出频率；默认 10 个一次结果聚合为 1 个二次结果，本次从 10157 个一次输出降到 958 个二次输出。
- 在当前 HMOG 缓存与既有一次 EMA 阈值下，二次迟滞把真实用户误报率 FRR 从 0.96% 降到 0.00%，说明它确实抑制了偶发误拒。
- 代价是攻击者漏报率 FAR 从 91.50% 升到 93.91%，攻击拒绝率从 8.50% 降到 6.09%，并且攻击平均首次拒绝时间从 9.16s 延迟到 18.75s/20.00s。
- 因此“二次迟滞一定同时降低误报率和漏报率”在当前数据上不成立；默认 8-of-10 更偏保守，适合优先保护真实用户体验，但需要继续搜索二次阈值/比例来平衡漏报。
- `EMA+EMA` 与 `EMA+vote 8/10` 在当前参数下总体 FAR/FRR 接近，差异主要体现在首次拒绝时间：EMA+EMA 略早，vote 更稳定但更晚。
- 本报告按已有 HMOG 测试集缓存离线复现，不重新训练模型；线上结果还会受 App 批量上传节奏、网络积压和 session 长度影响。

## 分用户结果

| 用户 | 方案 | 输出数 | FRR | FAR | 攻击拒绝率 | ERR | 攻击平均首次拒绝(s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3tCcdso4dOYpDcGI3yHBB2bzXorXXukBYJPg_jM8eK4= | 一次迟滞 EMA | 3899 | 1.53% | 82.94% | 17.06% | 42.04% | 11.82 |
| 3tCcdso4dOYpDcGI3yHBB2bzXorXXukBYJPg_jM8eK4= | 二次迟滞 EMA+EMA | 375 | 0.00% | 88.33% | 11.67% | 42.40% | 17.50 |
| 3tCcdso4dOYpDcGI3yHBB2bzXorXXukBYJPg_jM8eK4= | 二次迟滞 EMA+vote 8/10 | 375 | 0.00% | 88.33% | 11.67% | 42.40% | 20.00 |
| O3lLYw8FJ-i2-gMx_oXyjfnPh5UgM7MUObX-sJRiAws= | 一次迟滞 EMA | 3530 | 1.02% | 94.55% | 5.45% | 47.65% | 11.67 |
| O3lLYw8FJ-i2-gMx_oXyjfnPh5UgM7MUObX-sJRiAws= | 二次迟滞 EMA+EMA | 327 | 0.00% | 95.62% | 4.38% | 46.79% | 20.00 |
| O3lLYw8FJ-i2-gMx_oXyjfnPh5UgM7MUObX-sJRiAws= | 二次迟滞 EMA+vote 8/10 | 327 | 0.00% | 95.62% | 4.38% | 46.79% | 20.00 |
| SABwzzvhSEzsTNLmScnCtxJFVc2032FbpeHuHCai4MY= | 一次迟滞 EMA | 2728 | 0.07% | 99.78% | 0.22% | 49.78% | 4.00 |
| SABwzzvhSEzsTNLmScnCtxJFVc2032FbpeHuHCai4MY= | 二次迟滞 EMA+EMA | 256 | 0.00% | 100.00% | 0.00% | 46.88% | 0.00 |
| SABwzzvhSEzsTNLmScnCtxJFVc2032FbpeHuHCai4MY= | 二次迟滞 EMA+vote 8/10 | 256 | 0.00% | 100.00% | 0.00% | 46.88% | 0.00 |

## 明细

完整明细见同目录 CSV：`secondary_hysteresis_hmog_metrics.csv`。
