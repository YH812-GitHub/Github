# 实验结果：Alpha158 + LightGBM 沪深300 选股（TopkDropout30）

> 一次完整、可复现的 A 股量化研究流程：Qlib bin 数据 → Alpha158 因子 → LightGBM 训练 → 样本外 TopkDropout 组合回测（含交易成本与涨跌停约束）。
> 复现方式见文末；本文所有数字均由 `make train` / `make analyze` 自动产物中提取。

## 1. 结论速览

| 指标 | 数值 |
|---|---|
| 样本外区间 | 2025-07-01 ~ 2026-08-21（280 个交易日，纯样本外） |
| 日度 IC（均值） | **-0.0161**（ICIR -0.081，IC>0 占比 46.1%） |
| 日度 Rank IC（均值） | **+0.0147**（RankICIR +0.070，IC>0 占比 52.9%） |
| 策略累计收益（含成本） | **+6.62%** |
| 基准 SH000300 累计收益 | **+17.35%** |
| 超额收益（含成本） | **-9.14%**（年化 -17.42%，IR -1.19） |
| 超额收益（无成本） | 年化 -8.88%（IR -0.61） |
| 策略区间最大回撤 | -7.74%（基准同期 -10.49%） |
| 日均换手率 | 20.0%（单边） |

**核心结论**：模型排序能力弱正但接近随机（Rank IC ≈ +0.015），不足以覆盖成本；样本外策略显著跑输基准。日均 20% 换手带来的成本拖累约 8.5 个百分点年化（-8.9% → -17.4%）。这是一个如实的负结果：在本配置下（单次切分、默认超参）Alpha158+LightGBM 在该时段没有产生可用的超额收益。

## 2. 实验设置

### 数据
- 标的池：沪深300 成分股（剔除基准指数 SH000300 后 300 只），后复权（`factor` 字段）
- 数据区间：2020-09-01 ~ 2026-08-24（1449 个交易日）；字段 open/high/low/close/volume/factor/vwap
- 来源：akshare 东财接口，限流自动降级链（东财→新浪→腾讯）

### 特征与标签
- 特征：Alpha158 全集（158 个 K线/量价因子）
- 标准化：`RobustZScoreNorm` 仅在训练段（fit_start/end=2021-01-01~2024-12-31）内拟合，无标准化泄漏；`Fillna` 补缺
- 标签：`Ref($close,-2) / Ref($close,-1) - 1`（T+1 收盘信号 → T+2 相对 T+1 的收益，规避未来函数）；学习段做截面 `CSRankNorm`

### 时间切分（滚动前推最小版）
| 段 | 区间 | 用途 |
|---|---|---|
| train | 2021-01-01 ~ 2024-12-31（4 年） | 拟合（含 RobustZScoreNorm fit） |
| valid | 2025-01-01 ~ 2025-06-30 | 早停验证 |
| test | 2025-07-01 ~ 2026-08-21（约 14 个月） | 样本外 IC 与回测 |

### 模型
- LightGBM（qlib `LGBModel`，官方 Alpha158 默认超参）：400 棵树上限，早停 50 轮 → **best_iter=45**，早停于验证段

### 回测（含现实约束）
- 组合：`TopkDropoutStrategy` topk=30, n_drop=3，等权持有一篮子
- 账户 1 亿；成交价=收盘价；涨跌停约束 `limit_threshold=±9.5%`（对创业板/科创板 ±20% 属保守误禁，方向安全）
- 成本：买入 13bp（佣金+滑点）、卖出 23bp（佣金+印花税+滑点）、单笔最低 5 元

## 3. 结果解读

1. **预测力**：Rank IC +0.0147 说明排序信号有极弱的正向信息，但 ICIR 0.07 的月度稳定性几乎不可区分于噪声（280 天里仅 52.9% 天数为正）。普通 IC 均值为负，说明信号对极端收益的线性刻画甚至是反向的——典型的"只可排序、不可定价"形态。
2. **成本敏感性**：无成本超额年化 -8.88%，含成本 -17.42%。20% 日换手 × 平均约 18bp 单边成本 ≈ 年化 8.5pp 拖累。即使信号为真，当前换手水平也会吞噬全部 alpha。
3. **回撤**：策略最大回撤 -7.74%，小于基准的 -10.49%——分散持有 30 只等权组合天然压低波动，是本配置下唯一相对占优的风险特征。
4. **执行质量**：fill ratio（ffr）= 1.0，订单全部成交，无因涨跌停导致的未完成建仓异常。

## 4. 已知局限与注意事项

- **单次时间切分、单种子**：结论不构成对该因子体系的一般性否定；标准做法是多窗口滚动训练 + 多种子平均后再评估。
- **超参未调优**：LightGBM 使用官方演示超参，早停在验证段 45 轮即触发，模型容量明显受限（train l2 0.985 / valid l2 0.996，CSRankNorm 下 1.0 为随机水平）。
- **valid 段标签轻微溢出**：训练段末两日的标签窗口跨入验证段（官方示例同款口径），影响可忽略。
- **indicators 中 pa/pos 恒为 0**：`indicator_analysis_1day.pkl` 的 price-advantage/position 两项诊断指标为 0，属 qlib 指标统计口径问题，不影响主结果（report_normal_1day 中的真实成本与成交记录完整）。
- **mlflow 兼容性**：mlflow≥3 将本地文件存储置于维护模式，Makefile 已通过 `MLFLOW_ALLOW_FILE_STORE=true` 放行。
- **数据尾部说明**：本文产物由一次 `--end` 略宽于 `DATA_END` 的抓取生成，日历尾部含 2026-08-24（比复现口径多 1 个交易日，对测试段末端标签影响可忽略）；干净检出按 `make data` 重跑则以 `DATA_END ?= 20260821` 为准。

## 5. 产物清单

| 文件 | 说明 |
|---|---|
| `reports/ic_summary.txt` | IC/RankIC 汇总（入库） |
| `reports/equity_curve.png` | 策略 vs 基准累计净值曲线（本地生成） |
| `reports/sig_analysis_ic.csv` / `_ric.csv` | 每日 IC / RankIC 序列 |
| `reports/portfolio_analysis_port_analysis_1day.csv` | 年化收益/IR/最大回撤（含/不含成本） |
| `reports/portfolio_analysis_report_normal_1day.csv` | 逐日账户收益、换手、成本、持仓市值 |
| `reports/pred.csv`, `label.csv` | 样本外预测分数与标签 |
| `reports/qrun_output.log` | qrun 全量日志 |

## 6. 复现步骤

```bash
brew install libomp          # macOS 上 LightGBM 运行时依赖
make setup                   # uv venv + 依赖
make data                    # 下载 300 成分+指数日线 → CSV → Qlib bin
make train                   # 训练 + 样本外回测（自动写 reports/qrun_output.log）
make analyze                 # 导出 IC 汇总 / 净值图 / 各类 CSV
```

关键版本：Python 3.11 · pyqlib 0.9.7 · lightgbm 4.7.0 · pandas 2.3.3 · akshare 1.18.94。
数据截止日固定在 Makefile `DATA_END ?= 20260821` 以保证可复现。
