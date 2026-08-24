# qlib-ashare-starter

基于 **Microsoft Qlib + AkShare** 的 A 股量化研究最小可用项目：
从真实行情数据下载，到 Alpha158 因子 + LightGBM 选股模型训练，再到考虑涨跌停限制与交易成本的组合回测，一条命令跑通全流程。

```
AkShare(东方财富) ──► CSV(后复权) ──► Qlib bin 格式 ──► Alpha158 因子
                                                        │
                                        LightGBM 训练 ◄─┘
                                              │
                              样本外预测(IC/RankIC) + TopkDropout 回测(vs 沪深300)
```

## 功能特性

- ✅ 一条 `make` 命令完成：环境 → 数据 → 训练 → 回测报告
- ✅ 沪深300 成分股日线（后复权）+ 指数基准，成交量单位自动校验
- ✅ Qlib 官方 `Alpha158` 因子集（158 个价量因子）
- ✅ `LightGBM` 模型 + 样本外 `IC/RankIC/ICIR` 评价
- ✅ 组合回测：TopkDropout（持 30 只、每日换 3 只），计入佣金/印花税/滑点、±10% 涨跌停不可成交约束
- ✅ 自动导出 IC 汇总、净值曲线 PNG、回测明细 CSV

## 目录结构

```
qlib-ashare-starter/
├── Makefile                     # make setup / data / train / analyze
├── requirements.txt             # 锁定 pyqlib==0.9.7 + numpy<2 的关键组合
├── config/
│   └── workflow_lightgbm.yaml   # 实验配置(时间切分/模型/回测参数)
├── scripts/
│   ├── fetch_data.py            # AkShare 数据下载(成分股+指数)
│   ├── dump_bin.py              # 微软官方转换脚本(vendored)
│   └── analyze_recorder.py      # 从 mlruns 提取报告
├── reports/                     # 回测输出(git 忽略大文件)
├── data/                        # 行情数据(git 忽略)
└── RESULTS.md                   # 最近一次运行的指标摘要
```

## 快速开始

环境要求：macOS(Apple Silicon 或 Intel)、[uv](https://github.com/astral-sh/uv)、Homebrew `libomp`(LightGBM 原生库的运行时依赖)、约 10 分钟。

```bash
brew install uv libomp     # 若未安装(libomp 是 LightGBM macOS 轮子所需的 OpenMP 运行时)
make setup                 # Python 3.11 venv + 依赖(pyqlib 有预编译轮子, 无需编译)
make data                  # 下载沪深300+指数日线(~600 次请求, 约5-15分钟)并转 Qlib 格式
make train                 # Alpha158 + LightGBM 训练 + 样本外回测
make analyze               # 生成净值曲线与 IC 报告
```

或一步到位：`make all`

## 实验设计

| 项目 | 设定 |
|---|---|
| 股票池 | 当前沪深300成分股（约300只） |
| 数据区间 | 2020-09 ~ 今（后复权日线） |
| 训练集 | 2021-01-01 ~ 2024-12-31 |
| 验证集 | 2025-01-01 ~ 2025-06-30 |
| **测试集** | **2025-07-01 ~ 2026-08-21（纯样本外）** |
| 特征 | Alpha158（158 个价量因子，RobustZScoreNorm 标准化） |
| 标签 | `Ref($close,-2)/Ref($close,-1)-1`（T+1 可交易口径） |
| 模型 | LightGBM（官方基准调参，n_estimators=400） |
| 组合 | TopkDropout：持有 IC 最高 30 只，每日换出 3 只 |
| 成本 | 买 0.13% / 卖 0.23%（含印花税）/ 最低 5 元；涨跌停不可成交 |

### 如何解读输出

- **IC / RankIC**：预测值与次日收益的截面相关性，日度均值 >0.03 且 `ICIR>0.3` 即有实际选股价值；
- **回测表**：重点看样本外(test 段)的年化超额、夏普、最大回撤——对比基准 SH000300；
- `reports/equity_curve.png`：策略净值 vs 指数。

## 已知限制（诚实声明）

1. **幸存者偏差**：股票池使用*当前*成分股名单回测历史，退市/被调出个股未包含，结果偏乐观。生产级研究需使用历史时点成分名单。
2. 复权采用后复权口径，回测中的持仓数量为近似处理（不影响收益率结论的数量级）。
3. 数据来自免费接口（东财），仅供研究学习，**不构成投资建议**；实盘前请用更严格的数据源与滚动前推(walk-forward)验证。

## 接入你自己的 GitHub

```bash
cd qlib-ashare-starter
git remote add origin git@github.com:<你的用户名>/qlib-ashare-starter.git
git push -u origin main
```

## 扩展方向

- 换模型：把 `config/workflow_lightgbm.yaml` 中 `model.class` 改为 `qlib.contrib.model.pytorch_nn.DNN`、`qlib.contrib.model.gatstm.GATs` 等；
- 加数据：在 `scripts/fetch_data.py` 增加 `ak.stock_financial_abstract` 财务因子；
- 滚动前推：参考 Qlib 官方 `rolling` benchmark 把单次切分改为多窗口滚动。

## License

MIT（`scripts/dump_bin.py` 版权归 Microsoft Qlib 作者，遵循其原许可协议）。
