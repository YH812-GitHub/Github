#!/usr/bin/env python
"""
每日信号生成器（10万元模拟盘）:
  - 加载 mlruns 中最新 FINISHED recorder 的训练好的模型(params.pkl, LGBModel)
  - 用全部 csi300 标的、信号日(默认日历最后交易日)的 Alpha158 特征做预测
  - 输出今日 Top-8 可建仓清单(代码/预测分/收盘价/一手工需资金/预计占用金额),
    并单列因股价超过单只预算(100000/8=12500 元/手)被资金约束排除的高分标的
  - 结果写 reports/today_signal.txt 并打印

用法:
    make signal                                  # 默认日历最后交易日
    .venv/bin/python scripts/today_signal.py --date 2026-08-21

口径说明:
  - 【推理特征】完整复现训练期 infer_processors(RobustZScoreNorm clip_outlier + Fillna),
    fit 窗口锁训练段 2021-01-01~2024-12-31 与 workflow_lightgbm*.yaml 同口径;
    取数起点固定为数据起始日 2020-09-01, 同时覆盖 Alpha158 最长回看窗与处理器
    fit 窗口(否则归一化参数为 NaN、预测分退化)。
  - 【价格口径】库内 $close 为后复权价; 一手工需资金必须用真实价 = $close/$factor。
    例: 贵州茅台复权价 11306 元但真实价约 1273 元, 若误用复权价会严重高估资金门槛。
  - 资金约束: 一手 = 真实价×100 股, > 单只预算 12500 元即无法成交任何整数手。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

ACCOUNT = 100_000          # 与 workflow_lightgbm_10w.yaml 的 account 一致
TOPK = 8                   # 同上 strategy.topk
LOT = 100                  # A股一手股数


def main() -> None:
    parser = argparse.ArgumentParser(description="每日 Top-8 建仓信号")
    parser.add_argument("--date", default=None, help="信号日期 YYYY-MM-DD, 默认取日历最后交易日")
    parser.add_argument("--experiment", default="workflow_10w",
                        help="优先使用的实验名(默认 workflow_10w), 无 recorder 时回退主实验 workflow")
    args = parser.parse_args()

    REPORTS.mkdir(exist_ok=True)
    import qlib  # noqa: PLC0415

    qlib.init(provider_uri=str(ROOT / "data" / "qlib_data" / "cn_data"), region="cn")
    from qlib.contrib.data.handler import Alpha158  # noqa: PLC0415
    from qlib.data import D  # noqa: PLC0415
    from qlib.utils import init_instance_by_config  # noqa: PLC0415

    # ---- 1. 取最新 FINISHED recorder 的训练好的模型 --------------------------
    from qlib.workflow import R  # noqa: PLC0415

    def _pick(exp_name):
        recs_ = R.get_exp(experiment_name=exp_name).list_recorders()
        fin_ = [r for r in recs_.values() if "FINISHED" in str(getattr(r, "status", "")).upper()]
        pool_ = fin_ or list(recs_.values())
        newest = max(pool_, key=lambda r: str(getattr(r, "start_time", "") or "")) if pool_ else None
        return newest, len(fin_), len(recs_)

    rec, n_fin, n_all = _pick(args.experiment)
    used_exp = args.experiment
    if rec is None:                                         # 未跑过 train10w 时回退主实验
        rec, n_fin, n_all = _pick("workflow")
        used_exp = "workflow"
    if rec is None:
        raise SystemExit("未找到任何 recorder, 请先运行 make train / make train10w")
    print(f"使用实验 {used_exp} 的 recorder: {rec.id}  (FINISHED {n_fin}/{n_all})")

    model = rec.load_object("params.pkl")
    if not hasattr(model, "model") or not hasattr(model.model, "predict"):
        raise SystemExit(f"不支持的模型类型: {type(model).__name__} (期望 LGBModel)")

    # ---- 2. 信号日期 ----------------------------------------------------------
    cal = D.calendar(freq="day")
    last_day = pd.Timestamp(cal[-1])
    signal_date = pd.Timestamp(args.date) if args.date else last_day
    print(f"信号日期: {signal_date.date()}  (日历最后交易日: {last_day.date()})")

    # ---- 3. 信号日 Alpha158 特征 → 预测 ---------------------------------------
    # 【关键】必须复现训练期 infer_processors(RobustZScoreNorm+Fillna): 模型在
    # 归一化+截尾空间学的分裂阈值, 直接吃原始值会因分布漂移导致叶子错位(t8 #1)。
    # 取数起点固定为数据起始日: 一则覆盖 Alpha158 最长回看窗, 二则必须完整覆盖
    # 处理器 fit 窗口(2021-01-01~2024-12-31), 否则中位数/IQR 为 NaN、特征被
    # Fillna 成常数行, 预测分退化为同一值。
    start_pad = "2020-09-01"                                # 数据起始日, 覆盖回看窗与 fit 窗
    handler = init_instance_by_config({
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": start_pad,
            "end_time": signal_date.strftime("%Y-%m-%d"),
            "instruments": "csi300",
            "fit_start_time": "2021-01-01",
            "fit_end_time": "2024-12-31",
            "infer_processors": [
                {"class": "RobustZScoreNorm",
                 "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna",
                 "kwargs": {"fields_group": "feature"}},
            ],
        },
    })
    features = handler.fetch(col_set="feature")             # MultiIndex(datetime, instrument)

    # 自检防复发: RobustZScoreNorm 的 fit 窗口(2021~2024)必须被取数窗口覆盖,
    # 否则 mean/std 为 NaN → 特征全 0 → 预测分退化为同一值且不报错
    import numpy as np  # noqa: PLC0415

    procs = list(getattr(handler, "infer_processors", []) or [])
    rz = next((p for p in procs if type(p).__name__ == "RobustZScoreNorm"), None)
    if rz is not None:
        mtn = getattr(rz, "mean_train", None)
        assert mtn is None or np.isfinite(mtn).all(), \
            "RobustZScoreNorm fit 窗口未被数据覆盖(start_time 太晚), 特征将退化为全零"

    try:
        day_x = features.xs(signal_date, level=0)
    except KeyError:
        raise SystemExit(f"{signal_date.date()} 无特征数据, 请检查日期是否为交易日")
    if day_x.empty:
        raise SystemExit(f"{signal_date.date()} 无特征数据, 请检查日期是否为交易日")

    pred = pd.Series(model.model.predict(day_x.values), index=day_x.index)
    pred = pred.sort_values(ascending=False)

    # ---- 4. 真实收盘价与资金约束 ----------------------------------------------
    # 库内 $close 为后复权价, 下单资金门槛必须用真实价 = $close / $factor
    px = D.features(list(pred.index), ["$close", "$factor"],
                    start_time=signal_date, end_time=signal_date, freq="day")
    real_s = (px["$close"] / px["$factor"]).xs(signal_date, level="datetime")

    budget_per_slot = ACCOUNT / TOPK                        # 12,500 元/只

    def _info(inst, score):
        price = float(real_s.get(inst, float("nan")))       # 真实价(元/股)
        return {"instrument": inst, "score": float(score),
                "close": price, "lot_cost": price * LOT}

    picked, excluded, no_quote = [], [], []
    for inst, score in pred.items():                        # 分数降序全量扫描
        info = _info(inst, score)
        if info["close"] != info["close"]:                  # NaN = 当日停牌/无报价
            if len(picked) < TOPK:
                no_quote.append({**info, "score": float(score)})
            continue
        affordable = info["lot_cost"] <= budget_per_slot
        if len(picked) < TOPK:
            if affordable:
                picked.append({**info, "alloc_cash": round(budget_per_slot, 2)})
            else:
                excluded.append({**info,
                                 "reason": f"一手工需 {info['lot_cost']:,.0f} 元 > 单只预算 {budget_per_slot:,.0f} 元"})
        else:
            break                                           # 已选满 TOPK 只可成交标的, 停止扫描

    # ---- 5. 输出 -----------------------------------------------------------------
    lines = [f"=== 每日建仓信号 | {signal_date.date()} | 账户 {ACCOUNT:,} 元 | Top{TOPK} ===",
             f"(回测含 risk_degree=0.95: 实际投入上限约 {ACCOUNT*0.95:,.0f} 元, 下表按满额 12,500 元/只展示)"]
    lines.append(f"{'代码':<10}{'预测分':>10}{'真实收盘':>10}{'一手工需':>14}{'预计占用':>12}")
    used = sum(r["alloc_cash"] for r in picked)
    for r in picked:
        lines.append(f"{r['instrument']:<10}{r['score']:>10.4f}{r['close']:>10.2f}"
                     f"{r['lot_cost']:>14,.0f}{r['alloc_cash']:>12,.0f}")
    lines.append(f"合计预计占用: {used:,.0f} / {ACCOUNT:,} 元 (剩余现金 {ACCOUNT-used:,.0f} 元)")
    if excluded:
        lines.append("")
        lines.append(f"-- 因真实股价超预算被资金约束排除的高分标的 ({len(excluded)} 只, 按分数降序) --")
        for r in excluded:
            lines.append(f"{r['instrument']:<10} score={r['score']:.4f} close={r['close']:>9.2f}  {r['reason']}")
    if no_quote:
        lines.append("")
        lines.append(f"-- 当日停牌/无报价的高分标的 ({len(no_quote)} 只) --")
        for r in no_quote:
            lines.append(f"{r['instrument']:<10} score={r['score']:.4f}")

    text = "\n".join(lines)
    print("\n" + text)
    out = REPORTS / "today_signal.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
