#!/usr/bin/env python
"""
每日自动模拟盘引擎（10 万元虚拟资金记账系统）。

流程（每个交易日收盘后运行）:
  1. 增量拉取最近 ~180 自然日日线(复用 fetch_data.py 的数据源链), 与 data/csv_raw
     存量合并——跨供应商后复权基数差异按最近重叠交易日等比对齐, 保证特征窗口连续;
  2. dump_update 增量写入 Qlib bin;
  3. 加载最新 FINISHED 模型, 全池打分(完整复现训练期处理器, 含全零特征自检);
  4. 对每个未处理交易日依次模拟成交: 以【T 日真实收盘价】成交(理想化假设,
     无滑点模型; 与回测 Ref(-2)/Ref(-1) 的信号-成交节奏一致);
  5. 逐日记账并输出今日盈亏报告。

执行口径:
  - 特征计算用后复权价(喂模型); 成交与记账一律用真实价 = $close / $factor;
  - 一手 = 100 股整数倍; 买 13bp / 卖 23bp / 单笔最低 5 元; risk_degree=0.95;
  - 目标组合 = 分数 Top8(剔除一手超预算者顺延补位); 持仓跌出分数 Top16 全部换出。

用法:
  python scripts/paper_trade.py --init                    # 初始化 ¥100,000 空仓账本
  python scripts/paper_trade.py                           # 处理所有未入账交易日(支持补跑)
  python scripts/paper_trade.py --status                  # 只打印现状, 不入账
  python scripts/paper_trade.py --replay 2025-07-01:2026-08-21   # 历史回放自检
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

PAPER_DIR = ROOT / "paper"
STATE_PATH = PAPER_DIR / "state.json"
EQUITY_PATH = PAPER_DIR / "equity.csv"
TRADES_PATH = PAPER_DIR / "trades.csv"

ACCOUNT = 100_000
TOPK = 8
DROP_GUARD = 16           # 跌出分数前 16 名的持仓换出
LOT = 100
RISK_DEGREE = 0.95
BUY_FEE_RATE, SELL_FEE_RATE, MIN_FEE = 0.0013, 0.0023, 5.0

EQUITY_COLS = ["date", "cash", "market_value", "total", "daily_pnl",
               "cum_pnl", "bench_close", "bench_cum_ret", "excess_cum"]
TRADES_COLS = ["date", "code", "direction", "shares", "price_real", "cost_fee"]

# --------------------------------------------------------------------------- #
# 策略注册表(t13 A/B 扩展点): name -> factory() -> score(end_day)->Series[instrument]
# 新策略只需在此登记一个工厂函数, 引擎其余部分(记账/换仓/幂等)完全复用。
# --------------------------------------------------------------------------- #
STRATEGIES: dict[str, callable] = {}


def register_strategy(name: str):
    def deco(fn):
        STRATEGIES[name] = fn
        return fn
    return deco


@register_strategy("ml_top8")          # 当前唯一实现: Alpha158+LGBM 打分(与回测同口径)
def _ml_strategy_factory():
    return load_model()                # 返回 score(end_day) -> Series




# --------------------------------------------------------------------------- #
# 账本基础操作
# --------------------------------------------------------------------------- #
def load_state() -> dict:
    if not STATE_PATH.exists():
        sys.exit("账本不存在, 请先执行: python scripts/paper_trade.py --init")
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    PAPER_DIR.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def overwrite_ledger_day(path: Path, columns: list[str], date_key: str,
                         rows: list[dict]) -> None:
    """幂等写账: 删除该日期已有行后追加新行(同日重跑绝不重复入账)。"""
    PAPER_DIR.mkdir(exist_ok=True)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)
    kept = old[old["date"].astype(str) != date_key]
    new = pd.DataFrame(rows, columns=columns)
    pd.concat([kept, new], ignore_index=True).to_csv(path, index=False)


def fmt_positions(state: dict, prices: dict[str, float]) -> str:
    lines = []
    total_mv = 0.0
    for code, pos in sorted(state["positions"].items()):
        px = prices.get(code, pos.get("last_price", pos["cost_real"]))
        mv = pos["shares"] * px
        total_mv += mv
        pnl = mv - pos["shares"] * pos["cost_real"]
        lines.append(f"  {code:<10} {pos['shares']:>6}股  现价{px:>8.2f}  "
                     f"市值{mv:>10,.0f}  浮动盈亏{pnl:>+9,.0f}")
    lines.append(f"  {'现金':<10} {'':>6}      {'':>10}  {state['cash']:>10,.0f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 数据接入: 增量抓取 + 存量合并 + dump_update
# --------------------------------------------------------------------------- #
def fetch_and_merge_csv(days: int = 180) -> tuple[int, str | None]:
    """抓取最近 days 自然日并与 data/csv_raw 合并。返回(新增bar数, 最新日期)。"""
    import concurrent.futures as cf
    from fetch_data import fetch_index, fetch_one  # noqa: PLC0415  复用既有数据源链

    raw_dir = ROOT / "data" / "csv_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    syms = [ln.split("\t")[0].strip() for ln in
            (ROOT / "data/qlib_data/cn_data/instruments/csi300.txt")
            .read_text().splitlines() if ln.strip()]
    codes = [s[2:] for s in syms]
    end = datetime.now()
    start = end - timedelta(days=days)
    s, e = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

    def _merge(symbol: str, new: pd.DataFrame) -> int:
        """新行并入存量 CSV。跨供应商 hfq 基数差异: 以最近共同交易日的收盘比
        等比缩放新增行的价格与因子(真实价=close/factor 在缩放下不变)。"""
        fp = raw_dir / f"{symbol}.csv"
        new = new.sort_index()
        if not fp.exists():
            new.to_csv(fp)
            return len(new)
        old = pd.read_csv(fp, index_col="date", parse_dates=["date"]).sort_index()
        common = new.index.intersection(old.index)
        appended = new.loc[new.index > old.index.max()]
        if common.empty:
            if appended.empty:
                return 0
            print(f"    [warn] {symbol} 无重叠交易日, 新行未对齐直接追加")
        elif not appended.empty:
            t0 = common.max()
            scale = float(old.loc[t0, "close"]) / float(new.loc[t0, "close"])
            if abs(scale - 1.0) > 1e-6:
                appended = appended.copy()
                for col in ("open", "high", "low", "close", "vwap", "factor"):
                    appended[col] = appended[col] * scale
        if appended.empty:
            return 0
        merged = pd.concat([old, appended]).sort_index()
        merged.index.name = "date"
        merged.to_csv(fp)
        return len(appended)

    added, newest = 0, None
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(fetch_one, c, s, e): c for c in codes}
        futs[pool.submit(fetch_index, s, e)] = "IDX"
        for i, fut in enumerate(cf.as_completed(futs), 1):
            try:
                if futs[fut] == "IDX":
                    sym, df = "SH000300", fut.result()
                else:
                    sym, df = fut.result()
                n = _merge(sym, df)
                added += n
                if n and (newest is None or str(df.index.max()) > str(newest)):
                    newest = str(df.index.max())
            except Exception as exc:  # noqa: BLE001
                print(f"    [skip] {futs[fut]}: {str(exc)[:90]}")
            if i % 60 == 0 or i == len(futs):
                print(f"    抓取进度 {i}/{len(futs)}  新增bar={added}")
    return added, newest


def refresh_instruments_end() -> int:
    """dump_update 只追加 bin 不延长 instruments 的 end 日期, 而 D.features 会按
    instruments 范围裁剪 —— 不刷新则新交易日"数据在但读不到"。按每个标的 CSV 的
    实际最后日期延长 all.txt / csi300.txt 的 end 列(只延不缩)。返回更新的行数。"""
    raw_dir = ROOT / "data/csv_raw"
    inst_dir = ROOT / "data/qlib_data/cn_data/instruments"
    updated = 0
    for name in ("all.txt", "csi300.txt"):
        fp = inst_dir / name
        lines_out = []
        for ln in fp.read_text().splitlines():
            parts = ln.split("\t")
            if len(parts) != 3:
                lines_out.append(ln)
                continue
            sym, start, end = parts[0], parts[1], pd.Timestamp(parts[2])
            csv_fp = raw_dir / f"{sym}.csv"
            if not csv_fp.exists():
                lines_out.append(ln)
                continue
            last = pd.read_csv(csv_fp, usecols=["date"]).tail(1)["date"].iloc[0]
            last = pd.Timestamp(last)
            if last > end:
                lines_out.append(f"{sym}\t{start}\t{last.date()}")
                updated += 1
            else:
                lines_out.append(ln)
        fp.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return updated


def dump_update_bins() -> None:
    cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/dump_bin.py"), "dump_update",
           "--data_path", "data/csv_raw",
           "--qlib_dir", "data/qlib_data/cn_data",
           "--include_fields", "open,close,high,low,volume,factor,vwap",
           "--exclude_fields", "symbol"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"dump_update 失败:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    print("    Qlib bin 增量更新完成")


# --------------------------------------------------------------------------- #
# 模型加载与打分(复用 today_signal 已验证的全套防御逻辑)
# --------------------------------------------------------------------------- #
def load_model():
    import qlib  # noqa: PLC0415

    qlib.init(provider_uri=str(ROOT / "data" / "qlib_data" / "cn_data"), region="cn")
    from qlib.utils import init_instance_by_config  # noqa: PLC0415
    from qlib.workflow import R  # noqa: PLC0415

    def _pick(exp_name):
        recs = R.get_exp(experiment_name=exp_name).list_recorders()
        fin = [r for r in recs.values() if "FINISHED" in str(getattr(r, "status", "")).upper()]
        pool = fin or list(recs.values())
        return (max(pool, key=lambda r: str(getattr(r, "start_time", "") or ""))
                if pool else None), len(fin), len(recs)

    rec, nf, na = _pick("workflow_10w")
    used = "workflow_10w"
    if rec is None:
        rec, nf, na = _pick("workflow")
        used = "workflow"
    if rec is None:
        sys.exit("未找到任何 recorder, 请先运行 make train10w")
    print(f"[model] 实验 {used} recorder {rec.id[:8]} (FINISHED {nf}/{na})")
    model = rec.load_object("params.pkl")

    def score(end_day: pd.Timestamp) -> pd.Series:
        from qlib.contrib.data.handler import Alpha158  # noqa: PLC0415
        from qlib.data import D  # noqa: PLC0415

        handler = init_instance_by_config({
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": "2020-09-01",          # 必须覆盖回看窗与 fit 窗
                "end_time": end_day.strftime("%Y-%m-%d"),
                "instruments": "csi300",
                "fit_start_time": "2021-01-01",
                "fit_end_time": "2024-12-31",
                "infer_processors": [
                    {"class": "RobustZScoreNorm",
                     "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],
            },
        })
        feats = handler.fetch(col_set="feature")
        # 全零特征自检(RobustZScoreNorm fit 窗口未被覆盖时会静默退化)
        import numpy as np  # noqa: PLC0415

        procs = list(getattr(handler, "infer_processors", []) or [])
        rz = next((p for p in procs if type(p).__name__ == "RobustZScoreNorm"), None)
        if rz is not None:
            mtn = getattr(rz, "mean_train", None)
            assert mtn is None or np.isfinite(mtn).all(), \
                "RobustZScoreNorm fit 窗口未被数据覆盖(start_time 太晚)"
        pred = pd.Series(model.model.predict(feats.values), index=feats.index)
        return pred.sort_index()

    return score


def real_close_matrix(symbols: list[str], start: pd.Timestamp,
                      end: pd.Timestamp) -> pd.DataFrame:
    """真实价矩阵 = $close/$factor, 行=交易日 列=标的。"""
    from qlib.data import D  # noqa: PLC0415

    px = D.features(symbols, ["$close", "$factor"],
                    start_time=start, end_time=end, freq="day")
    real = (px["$close"] / px["$factor"]).unstack(level="instrument")  # dt×inst
    return real.sort_index()


# --------------------------------------------------------------------------- #
# 信号 → 目标持仓 → 模拟成交
# --------------------------------------------------------------------------- #
def fee(direction: str, amount: float) -> float:
    rate = BUY_FEE_RATE if direction == "BUY" else SELL_FEE_RATE
    return max(amount * rate, MIN_FEE)


def execute_day(day: pd.Timestamp, scores: pd.Series, real_px: pd.Series,
                bench_real: float, state: dict, bench_base: float,
                account: float = ACCOUNT):
    """单日: 打分→目标组合→按真实收盘价成交→返回(新状态, trades, equity行)。"""
    positions = {k: dict(v) for k, v in state["positions"].items()}
    cash = float(state["cash"])

    def mark_price(code: str) -> float:
        px = real_px.get(code, np.nan)
        if px == px:
            return float(px)
        return float(positions.get(code, {}).get("last_price",
                   positions.get(code, {}).get("cost_real", 0.0))) if code in positions else float("nan")

    # 1) 标记总资产(成交前)
    for code, pos in positions.items():
        px = mark_price(code)
        pos["last_price"] = px if px == px else pos.get("last_price", pos["cost_real"])
    market_value = sum(p["shares"] * p["last_price"] for p in positions.values())
    total_before = cash + market_value

    # 2) 目标组合: 分数降序取可负担的 TOPK 只(一手超预算者记录后顺延)
    desired, seen_unaffordable = [], []
    for inst, sc in scores.sort_values(ascending=False).items():
        px = float(real_px.get(inst, np.nan))
        if px != px:
            continue                                    # 停牌不进目标池
        if px * LOT > account / TOPK:
            if len(seen_unaffordable) < TOPK:
                seen_unaffordable.append(inst)
            continue
        desired.append((inst, float(sc)))
        if len(desired) >= TOPK:
            break
    guard_set = {inst for inst, _ in
                 list(scores.sort_values(ascending=False).items())[:DROP_GUARD]}

    trades = []
    # 3) 换出: 跌出 Top16 的持仓全部卖出
    for code in list(positions):
        if code in guard_set:
            continue
        pos = positions.pop(code)
        px = pos["last_price"]
        if px <= 0:
            positions[code] = pos                        # 无有效价格, 无法成交
            continue
        amount = pos["shares"] * px
        f = fee("SELL", amount)
        cash += amount - f
        trades.append({"date": day.date().isoformat(), "code": code, "direction": "SELL",
                       "shares": pos["shares"], "price_real": round(px, 4),
                       "cost_fee": round(f, 2)})
        print(f"    [SELL] {code} {pos['shares']}股 @{px:.2f} (跌出Top{DROP_GUARD})")

    # 4) 建仓: 目标市值 = 总资产×risk_degree/TOPK, 只对【新进目标组合】的标的买入;
       # 已持有的标的保持不动(与 TopkDropout 语义一致, 避免每日再平衡的小额订单
    # 被最低佣金持续侵蚀——replay 实测该 churn 单独造成约 20pp 收益差)
    total_after_sell = cash + sum(p["shares"] * p["last_price"] for p in positions.values())
    slot_target = total_after_sell * RISK_DEGREE / TOPK
    for inst, _sc in desired:
        if inst in positions:                            # 已持有: 继续持有, 不做微调
            continue
        px = float(real_px.get(inst, np.nan))
        if px != px:
            continue
        shares = int(slot_target / (px * LOT)) * LOT     # 向下取整到整手
        while shares > 0:
            amount = shares * px
            f = fee("BUY", amount)
            if amount + f <= cash + 1e-6:
                break
            shares -= LOT                                # 现金不足则减一手重试
        if shares <= 0:
            print(f"    [SKIP-BUY] {inst} 现金不足")
            continue
        f = fee("BUY", shares * px)
        cash -= shares * px + f
        positions[inst] = {"shares": shares, "cost_real": px, "last_price": px}
        trades.append({"date": day.date().isoformat(), "code": inst, "direction": "BUY",
                       "shares": shares, "price_real": round(px, 4),
                       "cost_fee": round(f, 2)})
        print(f"    [BUY ] {inst} {shares}股 @{px:.2f}")

    # 5) 收盘记账
    market_value = sum(p["shares"] * p["last_price"] for p in positions.values())
    total = cash + market_value
    daily_pnl = total - total_before
    equity_row = {"date": day.date().isoformat(), "cash": round(cash, 2),
                  "market_value": round(market_value, 2), "total": round(total, 2),
                  "daily_pnl": round(daily_pnl, 2),
                  "cum_pnl": round(total - account, 2),
                  "bench_close": round(bench_real, 4),
                  "bench_cum_ret": round(bench_real / bench_base - 1, 6),
                  "excess_cum": round((total / account - 1)
                                      - (bench_real / bench_base - 1), 6)}
    new_state = {"cash": cash,
                 "positions": {c: {"shares": p["shares"],
                                   "cost_real": round(p["cost_real"], 4),
                                   "last_price": round(p["last_price"], 4)}
                               for c, p in positions.items()},
                 "last_processed_date": day.date().isoformat()}
    return new_state, trades, equity_row


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def cmd_init(account: float = ACCOUNT, strategy: str = "ml_top8") -> None:
    PAPER_DIR.mkdir(exist_ok=True)
    save_state({"cash": float(account), "account": account, "strategy": strategy,
                "positions": {}, "last_processed_date": None, "prev_snapshot": None})
    for f in (EQUITY_PATH, TRADES_PATH):
        if f.exists():
            f.unlink()
    print(f"账本已初始化: 现金 ¥{account:,}, 空仓。状态文件: {STATE_PATH}")


def cmd_status() -> None:
    st = load_state()
    eq = pd.read_csv(EQUITY_PATH) if EQUITY_PATH.exists() else pd.DataFrame()
    print("=" * 62)
    print(f"模拟盘现状  (最后入账交易日: {st.get('last_processed_date')})")
    print("=" * 62)
    prices = {c: p.get("last_price", p["cost_real"])
              for c, p in st["positions"].items()}
    print(fmt_positions(st, prices))
    mv = sum(p["shares"] * prices.get(c, p["cost_real"])
             for c, p in st["positions"].items())
    print(f"  总资产 ≈ ¥{st['cash'] + mv:,.0f}")
    if not eq.empty:
        last = eq.iloc[-1]
        print(f"\n最近一条权益记录: {last['date']}  total={last['total']:,.0f}  "
              f"当日{last['daily_pnl']:+,.0f}  累计{last['cum_pnl']:+,.0f}  "
              f"超额(vs CSI300){last['excess_cum']:+.2%}")


def collect_days(score, days: list[pd.Timestamp], symbols: list[str]):
    """一次性算到最大日期再逐日切片(Alpha158 每行只依赖 ≤T 历史, 无未来泄漏)。"""
    end_max = max(days)
    pred = score(end_max)
    out = {}
    for d in days:
        try:
            out[d] = pred.xs(d, level=0)                # Series(index=instrument)
        except KeyError:
            out[d] = pd.Series(dtype=float)
    return out


def run_daily(args) -> None:
    state = load_state()
    print("[1/5] 增量抓取最近行情并与 csv_raw 合并...")
    added, newest_csv = fetch_and_merge_csv(days=180)

    # 【顺序关键】先落盘(dump)再读日历: 否则本轮新抓到的交易日会被下一轮才认领
    if added > 0:
        print("[2/5] Qlib bin 增量更新...")
        dump_update_bins()
        n = refresh_instruments_end()
        print(f"    instruments end 刷新 {n} 行")
    else:
        print("[2/5] 无新增 bar, 跳过 dump_update")

    cal = [pd.Timestamp(x) for x in
           (ROOT / "data/qlib_data/cn_data/calendars/day.txt").read_text().splitlines()]
    last_proc = (pd.Timestamp(state["last_processed_date"])
                 if state.get("last_processed_date") else None)
    cal_last = cal[-1]

    if last_proc is None:
        # 新账本首次运行: 只入账最新交易日(不回补历史——历史表现请用 --replay 验证)
        pending = [cal_last]
        print(f"[info] 新账本首次运行, 从最新交易日 {cal_last.date()} 开始记账")
    else:
        pending = [d for d in cal if d > last_proc]
    redo_day = None
    if not pending and last_proc is not None and cal_last == last_proc \
            and state.get("prev_snapshot"):
        redo_day = last_proc                              # 同日重跑: 回滚当日再重做
    if not pending and redo_day is None:
        print(f"今日休市(无新交易日)。最新交易日 {cal_last.date()} 已于 "
              f"{last_proc.date() if last_proc else '-'} 入账。")
        return

    if added == 0 and redo_day is None:
        print("[warn] 本次未抓到任何新 bar, 仍将尝试用现有数据处理待处理交易日。")

    print(f"[3/5] 策略打分 (strategy={state.get('strategy', 'ml_top8')})...")
    days = pending or ([redo_day] if redo_day else [])
    strat_name = state.get("strategy", "ml_top8")
    if strat_name not in STRATEGIES:
        sys.exit(f"未知策略 {strat_name!r}, 可选: {', '.join(STRATEGIES)}")
    score = STRATEGIES[strat_name]()
    scores_by_day = collect_days(score, days, [])

    px_all = real_close_matrix(
        [ln.split("\t")[0].strip() for ln in
         (ROOT / "data/qlib_data/cn_data/instruments/csi300.txt").read_text().splitlines()
         if ln.strip()],
        min(days), max(days))

    print("[4/5] 逐日模拟成交...")
    if redo_day is not None:                             # 回滚到前一交易日收盘状态
        prev = state["prev_snapshot"]
        state = {"cash": prev["cash"],
                 "positions": prev["positions"],
                 "last_processed_date": prev["last_processed_date"],
                 "prev_snapshot": None}
        days = [redo_day]
        print(f"    检测到同日重跑, 回滚 {redo_day.date()} 当日记录后重新处理")

    bench_real_all = real_close_matrix(["SH000300"], min(days), max(days))["SH000300"]
    bench_base = None
    eq_path, tr_path = EQUITY_PATH, TRADES_PATH
    if EQUITY_PATH.exists():
        old_eq = pd.read_csv(EQUITY_PATH)
        old_eq = old_eq[~old_eq["date"].isin([d.date().isoformat() for d in days])]
        if not old_eq.empty:
            bench_base = float(old_eq["bench_close"].iloc[0])
    if bench_base is None:
        bench_base = float(bench_real_all.iloc[0])

    for d in days:
        sc = scores_by_day.get(d)
        if sc is None or sc.empty:
            print(f"    [skip] {d.date()} 无打分结果(疑似停牌日历异常)")
            continue
        real_px = px_all.loc[d] if d in px_all.index else pd.Series(dtype=float)
        bench_real = float(bench_real_all.get(d, np.nan))
        state_prev_of_day = {"cash": state["cash"],
                             "positions": {k: dict(v) for k, v in state["positions"].items()},
                             "last_processed_date": state["last_processed_date"]}
        new_state, trades, eq_row = execute_day(
            d, sc, real_px, bench_real, state, bench_base,
            account=float(state.get("account", ACCOUNT)))   # t11-A: 账户参数必须透传
        overwrite_ledger_day(eq_path, EQUITY_COLS, d.date().isoformat(), [eq_row])
        overwrite_ledger_day(tr_path, TRADES_COLS, d.date().isoformat(), trades)
        state = dict(new_state, prev_snapshot=state_prev_of_day)
        save_state(state)

    print("[5/5] 完成。今日报告:")
    last = pd.read_csv(EQUITY_PATH).iloc[-1]
    print("=" * 64)
    acct0 = float(state.get("account", ACCOUNT))
    print(f"交易日 {last['date']}:  总资产 ¥{last['total']:,.0f}"
          f"   今日盈亏 {last['daily_pnl']:+,.0f}"
          f"   累计盈亏 {last['cum_pnl']:+,.0f}")
    print(f"同期沪深300 {last['bench_cum_ret']:+.2%}   "
          f"策略累计超额 {last['excess_cum']:+.2%}")
    prices = {c: p.get("last_price", p["cost_real"]) for c, p in state["positions"].items()}
    print("-" * 64)
    print(fmt_positions(state, prices))
    print("=" * 64)


def run_replay(range_str: str, account: float = ACCOUNT,
               strategy: str = "ml_top8") -> None:
    if ":" not in range_str:
        sys.exit("--replay 格式: START:END, 例 2025-07-01:2026-08-21")
    s, e = (pd.Timestamp(x) for x in range_str.split(":"))
    import qlib  # noqa: PLC0415

    qlib.init(provider_uri=str(ROOT / "data" / "qlib_data" / "cn_data"), region="cn")
    from qlib.data import D  # noqa: PLC0415

    cal = [pd.Timestamp(x) for x in D.calendar(freq="day")]
    days = [d for d in cal if s <= d <= e]
    if not days:
        sys.exit("回放区间内无交易日")
    print(f"[replay] 区间 {days[0].date()} ~ {days[-1].date()} 共 {len(days)} 个交易日")

    print(f"[replay] 策略打分 (strategy={strategy})...")
    if strategy not in STRATEGIES:
        sys.exit(f"未知策略 {strategy!r}, 可选: {', '.join(STRATEGIES)}")
    score = STRATEGIES[strategy]()
    scores_by_day = collect_days(score, days, [])
    syms = [ln.split("\t")[0].strip() for ln in
            (ROOT / "data/qlib_data/cn_data/instruments/csi300.txt")
            .read_text().splitlines() if ln.strip()]
    px_all = real_close_matrix(syms, days[0], days[-1])
    bench_real = real_close_matrix(["SH000300"], days[0], days[-1])["SH000300"]

    state = {"cash": float(account), "account": account,
             "positions": {}, "last_processed_date": None, "prev_snapshot": None}
    bench_base = float(bench_real.iloc[0])
    eq_rows, tr_rows = [], []
    for i, d in enumerate(days, 1):
        sc = scores_by_day.get(d)
        if sc is None or sc.empty:
            print(f"    [skip] {d.date()} 无打分")
            continue
        new_state, trades, eq_row = execute_day(
            d, sc, px_all.loc[d], float(bench_real.get(d, np.nan)), state, bench_base,
            account=float(account))          # t14-A 补遗: replay 路径同样透传账户
        eq_rows.append(eq_row)
        tr_rows.extend(trades)
        state = new_state
        if i % 40 == 0 or i == len(days):
            print(f"    进度 {i}/{len(days)}  total={eq_row['total']:,.0f}")

    eq_df = pd.DataFrame(eq_rows, columns=EQUITY_COLS)
    PAPER_DIR.mkdir(exist_ok=True)
    eq_df.to_csv(PAPER_DIR / "replay_equity.csv", index=False)
    pd.DataFrame(tr_rows, columns=TRADES_COLS).to_csv(
        PAPER_DIR / "replay_trades.csv", index=False)

    totals = [account] + eq_df["total"].tolist()
    peak, mdd = -np.inf, 0.0
    for v in totals:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    cum_ret = eq_df["total"].iloc[-1] / account - 1
    bench_cum = eq_df["bench_cum_ret"].iloc[-1]
    neg_cash_days = int((eq_df["cash"] < -1e-6).sum())

    print("\n" + "=" * 64)
    print(f"[replay 自检结果] {days[0].date()} ~ {days[-1].date()}")
    print(f"  策略累计收益 : {cum_ret:+.2%}")
    print(f"  最大回撤     : {mdd:.2%}")
    print(f"  同期沪深300  : {bench_cum:+.2%}")
    print(f"  超额         : {cum_ret - bench_cum:+.2%}")
    print(f"  订单笔数     : {len(tr_rows)}  ({len(tr_rows)/len(days):.1f} 笔/日)")
    print(f"  成本合计     : ¥{sum(t['cost_fee'] for t in tr_rows):,.0f}")
    print(f"  负现金天数   : {neg_cash_days} (应为 0)")
    print(f"  明细已写     : paper/replay_equity.csv / paper/replay_trades.csv (非正式账本)")
    print("=" * 64)


def main() -> None:
    ap = argparse.ArgumentParser(description="10 万元每日自动模拟盘引擎")
    ap.add_argument("--init", action="store_true", help="初始化账本(¥100,000 空仓)")
    ap.add_argument("--status", action="store_true", help="只打印现状不入账")
    ap.add_argument("--replay", metavar="START:END", help="历史区间回放自检(不落正式账本)")
    ap.add_argument("--strategy", default="ml_top8",
                    help=f"策略名, 可选: {', '.join(STRATEGIES)} (t13 将新增红利低波质量等)")
    ap.add_argument("--account", type=int, default=ACCOUNT,
                    help="账户初始资金(--init 时写入账本; replay 用作回放基数)")
    args = ap.parse_args()

    if args.init:
        cmd_init(account=args.account, strategy=args.strategy)
    elif args.status:
        cmd_status()
    elif args.replay:
        run_replay(args.replay, account=args.account, strategy=args.strategy)
    else:
        run_daily(None)


if __name__ == "__main__":
    main()
