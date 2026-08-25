#!/usr/bin/env python
"""
每日自动模拟盘引擎(t13 起 A/B 双账户: ml_top8 红利低波 dq_dvpb)。

流程(每个交易日收盘后运行):
  1. 增量拉取最近 ~180 自然日日线(复用 fetch_data.py 的数据源链), 与 csv_raw 合并;
  2. dump_update 增量写入 Qlib bin + instruments end 刷新;
  3. 对每个注册策略账户独立打分(ml=Alpha158+LGBM; dq=红利低波质量因子);
  4. 按各账户调度语义模拟成交(ml=每日; dq=每月首个交易日+MA200 闸门),
     以【T 日真实收盘价】成交(理想化假设, 无滑点);
  5. 双账本逐日记账并输出各自"今天赚没赚"报告。

执行口径:
  - 特征计算用后复权价(喂模型)/基本面因子用真实价; 成交与记账一律真实价=$close/$factor;
  - 一手=100股整数倍; 买13bp/卖23bp/单笔最低5元; risk_degree=0.95;
  - ml 口径自 t10 起一字不动(Top8进/跌出Top16出/持有不再平衡);
  - 幂等: 同日重跑先回滚当日再重做; 断档自动补跑。

用法:
  python scripts/paper_trade.py --init                     # 初始化全部策略账户
  python scripts/paper_trade.py                            # 处理所有未入账交易日
  python scripts/paper_trade.py --status                   # 只打印现状不入账
  python scripts/paper_trade.py --replay 2025-07-01:2026-08-21     # 双账户对比回放
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

from strategy_registry import POLICIES, STRATEGIES, register_strategy, short_tag  # noqa: E402

# 策略模块副作用注册: 新策略落成 scripts/strategy_*.py 并在此追加一行即可
import strategy_dq  # noqa: F401,E402  (注册 dq_dvpb)

PAPER_DIR = ROOT / "paper"
ACCOUNT = 100_000
TOPK = 8
DROP_GUARD = 16           # 跌出分数前 16 名的持仓换出
LOT = 100
RISK_DEGREE = 0.95
BUY_FEE_RATE, SELL_FEE_RATE, MIN_FEE = 0.0013, 0.0023, 5.0
MA200_WINDOW = 200
GATE_CUT = 0.5            # MA200 闸门触发后的仓位系数(股票仓位减半)

EQUITY_COLS = ["date", "cash", "market_value", "total", "daily_pnl",
               "cum_pnl", "bench_close", "bench_cum_ret", "excess_cum"]
TRADES_COLS = ["date", "code", "direction", "shares", "price_real", "cost_fee"]


def account_paths(strategy: str) -> dict[str, Path]:
    """每策略账户独立账本文件。ml 迁移后为 *_ml.*; dq 为 *_dq.*。"""
    tag = short_tag(strategy)
    return {"state": PAPER_DIR / f"state_{tag}.json",
            "equity": PAPER_DIR / f"equity_{tag}.csv",
            "trades": PAPER_DIR / f"trades_{tag}.csv"}


# --------------------------------------------------------------------------- #
# 打分工厂: ml_top8(Alpha158+LGBM, 与回测同口径)
# --------------------------------------------------------------------------- #
@register_strategy("ml_top8")
def _ml_strategy_factory():
    return load_model()                # 返回 score(end_day) -> Series


# --------------------------------------------------------------------------- #
# 账本基础操作
# --------------------------------------------------------------------------- #
def migrate_legacy_ledgers() -> None:
    """t13 前的单账户文件(state.json/equity.csv/trades.csv)一次性迁移为 *_ml。"""
    pairs = [("state.json", "state_ml.json"), ("equity.csv", "equity_ml.csv"),
             ("trades.csv", "trades_ml.csv"), ("replay_equity.csv", "replay_equity_ml.csv"),
             ("replay_trades.csv", "replay_trades_ml.csv")]
    moved = []
    for old, new in pairs:
        fo, fn = PAPER_DIR / old, PAPER_DIR / new
        if fo.exists() and not fn.exists():
            fo.rename(fn)
            moved.append(new)
    if moved:
        print(f"[migrate] 单账户账本已迁移: {', '.join(moved)}")


def load_state(strategy: str) -> dict:
    fp = account_paths(strategy)["state"]
    if not fp.exists():
        sys.exit(f"账本不存在({fp.name}), 请先执行: python scripts/paper_trade.py --init")
    return json.loads(fp.read_text(encoding="utf-8"))


def save_state(strategy: str, state: dict) -> None:
    PAPER_DIR.mkdir(exist_ok=True)
    fp = account_paths(strategy)["state"]
    fp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


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
    for code, pos in sorted(state["positions"].items()):
        px = prices.get(code, pos.get("last_price", pos["cost_real"]))
        mv = pos["shares"] * px
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
# ml 打分(复用 today_signal 已验证的全套防御逻辑)
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


def month_first_days(all_days: list[pd.Timestamp]) -> set[pd.Timestamp]:
    """每个自然月的第一个交易日集合(月度调仓日)。"""
    seen, out = set(), set()
    for d in all_days:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            out.add(d)
    return out


def ma200_gate_series(bench_close: pd.Series) -> pd.Series:
    """MA200 闸门: 收盘≥200日线 → 1.0(满仓系数); <200日线 → GATE_CUT(减半)。
    均线预热期(<MA200_WINDOW)视为满仓系数 1.0。"""
    ma = bench_close.rolling(MA200_WINDOW).mean()
    gate = np.where(bench_close.values >= ma.values, 1.0, GATE_CUT)
    gate[np.isnan(ma.values)] = 1.0
    return pd.Series(gate, index=bench_close.index)


def execute_day(day: pd.Timestamp, scores: pd.Series, real_px: pd.Series,
                bench_real: float, state: dict, bench_base: float,
                account: float = ACCOUNT, allow_trade: bool = True,
                gate_mult: float = 1.0):
    """单日: 打分→目标组合→按真实收盘价成交→返回(新状态, trades, equity行)。
    allow_trade=False 时只做收盘记账不产生任何订单(月度调仓的非调仓日)。"""
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

    trades = []
    if allow_trade:
        # 2) 目标组合: 分数降序取可负担的 TOPK 只(一手超预算者记录后顺延)
        desired = []
        for inst, sc in scores.sort_values(ascending=False).items():
            px = float(real_px.get(inst, np.nan))
            if px != px:
                continue                                # 停牌不进目标池
            if px * LOT > account / TOPK:
                continue
            desired.append((inst, float(sc)))
            if len(desired) >= TOPK:
                break
        guard_set = {inst for inst, _ in
                     list(scores.sort_values(ascending=False).items())[:DROP_GUARD]}

        # 3) 换出: 跌出 Top16 的持仓全部卖出
        for code in list(positions):
            if code in guard_set:
                continue
            pos = positions.pop(code)
            px = pos["last_price"]
            if px <= 0:
                positions[code] = pos                    # 无有效价格, 无法成交
                continue
            amount = pos["shares"] * px
            f = fee("SELL", amount)
            cash += amount - f
            trades.append({"date": day.date().isoformat(), "code": code,
                           "direction": "SELL", "shares": pos["shares"],
                           "price_real": round(px, 4), "cost_fee": round(f, 2)})
            print(f"      [SELL] {code} {pos['shares']}股 @{px:.2f}")

        # 4) 建仓: 目标市值 = 总资产×risk_degree×gate/TOPK, 只买新进目标组合的标的;
        # 已持有的保持不动(避免小额再平衡订单被最低佣金持续侵蚀)
        total_after_sell = cash + sum(p["shares"] * p["last_price"]
                                      for p in positions.values())
        slot_target = total_after_sell * RISK_DEGREE * gate_mult / TOPK
        for inst, _sc in desired:
            if inst in positions:
                continue
            px = float(real_px.get(inst, np.nan))
            if px != px:
                continue
            shares = int(slot_target / (px * LOT)) * LOT
            while shares > 0:
                amount = shares * px
                f = fee("BUY", amount)
                if amount + f <= cash + 1e-6:
                    break
                shares -= LOT
            if shares <= 0:
                print(f"      [SKIP-BUY] {inst} 现金不足")
                continue
            f = fee("BUY", shares * px)
            cash -= shares * px + f
            positions[inst] = {"shares": shares, "cost_real": px, "last_price": px}
            trades.append({"date": day.date().isoformat(), "code": inst,
                           "direction": "BUY", "shares": shares,
                           "price_real": round(px, 4), "cost_fee": round(f, 2)})
            print(f"      [BUY ] {inst} {shares}股 @{px:.2f}"
                  + (f"  (gate×{gate_mult:.2f})" if gate_mult != 1.0 else ""))

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
def cmd_init(account: float = ACCOUNT, strategies: list[str] | None = None) -> None:
    PAPER_DIR.mkdir(exist_ok=True)
    for strat in (strategies or list(STRATEGIES)):
        save_state(strat, {"cash": float(account), "account": account,
                           "strategy": strat, "positions": {},
                           "last_processed_date": None, "prev_snapshot": None})
        p = account_paths(strat)
        for f in (p["equity"], p["trades"]):
            if f.exists():
                f.unlink()
        tag = short_tag(strat)
        print(f"账本[{tag}] 已初始化: 现金 ¥{account:,}, 空仓 ({p['state'].name})")


def cmd_status() -> None:
    for strat in STRATEGIES:
        tag = short_tag(strat)
        p = account_paths(strat)
        if not p["state"].exists():
            print(f"[{tag}] 未初始化 ({p['state'].name} 不存在)")
            continue
        st = json.loads(p["state"].read_text(encoding="utf-8"))
        eq = pd.read_csv(p["equity"]) if p["equity"].exists() else pd.DataFrame()
        print("=" * 62)
        print(f"账户[{tag}/{st.get('strategy')}]  最后入账: {st.get('last_processed_date')}"
              f"  资金 ¥{float(st.get('account', ACCOUNT)):,.0f}")
        prices = {c: q.get("last_price", q["cost_real"])
                  for c, q in st["positions"].items()}
        print(fmt_positions(st, prices))
        mv = sum(q["shares"] * prices.get(c, q["cost_real"])
                 for c, q in st["positions"].items())
        print(f"  总资产 ≈ ¥{st['cash'] + mv:,.0f}")
        if not eq.empty:
            last = eq.iloc[-1]
            print(f"  最近权益: {last['date']}  total={last['total']:,.0f}  "
                  f"当日{last['daily_pnl']:+,.0f}  累计{last['cum_pnl']:+,.0f}  "
                  f"超额{last['excess_cum']:+.2%}")
    print("=" * 62)


def collect_days(score, days: list[pd.Timestamp]):
    """一次性算到最大日期再逐日切片(各打分只依赖 ≤T 历史, 无未来泄漏)。"""
    pred = score(max(days))
    out = {}
    for d in days:
        try:
            out[d] = pred.xs(d, level=0)
        except KeyError:
            out[d] = pd.Series(dtype=float)
    return out


def _csi_symbols() -> list[str]:
    return [ln.split("\t")[0].strip() for ln in
            (ROOT / "data/qlib_data/cn_data/instruments/csi300.txt")
            .read_text().splitlines() if ln.strip()]


def run_daily(args) -> None:
    migrate_legacy_ledgers()
    print("[1/4] 增量抓取最近行情并与 csv_raw 合并...")
    added, _ = fetch_and_merge_csv(days=180)

    # 【顺序关键】先落盘(dump)再读日历: 否则本轮新抓到的交易日会被下一轮才认领
    if added > 0:
        print("[2/4] Qlib bin 增量更新...")
        dump_update_bins()
        print(f"    instruments end 刷新 {refresh_instruments_end()} 行")
    else:
        print("[2/4] 无新增 bar, 跳过 dump_update")

    cal = [pd.Timestamp(x) for x in
           (ROOT / "data/qlib_data/cn_data/calendars/day.txt").read_text().splitlines()]
    cal_last = cal[-1]
    mfirst = month_first_days(cal)

    for strat in STRATEGIES:
        pol = POLICIES.get(strat, {"cadence": "daily", "ma200_gate": False})
        tag = short_tag(strat)
        p = account_paths(strat)
        if not p["state"].exists():                      # 缺账本则静默初始化
            cmd_init(strategies=[strat])
        state = load_state(strat)

        last_proc = (pd.Timestamp(state["last_processed_date"])
                     if state.get("last_processed_date") else None)
        if last_proc is None:
            pending = [cal_last]
            print(f"\n[账户{tag}] 新账本首次运行, 从 {cal_last.date()} 开始记账")
        else:
            pending = [d for d in cal if d > last_proc]
        redo_day = None
        if not pending and last_proc is not None and cal_last == last_proc \
                and state.get("prev_snapshot"):
            redo_day = last_proc
        if not pending and redo_day is None:
            print(f"\n[账户{tag}] 今日休市或已处理({cal_last.date()}), 跳过")
            continue

        days = pending or ([redo_day])
        if redo_day is not None:
            prev = state["prev_snapshot"]
            state = {"cash": prev["cash"], "positions": prev["positions"],
                     "last_processed_date": prev["last_processed_date"],
                     "prev_snapshot": None}
            print(f"\n[账户{tag}] 同日重跑, 回滚 {redo_day.date()} 后重做")

        print(f"[3/4] [{tag}] 策略打分...")
        score = STRATEGIES[strat]()
        scores_by_day = collect_days(score, days)
        px_all = real_close_matrix(_csi_symbols(), min(days), cal_last)
        bench_all = real_close_matrix(["SH000300"],
                                      min(days) - pd.Timedelta(days=400),
                                      cal_last)["SH000300"]

        print(f"[4/4] [{tag}] 逐日模拟成交 (cadence={pol['cadence']})...")
        bench_base = None
        if p["equity"].exists():
            old_eq = pd.read_csv(p["equity"])
            old_eq = old_eq[~old_eq["date"].isin([d.date().isoformat() for d in days])]
            if not old_eq.empty:
                bench_base = float(old_eq["bench_close"].iloc[0])
        if bench_base is None:
            bench_base = float(bench_all.loc[min(days)])
        deployed = bool(state["positions"])   # 空仓新账户须先完成首次建仓

        for d in days:
            trade_due = ((pol["cadence"] == "daily") or (d in mfirst)
                         or not deployed)
            gate = 1.0
            if pol.get("ma200_gate"):
                gate = float(ma200_gate_series(bench_all.loc[:d]).iloc[-1])
            sc = scores_by_day.get(d)
            if (sc is None or sc.empty) and trade_due:
                print(f"    [{d.date()}] 调仓日无打分结果, 本日仅记账不开仓")
                trade_due = False
            real_px = px_all.loc[d] if d in px_all.index else pd.Series(dtype=float)
            state_prev_of_day = {"cash": state["cash"],
                                 "positions": {k: dict(v) for k, v in state["positions"].items()},
                                 "last_processed_date": state["last_processed_date"]}
            new_state, trades, eq_row = execute_day(
                d, sc if sc is not None and not sc.empty else pd.Series(dtype=float),
                real_px,
                float(bench_all.get(d, np.nan)), state, bench_base,
                account=float(state.get("account", ACCOUNT)),
                allow_trade=trade_due, gate_mult=gate)
            overwrite_ledger_day(p["equity"], EQUITY_COLS, d.date().isoformat(), [eq_row])
            overwrite_ledger_day(p["trades"], TRADES_COLS, d.date().isoformat(), trades)
            state = dict(new_state, prev_snapshot=state_prev_of_day)
            save_state(strat, state)
            if trade_due and (new_state["positions"] or trades):
                deployed = True
            acts = "调仓" if trade_due else "记账"
            print(f"    [{tag}] {d.date()} {acts} gate={gate:.2f} "
                  f"total={eq_row['total']:,.0f}")

    print("\n完成。双账户今日报告:")
    for strat in STRATEGIES:
        tag = short_tag(strat)
        p = account_paths(strat)
        if not p["equity"].exists():
            continue
        eq = pd.read_csv(p["equity"])
        if eq.empty:
            continue
        last = eq.iloc[-1]
        st = json.loads(p["state"].read_text(encoding="utf-8"))
        print("=" * 64)
        print(f"[{tag}] 交易日 {last['date']}: 总资产 ¥{last['total']:,.0f}"
              f"   今日 {last['daily_pnl']:+,.0f}   累计 {last['cum_pnl']:+,.0f}"
              f"   超额(vs CSI300) {last['excess_cum']:+.2%}")
        prices = {c: q.get("last_price", q["cost_real"])
                  for c, q in st["positions"].items()}
        print(fmt_positions(st, prices))
    print("=" * 64)


def _simulate_replay(name: str, days: list[pd.Timestamp], px_all: pd.DataFrame,
                     bench_all: pd.Series, account: float) -> tuple[pd.DataFrame, list]:
    """单策略回放模拟: 返回(权益df, 成交rows)。月度策略首日强制建仓。"""
    pol = POLICIES.get(name, {"cadence": "daily", "ma200_gate": False})
    tag = short_tag(name)
    score = STRATEGIES[name]()
    scores_by_day = collect_days(score, days)
    mfirst = month_first_days(days)
    gate_all = ma200_gate_series(bench_all) if pol.get("ma200_gate") \
        else pd.Series(1.0, index=bench_all.index)
    bench_base = float(bench_all.loc[days[0]])

    state = {"cash": float(account), "positions": {}, "last_processed_date": None}
    eq_rows, tr_rows = [], []
    deployed = False
    for i, d in enumerate(days, 1):
        trade_due = (pol["cadence"] == "daily") or (not deployed) or (d in mfirst)
        gate = float(gate_all.loc[d])
        sc = scores_by_day.get(d)
        if (sc is None or sc.empty) and trade_due:
            trade_due = False
        new_state, trades, eq_row = execute_day(
            d, sc if sc is not None and not sc.empty else pd.Series(dtype=float),
            px_all.loc[d],
            float(bench_all.get(d, np.nan)), state, bench_base,
            account=float(account), allow_trade=trade_due, gate_mult=gate)
        if trade_due and (new_state["positions"] or tr_rows):
            deployed = True
        eq_rows.append(eq_row)
        tr_rows.extend(trades)
        state = new_state
        if i % 60 == 0 or i == len(days):
            print(f"    [{tag}] 进度 {i}/{len(days)}  total={eq_row['total']:,.0f}")
    return pd.DataFrame(eq_rows, columns=EQUITY_COLS), tr_rows


def run_replay(range_str: str, account: float = ACCOUNT,
               strategies: list[str] | None = None) -> None:
    if ":" not in range_str:
        sys.exit("--replay 格式: START:END, 例 2025-07-01:2026-08-21")
    s, e = (pd.Timestamp(x) for x in range_str.split(":"))
    names = strategies or list(STRATEGIES)
    for n in names:
        if n not in STRATEGIES:
            sys.exit(f"未知策略 {n!r}, 可选: {', '.join(STRATEGIES)}")

    import qlib  # noqa: PLC0415

    qlib.init(provider_uri=str(ROOT / "data" / "qlib_data" / "cn_data"), region="cn")

    from qlib.data import D  # noqa: PLC0415

    cal = [pd.Timestamp(x) for x in D.calendar(freq="day")]
    days = [d for d in cal if s <= d <= e]
    if not days:
        sys.exit("回放区间内无交易日")
    print(f"[replay] 区间 {days[0].date()} ~ {days[-1].date()} 共 {len(days)} 个交易日"
          f" | 账户 ¥{account:,} | 策略 {','.join(short_tag(n) for n in names)}")

    px_all = real_close_matrix(_csi_symbols(), days[0], days[-1])
    bench_all = real_close_matrix(["SH000300"],
                                  days[0] - pd.Timedelta(days=400),
                                  days[-1])["SH000300"]

    PAPER_DIR.mkdir(exist_ok=True)
    results = {}
    for name in names:
        tag = short_tag(name)
        print(f"[replay][{tag}] 模拟中...")
        eq_df, tr_rows = _simulate_replay(name, days, px_all, bench_all, account)
        eq_df.to_csv(PAPER_DIR / f"replay_equity_{tag}.csv", index=False)
        pd.DataFrame(tr_rows, columns=TRADES_COLS).to_csv(
            PAPER_DIR / f"replay_trades_{tag}.csv", index=False)

        totals = [account] + eq_df["total"].tolist()
        peak, mdd = -np.inf, 0.0
        for v in totals:
            peak = max(peak, v)
            mdd = min(mdd, v / peak - 1)
        results[tag] = {
            "cum": eq_df["total"].iloc[-1] / account - 1,
            "mdd": mdd,
            "bench": eq_df["bench_cum_ret"].iloc[-1],
            "orders": len(tr_rows),
            "fees": sum(t["cost_fee"] for t in tr_rows),
            "neg_cash": int((eq_df["cash"] < -1e-6).sum()),
            "file": f"paper/replay_equity_{tag}.csv",
        }

    print("\n" + "=" * 78)
    print(f"[replay 双账户对比] {days[0].date()} ~ {days[-1].date()}  基准账户 ¥{account:,}")
    print(f"  {'账户':<6}{'累计收益':>10}{'最大回撤':>10}{'超额':>10}"
          f"{'笔/日':>8}{'成本¥':>10}{'负现金天':>9}")
    bench_cum = next(iter(results.values()))["bench"]
    print(f"  {'基准':<6}{bench_cum:>9.2%}{'':>10}{'':>10}")
    for tag, r in results.items():
        print(f"  {tag:<6}{r['cum']:>9.2%}{r['mdd']:>9.2%}"
              f"{r['cum'] - r['bench']:>+9.2%}"
              f"{r['orders']/len(days):>8.1f}{r['fees']:>10,.0f}{r['neg_cash']:>9}")
    print("  明细: " + " / ".join(r["file"] for r in results.values())
          + "  (非正式账本)")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B 双账户每日自动模拟盘引擎")
    ap.add_argument("--init", action="store_true", help="初始化全部策略账户(默认 ¥100,000)")
    ap.add_argument("--status", action="store_true", help="只打印现状不入账")
    ap.add_argument("--replay", metavar="START:END",
                    help="历史区间回放自检(默认双账户对比, 不落正式账本)")
    ap.add_argument("--strategy", default=None,
                    help="逗号分隔策略名(init/replay 可选过滤), 可选: "
                         f"{', '.join(STRATEGIES)}")
    ap.add_argument("--account", type=int, default=ACCOUNT,
                    help="账户初始资金(--init 写入账本; replay 用作回放基数)")
    args = ap.parse_args()
    sel = ([s.strip() for s in args.strategy.split(",")]
           if args.strategy else None)

    if args.init:
        cmd_init(account=args.account, strategies=sel)
    elif args.status:
        cmd_status()
    elif args.replay:
        run_replay(args.replay, account=args.account, strategies=sel)
    else:
        run_daily(args)


if __name__ == "__main__":
    main()
