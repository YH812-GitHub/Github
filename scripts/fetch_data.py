#!/usr/bin/env python
"""
AkShare 数据下载脚本：拉取沪深300成分股 + 基准指数的日线行情，
输出为 Qlib 官方 dump_bin.py 兼容的 CSV（每只标的一个文件）。

用法:
    .venv/bin/python scripts/fetch_data.py [--start 20200901] [--end 20260821]

输出:
    data/csv_raw/SH600519.csv ... （列: date,open,high,low,close,volume,factor,vwap）

口径说明:
    - open/high/low/close 为【后复权】价格，factor 列保存后复权因子(=后复权收盘/原始收盘)，
      方便用户还原真实价格；
    - vwap = 成交额/成交量，再乘以复权因子，对齐到后复权口径；
    - 成交量的“手/股”单位差异通过 价格区间 自检自动修正；
    - 数据源为东方财富(经 AkShare 封装)，仅用于研究学习。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import akshare as ak
except ImportError:
    sys.exit("请先安装依赖: make setup")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "csv_raw"
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _col(df: pd.DataFrame, key: str) -> str:
    """在中文列名里模糊匹配包含 key 的第一列。"""
    for c in df.columns:
        if key in str(c):
            return str(c)
    raise KeyError(f"找不到包含 {key!r} 的列, 现有列: {list(df.columns)}")


def _retry(fn, tries: int = 3, base: float = 2.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(base * (i + 1))
    raise RuntimeError(f"重试{tries}次仍失败: {last}")


# --------------------------------------------------------------------------- #
# 沪深300 成分股
# --------------------------------------------------------------------------- #
def get_csi300_codes() -> list[str]:
    errors = []
    for fn, colkey in ((ak.index_stock_cons_csindex, "成分券代码"),
                       (ak.index_stock_cons, "代码")):
        try:
            df = _retry(lambda f=fn, k=colkey: f(symbol="000300"))
            col = _col(df, colkey)
            codes = df[col].astype(str).str.extract(r"(\d+)")[0].dropna().str.zfill(6)
            codes = sorted(set(codes))
            if len(codes) >= 200:
                return codes
            errors.append(f"{fn.__name__}: 仅 {len(codes)} 只")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{fn.__name__}: {e}")
    raise RuntimeError("获取沪深300成分失败: " + "; ".join(errors))


def to_symbol(code: str) -> str:
    """6 位代码 → Qlib 规范符号(SH/SZ 前缀)。"""
    if code.startswith(("6", "9")):
        return "SH" + code
    if code.startswith(("0", "3")):
        return "SZ" + code
    if code.startswith(("4", "8")):
        return "BJ" + code  # 沪深300 理论上不含北交所, 防御性兜底
    return "SZ" + code


# --------------------------------------------------------------------------- #
# 行情下载
# --------------------------------------------------------------------------- #
def _stock_history(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    """个股日线(东财)。返回以 date 为索引的标准化 DataFrame。"""
    df = ak.stock_zh_a_hist(symbol=code, period="daily",
                            start_date=start, end_date=end, adjust=adjust)
    if df is None or df.empty:
        raise ValueError(f"空数据({adjust})")
    out = pd.DataFrame(index=pd.to_datetime(df[_col(df, "日期")]))
    for k, v in (("开盘", "open"), ("最高", "high"), ("最低", "low"),
                 ("成交量", "volume"), ("成交额", "amount"), ("收盘", "close")):
        out[v] = pd.to_numeric(df[_col(df, k)], errors="coerce")
    out = (out.dropna(subset=["open", "close"])
              .sort_index().loc[lambda d: ~d.index.duplicated()])
    if out.empty:
        raise ValueError(f"清洗后为空({adjust})")
    return out


def _smart_vwap(amount: pd.Series, volume: pd.Series,
                low: pd.Series, high: pd.Series) -> pd.Series:
    """成交量可能是『手』或『股』：用 vwap 是否落在 [low, high] 区间来自动判定。"""
    vol = volume.replace(0, np.nan)
    best, best_ok = None, -1.0
    for unit in (1.0, 100.0):
        v = amount / (vol * unit)
        ok = float(((v >= low * 0.8) & (v <= high * 1.2)).mean())
        if ok > best_ok:
            best, best_ok = v, ok
    return best


def fetch_one(code: str, start: str, end: str) -> tuple[str, pd.DataFrame]:
    hfq = _retry(lambda: _stock_history(code, start, end, adjust="hfq"))
    time.sleep(0.12)
    raw = _retry(lambda: _stock_history(code, start, end, adjust=""))

    factor = hfq["close"] / raw["close"].reindex(hfq.index).ffill()
    vwap = (_smart_vwap(raw["amount"], raw["volume"], raw["low"], raw["high"])
            .reindex(hfq.index) * factor)

    out = pd.DataFrame({
        "open":   hfq["open"],
        "high":   hfq["high"],
        "low":    hfq["low"],
        "close":  hfq["close"],
        "volume": hfq["volume"],
        "factor": factor,
        "vwap":   vwap,
    }).round(6).dropna(subset=["close"])
    out.index.name = "date"
    return to_symbol(code), out


def fetch_index(start: str, end: str) -> pd.DataFrame:
    """沪深300 指数日线(作为回测基准 SH000300)。"""
    def _pull():
        df = ak.index_zh_a_hist(symbol="000300", period="daily",
                                start_date=start, end_date=end)
        if df is None or df.empty:
            raise ValueError("指数空数据")
        return df
    df = _retry(_pull, tries=5)
    out = pd.DataFrame(index=pd.to_datetime(df[_col(df, "日期")]))
    for k, v in (("开盘", "open"), ("最高", "high"), ("最低", "low"),
                 ("成交量", "volume"), ("成交额", "amount"), ("收盘", "close")):
        out[v] = pd.to_numeric(df[_col(df, k)], errors="coerce")
    out = out.sort_index().loc[lambda d: ~d.index.duplicated()]
    res = pd.DataFrame({
        "open": out["open"], "high": out["high"], "low": out["low"],
        "close": out["close"], "volume": out["volume"], "factor": 1.0,
        "vwap": _smart_vwap(out["amount"], out["volume"], out["low"], out["high"]),
    }).round(6)
    res.index.name = "date"
    return res


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="AkShare → Qlib CSV 数据下载")
    ap.add_argument("--start", default="20200901", help="开始日期 YYYYMMDD")
    ap.add_argument("--end", default=(dt.date.today() + dt.timedelta(days=1)).strftime("%Y%m%d"))
    ap.add_argument("--workers", type=int, default=3, help="并发线程数(勿调太大防封禁)")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    codes = get_csi300_codes()
    log(f"[1/2] 共获取沪深300成分 {len(codes)} 只, 开始下载 {args.start}~{args.end} 日线...")

    ok, failed = [], []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_one, c, args.start, args.end): c for c in codes}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            code = futs[fut]
            try:
                symbol, df = fut.result()
                df.to_csv(RAW_DIR / f"{symbol}.csv")
                ok.append(symbol)
            except Exception as e:  # noqa: BLE001
                failed.append((code, str(e)[:120]))
            if i % 25 == 0 or i == len(codes):
                log(f"    进度 {i}/{len(codes)}  成功={len(ok)} 失败={len(failed)}")

    log("[2/2] 下载基准指数 SH000300...")
    try:
        fetch_index(args.start, args.end).to_csv(RAW_DIR / "SH000300.csv")
        ok.append("SH000300")
    except Exception as e:  # noqa: BLE001
        failed.append(("000300(指数)", str(e)[:120]))

    log(f"\n完成: 成功 {len(ok)}, 失败 {len(failed)}")
    for c, err in failed[:20]:
        log(f"  ✗ {c}: {err}")
    if failed:
        log("提示: 少数标的失败通常为新上市/长期停牌, 可忽略; 大面积失败请检查网络。")
    if len(ok) < int(0.8 * (len(codes) + 1)):
        sys.exit("成功率过低, 请检查网络后重试")


if __name__ == "__main__":
    main()
