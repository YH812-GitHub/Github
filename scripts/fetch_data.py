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
_JS_LOCK = threading.Lock()  # 保护 akshare 内部的 py_mini_racer(V8) 初始化(非线程安全)


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
# 数据源探测与备用源(东财被限流时自动降级: 新浪 → 腾讯)
# --------------------------------------------------------------------------- #
_EM_STATE: bool | None = None


def _em_available() -> bool:
    """启动时探测一次东方财富接口是否可用, 结果进程内缓存。"""
    global _EM_STATE
    if _EM_STATE is not None:
        return _EM_STATE
    today = dt.date.today().strftime("%Y%m%d")
    try:
        df = ak.stock_zh_a_hist(symbol="600519", period="daily",
                                start_date=(dt.date.today() - dt.timedelta(days=20)).strftime("%Y%m%d"),
                                end_date=today, adjust="")
        _EM_STATE = df is not None and not df.empty
    except Exception:  # noqa: BLE001
        _EM_STATE = False
    log("[provider] 东方财富接口 " + ("可用" if _EM_STATE
        else "不可用(疑似限流), 本次自动切换 新浪/腾讯 备用源"))
    return _EM_STATE


def _exchange_prefix(code: str) -> str:
    """6 位代码 → 新浪/腾讯风格的小写交易所前缀。"""
    return {"6": "sh", "9": "sh", "0": "sz", "3": "sz",
            "4": "bj", "8": "bj"}.get(code[:1], "sz")


def _standardize(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=pd.to_datetime(df[mapping.get("index", "日期")]))
    for src, dst in mapping.items():
        if dst == "index":
            continue
        out[dst] = pd.to_numeric(df[_col(df, src)], errors="coerce").to_numpy()
    return out


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
_STOCK_MAP = {"index": "日期", "开盘": "open", "最高": "high", "最低": "low",
              "成交量": "volume", "成交额": "amount", "收盘": "close"}


def _em_history(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist(symbol=code, period="daily",
                            start_date=start, end_date=end, adjust=adjust)
    if df is None or df.empty:
        raise ValueError(f"东财空数据({adjust})")
    return _standardize(df, _STOCK_MAP)


def _sina_history(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    # akshare 新浪接口每次调用都会创建 py_mini_racer(V8) 实例,
    # 多线程并发初始化 V8 会直接 FATAL 崩溃(address_pool_manager), 故全局串行化
    with _JS_LOCK:
        df = ak.stock_zh_a_daily(symbol=_exchange_prefix(code) + code,
                                 start_date=start, end_date=end, adjust=adjust)
    if df is None or df.empty:
        raise ValueError(f"新浪空数据({adjust})")
    out = pd.DataFrame(index=pd.to_datetime(df["date"]))
    for c in ("open", "high", "low", "close", "volume", "amount"):
        out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy()
    return out


def _tx_history(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist_tx(symbol=_exchange_prefix(code) + code,
                               start_date=start, end_date=end, adjust=adjust)
    if df is None or df.empty:
        raise ValueError(f"腾讯空数据({adjust})")
    out = pd.DataFrame(index=pd.to_datetime(df["date"]))
    for c in ("open", "high", "low", "close", "volume", "amount"):
        out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy()
    return out


def _providers() -> list:
    """供应商候选列表(东财可用则置顶)。hfq/raw 必须取自同一供应商。"""
    return ([_em_history] if _em_available() else []) + [_sina_history, _tx_history]


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
    # 同源取数: hfq 与 raw 必须来自同一供应商。若各自独立走降级链,
    # 东财间歇性限流时会出现"新浪 hfq / 东财 raw"的跨厂商 factor,
    # 导致 $vwap 与后复权 OHLC 尺度系统性错位且无任何报错(静默污染)。
    hfq = raw = None
    last = None
    for fn in _providers():
        try:
            hfq = fn(code, start, end, "hfq")
            time.sleep(0.12)
            raw = fn(code, start, end, "")
            break
        except Exception as e:  # noqa: BLE001
            hfq = raw = None
            last = e
            log(f"    [fallback] {code} {fn.__name__} 整组失败: {str(e)[:80]}")
    if hfq is None or raw is None or hfq.empty or raw.empty:
        raise RuntimeError(f"{code} 全部数据源失败: {last}")
    hfq = (hfq.dropna(subset=["open", "close"])
              .sort_index().loc[lambda d: ~d.index.duplicated()])
    raw = (raw.dropna(subset=["close"])
              .sort_index().loc[lambda d: ~d.index.duplicated()])
    if hfq.empty or raw.empty:
        raise ValueError(f"{code} 清洗后为空")

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
    """沪深300 指数日线(作为回测基准 SH000300)。优先东财, 限流时降级新浪。"""
    def _pull_em():
        df = ak.index_zh_a_hist(symbol="000300", period="daily",
                                start_date=start, end_date=end)
        if df is None or df.empty:
            raise ValueError("指数空数据")
        out = pd.DataFrame(index=pd.to_datetime(df[_col(df, "日期")]))
        for k, v in (("开盘", "open"), ("最高", "high"), ("最低", "low"),
                     ("成交量", "volume"), ("成交额", "amount"), ("收盘", "close")):
            out[v] = pd.to_numeric(df[_col(df, k)], errors="coerce").to_numpy()
        return out.sort_index().loc[lambda d: ~d.index.duplicated()]

    def _pull_sina():
        df = ak.stock_zh_index_daily(symbol="sh000300")
        if df is None or df.empty:
            raise ValueError("指数空数据(新浪)")
        m = (pd.to_datetime(df["date"]) >= pd.Timestamp(start)) & \
            (pd.to_datetime(df["date"]) <= pd.Timestamp(end))
        df = df.loc[m]
        if df.empty:
            raise ValueError("指数区间为空(新浪)")
        out = pd.DataFrame(index=pd.to_datetime(df["date"]))
        for c in ("open", "high", "low", "close", "volume"):
            out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy()
        return out.sort_index().loc[lambda d: ~d.index.duplicated()]

    try:
        if _em_available():
            out = _retry(_pull_em, tries=5)
        else:
            raise RuntimeError("东财不可用")
    except Exception as e:  # noqa: BLE001
        log(f"    [fallback] 指数改用新浪源 (东财失败: {str(e)[:80]})")
        out = _retry(_pull_sina, tries=5)

    vwap = (_smart_vwap(out["amount"], out["volume"], out["low"], out["high"])
            if "amount" in out.columns else None)
    if vwap is None or vwap.isna().all():  # 新浪指数无成交额: 以典型价 (H+L+2C)/4 近似, 仅作基准参考
        log("    [warn] 该指数源无成交额, vwap 用典型价近似")
        vwap = (out["high"] + out["low"] + 2 * out["close"]) / 4
    res = pd.DataFrame({
        "open": out["open"], "high": out["high"], "low": out["low"],
        "close": out["close"], "volume": out["volume"], "factor": 1.0,
        "vwap": vwap,
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

    # 主线程预初始化: V8(mini_racer) 首次初始化 + 数据源探测, 均只做一次, 避免多线程竞态
    try:
        import py_mini_racer
        py_mini_racer.MiniRacer().eval("1+1")
    except Exception:  # noqa: BLE001
        pass
    _em_available()

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
