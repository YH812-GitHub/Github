#!/usr/bin/env python
"""
dq 策略: 红利低波质量(t13) —— 截面百分位加权打分, 月度调仓 + MA200 闸门。

打分 = 股息率40% + 低波30% + 低PB30% (三项均取截面百分位, 低波/低PB 取逆序):
    股息率 dv_ttm = 近365日每股现金分红(按除权除息日归属) / 真实价
    低波 vol60   = 近60交易日日收益率标准差(真实价口径)
    低PB         = 真实价 / 最近一期公告的每股净资产
数据源见 data_fundamentals.py(EM 批量端点替代已失效的乐咕接口, 缓存在 paper/cache/)。

有效性窗口: FACTOR_START=2023-06-01 起, 波动率预热约 60 个交易日,
即 T ≥ 2023-09-01 的打分有效(本盘与回放区间均满足)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from strategy_registry import register_strategy  # noqa: E402

FACTOR_START = "2023-06-01"
W_VOL, W_DV, W_PB = 0.30, 0.40, 0.30


def _csi300_codes() -> list[str]:
    return [ln.split("\t")[0].strip() for ln in
            (ROOT / "data/qlib_data/cn_data/instruments/csi300.txt")
            .read_text().splitlines() if ln.strip()]


def _real_close_matrix(end_day: pd.Timestamp) -> pd.DataFrame:
    """dt×inst 真实收盘价矩阵($close/$factor), 自 FACTOR_START 起。"""
    import qlib  # noqa: PLC0415

    qlib.init(provider_uri=str(ROOT / "data" / "qlib_data" / "cn_data"), region="cn")
    from qlib.data import D  # noqa: PLC0415

    px = D.features(_csi300_codes(), ["$close", "$factor"],
                    start_time=FACTOR_START, end_time=end_day, freq="day")
    real = (px["$close"] / px["$factor"]).unstack(level="instrument").sort_index()
    return real


def _inst_map() -> dict[str, str]:
    """裸6位码 -> qlib instrument(带交易所前缀), 与 csv_raw 符号规则一致。"""
    m = {}
    for inst in _csi300_codes():
        code = inst[2:]
        m[code] = inst
    # 兜底: 万一某裸码不在池内(如指数), 按首位数字推断
    for code in ("000300",):
        pass
    return m


def _reindex_to_instruments(df: pd.DataFrame) -> pd.DataFrame:
    m = _inst_map()
    renamed = df.rename(columns={c: m.get(c, c) for c in df.columns})
    return renamed


def _bvps_effective(close_idx: pd.DatetimeIndex,
                    close_cols: pd.Index) -> pd.DataFrame:
    """bvps 长表 → dt×inst 有效值矩阵: 公告日起生效, 公告前 NaN, 之后沿用最新。"""
    from data_fundamentals import load_bvps  # noqa: PLC0415

    b = load_bvps()
    bare = {c[2:] for c in close_cols}
    b = b[b["code"].isin(bare)]
    wide = {}
    for code, g in b.groupby("code"):
        s = g.set_index("announce")["bvps"].sort_index()
        eff = s.reindex(s.index.union(close_idx)).ffill().reindex(close_idx)
        eff.iloc[: s.index.searchsorted(close_idx[0])] = np.nan   # 公告前不可用
        wide[code] = eff
    return _reindex_to_instruments(pd.DataFrame(wide)).reindex(columns=close_cols)


def _dv_ttm(close: pd.DataFrame) -> pd.DataFrame:
    """近365日每股分红(除权除息日归属)/真实价。逐码 searchsorted 向量化累计。"""
    from data_fundamentals import load_dividends  # noqa: PLC0415

    ev = load_dividends()
    m = _inst_map()
    bare = set(m)
    ev = ev[ev["code"].isin(bare)]
    days = close.index
    day_i8 = days.to_numpy(dtype="datetime64[ns]").astype("int64")
    lo_bound = (days - pd.Timedelta(days=365)).to_numpy(dtype="datetime64[ns]").astype("int64")
    out = {}
    for inst in close.columns:
        code = inst[2:]
        g = ev[ev["code"] == code].sort_values("ex_date")
        dps = g["dps"].to_numpy(dtype=float)
        dts = g["ex_date"].to_numpy(dtype="datetime64[ns]").astype("int64")
        cum = np.concatenate([[0.0], np.cumsum(dps)])
        lo = np.searchsorted(dts, lo_bound, side="left")
        hi = np.searchsorted(dts, day_i8, side="right")
        out[inst] = pd.Series(cum[hi] - cum[lo], index=days)
    return pd.DataFrame(out)[close.columns]


def make_dq_score(end_day: pd.Timestamp) -> pd.Series:
    close = _real_close_matrix(end_day)
    vol60 = close.pct_change(fill_method=None).rolling(60).std()
    bvps_eff = _bvps_effective(close.index, close.columns)
    pb = close / bvps_eff.replace(0, np.nan)
    dv = _dv_ttm(close) / close.replace(0, np.nan)

    r_dv = dv.rank(axis=1, pct=True)
    r_vol = vol60.rank(axis=1, pct=True)
    r_pb = pb.rank(axis=1, pct=True)
    score = W_DV * r_dv + W_VOL * (1 - r_vol) + W_PB * (1 - r_pb)

    long = score.stack()                      # MultiIndex(datetime, instrument)
    long = long.dropna()
    long.name = "score"
    return long.sort_index()


@register_strategy("dq_dvpb")
def _dq_factory():
    return make_dq_score


if __name__ == "__main__":
    s = make_dq_score(pd.Timestamp.today().normalize())
    last_day = s.index.get_level_values(0).max()
    top = s.xs(last_day).sort_values(ascending=False).head(8)
    print(f"最近打分日 {last_day.date()} Top8:")
    for inst, v in top.items():
        px_close = None
        print(f"  {inst:<10} {v:.4f}")
