#!/usr/bin/env python
"""
A股基本面因子数据模块（t13 dq 策略数据源）。

【数据源变更说明】任务书原指定 乐咕 legulegu 的 ak.stock_a_indicator_lg，实测不可用：
该函数在 akshare≥1.17 已被上游移除（本项目锁定 1.18.94），且其底层端点
POST legulegu.com/api/s/base-info/ 现返回 401(需站点会员)。故改用项目内已验证的
东方财富批量端点构建同口径因子：
  - 每股净资产(bvps): ak.stock_yjbb_em(date=报告期)  —— 全市场一次一报告期
  - 现金分红明细:     ak.stock_fhps_em(date=报告期)  —— 含每10股派现与除权除息日
  单位已用贵州茅台2023年报核验：现金分红比例 308.76 即每10股308.76元(dps=30.876)，
  公告日股息率 2.03% 与公开资料一致。

缓存策略: 每报告期一个 parquet 落在 paper/cache/, mtime 超 ttl 才重取——
防限流且离线可复算。基本面为季度/年度低频数据，默认 ttl=30 天。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "paper" / "cache"
TTL_DAYS = 30


def _cache_read(name: str, ttl_days: int = TTL_DAYS):
    fp = CACHE_DIR / f"{name}.parquet"
    if not fp.exists():
        return None
    age = datetime.now() - datetime.fromtimestamp(fp.stat().st_mtime)
    if age > timedelta(days=ttl_days):
        return None
    return pd.read_parquet(fp)


def _cache_write(name: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_DIR / f"{name}.parquet")


def quarter_periods(start_year: int = 2022, end: datetime | None = None) -> list[str]:
    """生成 'YYYYMMDD' 报告期列表(1231/0930/0630/0331), 至最近已结束季度。"""
    end = end or datetime.now()
    out = []
    y = start_year
    while y <= end.year:
        for md in ("0331", "0630", "0930", "1231"):
            d = pd.Timestamp(f"{y}{md}")
            if d <= pd.Timestamp(end) - pd.Timedelta(days=45):  # 披露缓冲
                out.append(f"{y}{md}")
        y += 1
    return out


def load_bvps(start_year: int = 2022, force: bool = False) -> pd.DataFrame:
    """每股净资产长表: columns=[code6, announce, bvps], 每行=某报告期公告值。"""
    name = "em_yjbb_bvps"
    if not force:
        hit = _cache_read(name)
        if hit is not None:
            return hit
    import akshare as ak  # noqa: PLC0415

    frames = []
    for period in quarter_periods(start_year):
        try:
            df = ak.stock_yjbb_em(date=period)
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] yjbb {period} 失败: {str(exc)[:80]}")
            continue
        sub = df[["股票代码", "最新公告日期", "每股净资产"]].copy()
        sub.columns = ["code", "announce", "bvps"]
        sub["code"] = sub["code"].str.replace(r"\D", "", regex=True)
        sub["announce"] = pd.to_datetime(sub["announce"], errors="coerce")
        sub["bvps"] = pd.to_numeric(sub["bvps"], errors="coerce")
        frames.append(sub.dropna())
        print(f"    bvps {period}: {len(sub)} 行")
    out = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["code", "announce"])
    _cache_write(name, out)
    return out


def load_dividends(start_year: int = 2022, force: bool = False) -> pd.DataFrame:
    """现金分红事件长表: columns=[code6, ex_date, dps(元/股)], 仅实施分配。"""
    name = "em_fhps_dps"
    if not force:
        hit = _cache_read(name)
        if hit is not None:
            return hit
    import akshare as ak  # noqa: PLC0415

    frames = []
    for period in quarter_periods(start_year):
        try:
            df = ak.stock_fhps_em(date=period)
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] fhps {period} 失败: {str(exc)[:80]}")
            continue
        df = df[df.get("方案进度", "").eq("实施分配")] if "方案进度" in df else df
        sub = df[["代码", "除权除息日", "现金分红-现金分红比例"]].copy()
        sub.columns = ["code", "ex_date", "per10"]
        sub["code"] = sub["code"].astype(str).str.replace(r"\D", "", regex=True)
        sub["ex_date"] = pd.to_datetime(sub["ex_date"], errors="coerce")
        sub["dps"] = pd.to_numeric(sub["per10"], errors="coerce") / 10.0
        frames.append(sub.dropna()[["code", "ex_date", "dps"]])
        print(f"    div {period}: {len(sub)} 行")
    out = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["code", "ex_date"])
    _cache_write(name, out)
    return out


if __name__ == "__main__":
    b = load_bvps()
    d = load_dividends()
    print("bvps:", b.shape, "| div:", d.shape)
