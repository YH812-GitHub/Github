"""策略注册表(t13 A/B 扩展点)。策略模块 import 本文件并用 @register_strategy 登记;
paper_trade 引擎只认注册表, 记账/换仓/幂等逻辑与具体策略解耦。"""
from __future__ import annotations

from typing import Callable

# name -> factory() -> score(end_day)->Series[MultiIndex(datetime, instrument)]
STRATEGIES: dict[str, Callable] = {}

# 调度语义(引擎侧): cadence=每日调仓 or 每月首个交易日; ma200_gate=是否启用闸门
POLICIES: dict[str, dict] = {
    "ml_top8": {"cadence": "daily", "ma200_gate": False},   # ml 口径一字不动
    "dq_dvpb": {"cadence": "monthly", "ma200_gate": True},
}

TAGS: dict[str, str] = {"ml_top8": "ml", "dq_dvpb": "dq"}   # 账户文件后缀


def register_strategy(name: str):
    def deco(fn):
        STRATEGIES[name] = fn
        POLICIES.setdefault(name, {"cadence": "daily", "ma200_gate": False})
        TAGS.setdefault(name, name.split("_")[0][:8])
        return fn
    return deco


def short_tag(name: str) -> str:
    return TAGS.get(name, name[:8])
