#!/usr/bin/env python
"""
从 mlruns 里取出最近一次 qrun 的实验产物，导出人类可读的报告：
  - reports/ic_summary.txt         IC / RankIC 汇总
  - reports/ic_daily.csv           每日 RankIC 序列
  - reports/portfolio_*.csv        组合回测相关对象(自动发现并导出 DataFrame)
  - reports/equity_curve.png       策略累计收益 vs 指数基准

设计上完全防御式：不依赖 qlib 内部对象命名，逐个尝试加载 pkl。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    from qlib.workflow import R  # noqa: PLC0415

    exp = R.get_exp(experiment_name="workflow")
    recorders = exp.list_recorders()
    if not recorders:
        raise SystemExit("未找到任何 recorder, 请先运行 make train")
    rec = max(recorders.values(), key=lambda r: getattr(r, "start_time", None) or "")
    print(f"使用 recorder: {rec.id}  ({len(recorders)} 个候选)")

    # ---- 逐个加载所有 pkl 产物 ------------------------------------------------
    objs: dict[str, object] = {}
    for art in rec.list_artifacts():
        path = art.path if hasattr(art, "path") else art
        name = str(path)
        if not name.endswith(".pkl"):
            continue
        local = rec.download_artifact(name) if hasattr(rec, "download_artifact") else None
        try:
            if local is not None:
                with open(local, "rb") as f:
                    objs[name] = pickle.load(f)
            else:
                objs[name] = rec.load_object(name)
        except Exception as e:  # noqa: BLE001
            print(f"  [跳过] {name}: {e}")

    print("\n=== 实验产物清单 ===")
    for name, obj in objs.items():
        kind = type(obj).__name__
        extra = ""
        if isinstance(obj, pd.DataFrame):
            extra = f" shape={obj.shape} cols={list(obj.columns)[:8]}"
        elif isinstance(obj, pd.Series):
            extra = f" len={len(obj)}"
        print(f"  {name:<55} {kind}{extra}")

    # ---- 导出所有 DataFrame/Series -------------------------------------------
    for i, (name, obj) in enumerate(objs.items()):
        tag = Path(name).as_posix().replace("/", "_").removesuffix(".pkl")
        try:
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(REPORTS / f"{tag}.csv")
            elif isinstance(obj, pd.Series):
                obj.to_frame().to_csv(REPORTS / f"{tag}.csv")
        except Exception as e:  # noqa: BLE001
            print(f"  [导出失败] {name}: {e}")
        if i > 40:
            break

    # ---- IC 汇总 ---------------------------------------------------------------
    lines = []
    for name, obj in objs.items():
        if isinstance(obj, (pd.DataFrame, pd.Series)) and "ic" in name.lower():
            s = obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj
            try:
                mean, std = float(s.mean()), float(s.std())
                lines.append(f"{name}: mean={mean:+.4f}  std={std:.4f}  "
                             f"ICIR={mean / (std + 1e-12):+.3f}  "
                             f"IC>0占比={(s > 0).mean():.1%}")
            except Exception:  # noqa: BLE001
                pass
    summary = "\n".join(lines) if lines else "(未找到 IC 对象)"
    (REPORTS / "ic_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print("\n=== IC 汇总 ===\n" + summary)

    # ---- 净值曲线 ---------------------------------------------------------------
    curve = None
    for name, obj in objs.items():
        if isinstance(obj, pd.DataFrame) and {"return", "bench"} <= set(obj.columns):
            curve = obj.rename(columns={"return": "strategy"})
            break
        if isinstance(obj, pd.DataFrame) and {"return"} <= set(obj.columns) \
                and "account" not in obj.columns[:2]:
            curve = obj
    if curve is not None and "strategy" in curve.columns:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        (1 + curve["strategy"].fillna(0)).cumprod().plot(ax=ax, lw=1.6, label="Strategy(Topk30)")
        if "bench" in curve.columns:
            (1 + curve["bench"].fillna(0)).cumprod().plot(ax=ax, lw=1.4, alpha=0.8, label="CSI300 Bench")
        ax.set_title("Out-of-sample Cumulative Return (test segment)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(REPORTS / "equity_curve.png", dpi=130)
        print(f"\n净值图已保存: {REPORTS / 'equity_curve.png'}")
    else:
        print("\n(未识别净值序列对象, 请查看上方导出的 portfolio_*.csv)")


if __name__ == "__main__":
    main()
