#!/usr/bin/env python3
"""誠實績效報告:把目前 main 系統(順勢三件套)的真實表現整理成一份,可重跑。

用 end-anchored 多窗(近1/2/3/4/5年)為主——比單一「後30%」切點對資料 vintage 穩定(見研究教訓)。
每個窗把組合放大到 18% 最大回撤預算,報 年化/回撤/Sharpe/週勝率/週達標(+1.5%)。
誠實告誡直接印在報告裡,避免被漂亮數字誤導。

執行:./venv/bin/python scripts/performance_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import research_combined as rc  # noqa: E402

ANN = 365.0


def maxdd(eq):
    peak = eq.cummax()
    return ((peak - eq) / peak).max()


def metrics(ret, target_dd=0.18):
    ret = ret.dropna()
    if len(ret) < 60 or ret.std() == 0:
        return None
    best = 0.5
    for k in np.arange(0.5, 60.0, 0.5):
        if maxdd((1 + k * ret).cumprod()) <= target_dd:
            best = k
        else:
            break
    eq = (1 + best * ret).cumprod()
    years = len(ret) / ANN
    wk = eq.resample("W").last().pct_change().dropna() if isinstance(
        ret.index, pd.DatetimeIndex) else pd.Series(dtype=float)
    return dict(
        years=years, k=best, cagr=eq.iloc[-1] ** (1 / years) - 1, dd=maxdd(eq),
        sharpe=ret.mean() / ret.std() * np.sqrt(ANN),
        wk_win=float((wk > 0).mean()) if len(wk) else float("nan"),
        wk_15=float((wk >= 0.015).mean()) if len(wk) else float("nan"),
    )


def main():
    port = rc.collect_legs()
    mat = pd.concat(port, axis=1)
    mat.columns = range(mat.shape[1])
    p = mat.mean(axis=1, skipna=True).dropna()
    p.index = pd.DatetimeIndex(p.index)

    wins = [("全期", p), ("樣本外後30%", p.iloc[int(len(p) * 0.7):]),
            ("近1年", p.iloc[-int(ANN):]), ("近2年", p.iloc[-int(ANN * 2):]),
            ("近3年", p.iloc[-int(ANN * 3):]), ("近4年", p.iloc[-int(ANN * 4):]),
            ("近5年", p.iloc[-int(ANN * 5):])]

    lines = []
    lines.append(f"# 誠實績效報告(自動產生,{pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC)\n")
    lines.append("系統:高週期順勢(EMA/Donchian/ribbon)+ regime(EMA200)+ 距離 gate + "
                 "ATR 移動停損(4×)+ 時間停損(max_hold 80)+ 每筆風險 2% + 多標的分散。")
    lines.append(f"宇宙:加密 {rc.CRYPTO}(4h/8h/12h)+ 美股 {len(rc.EQUITY)} 檔非槓桿(日線)。"
                 f"共 {len(port)} 條 leg,組合 {len(p)} 日。每窗放大到 18% 最大回撤預算。\n")
    lines.append(f"{'時間窗':<12}{'年化':>9}{'回撤':>8}{'Sharpe':>9}{'週勝率':>9}{'週達標+1.5%':>12}")
    for name, seg in wins:
        m = metrics(seg)
        if m is None:
            lines.append(f"{name:<12}  (資料不足)")
            continue
        lines.append(f"{name:<12}{m['cagr']:>+8.1%}{m['dd']:>8.1%}{m['sharpe']:>9.2f}"
                     f"{m['wk_win']:>9.1%}{m['wk_15']:>12.1%}")

    lines.append("\n## 誠實告誡(務必讀)")
    lines.append("- **樣本外(後30%)Sharpe 對資料 vintage 敏感**(cache 刷新會移動切點);以 end-anchored "
                 "近N年多窗為準較穩。")
    lines.append("- **近4年數字偏高(科技/AI 大多頭+加密牛市)不是常態**;務實定錨看較長/較保守的窗。")
    lines.append("- **+1.5%/週(≈年化+117%)結構上不可達**;看「週達標+1.5%」欄多數窗僅個位數~二十餘%,"
                 "即多數週達不到。務實目標=年化雙位數、回撤可控。")
    lines.append("- 所有數字為回測(含成本:加密 taker、美股低費+滑價),**非實盤**;美股側用 yfinance 真實日線,"
                 "與幣安美股合約實盤(尤其休市段預言機定價)有落差,打折看待。")
    lines.append("- 這是目前誠實最佳版本,但無任何保證;過去表現不代表未來。")

    report = "\n".join(lines)
    print(report)
    out = ROOT / "PERFORMANCE.md"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\n(已寫入 {out})")


if __name__ == "__main__":
    main()
