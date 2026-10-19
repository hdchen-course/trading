#!/usr/bin/env python3
"""績效視覺化報表:把跨資產順勢組合的權益曲線、回撤、月報酬畫成圖+表。

重用 research_combined 的 leg 建構(含 regime + 距離 gate),把組合縮放到 18% 回撤預算,
輸出:
- output/performance.png(上:權益曲線;下:水下回撤圖)
- 終端印出月報酬表與摘要

執行:./venv/bin/python scripts/report_performance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
# 繁中字型(macOS 內建 Heiti TC / Arial Unicode MS),避免中文顯示成方框
matplotlib.rcParams["font.sans-serif"] = ["Heiti TC", "Arial Unicode MS", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import research_combined as rc  # noqa: E402


def scale_factor(port_ret: pd.Series, target_dd: float = 0.18) -> float:
    best = 0.5
    for k in np.arange(0.5, 25.0, 0.5):
        if rc.maxdd((1 + k * port_ret).cumprod()) <= target_dd:
            best = k
        else:
            break
    return best


def main() -> None:
    legs = rc.collect_legs()
    mat = pd.concat(legs, axis=1)
    mat.columns = range(mat.shape[1])
    port = mat.mean(axis=1, skipna=True).dropna()

    # 聚焦近 4 年(全宇宙都有資料的代表期)
    port = port.iloc[-int(365 * 4):]
    k = scale_factor(port, 0.18)
    eq = (1 + k * port).cumprod()
    peak = eq.cummax()
    underwater = (eq - peak) / peak

    years = len(port) / 365.25
    cagr = eq.iloc[-1] ** (1 / years) - 1
    sharpe = port.mean() / port.std() * np.sqrt(365) if port.std() > 0 else 0

    # 月報酬表
    monthly = eq.resample("ME").last().pct_change().dropna()
    print(f"\n=== 績效摘要(近4年,縮放至18%回撤,倍數k={k:.1f})===")
    print(f"年化 {cagr:+.1%} | 最大回撤 {rc.maxdd(eq):.1%} | Sharpe {sharpe:.2f} | "
          f"月勝率 {(monthly > 0).mean():.1%}")
    print("\n各年月報酬(%):")
    mtab = monthly.copy()
    mtab.index = mtab.index.strftime("%Y-%m")
    print((mtab * 100).round(1).to_string())

    # 畫圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(eq.index, eq.values, color="#2563eb", lw=1.3)
    ax1.set_title(f"順勢組合權益曲線(近4年, 18%回撤預算)  年化{cagr:+.1%} / 回撤{rc.maxdd(eq):.1%} / Sharpe{sharpe:.2f}")
    ax1.set_ylabel("權益(起始=1)")
    ax1.grid(alpha=0.3)
    ax2.fill_between(underwater.index, underwater.values * 100, 0,
                     color="#dc2626", alpha=0.5)
    ax2.set_ylabel("回撤 %")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "output" / "performance.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\n📈 圖已存:{out}")


if __name__ == "__main__":
    main()
