#!/usr/bin/env python3
"""分段 walk-forward 一致性報表:把組合報酬切成連續時段,逐段看績效。

目的:誠實量化「edge 是跨時間穩定,還是集中在某一段 regime」。順勢策略本質上
會有好時段與壞時段;這張表把它攤開,避免被單一亮眼區間誤導。

執行:./venv/bin/python scripts/walkforward.py [段數,預設6]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import research_combined as rc  # noqa: E402


def seg_stats(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if len(ret) < 20:
        return {}
    eq = (1 + ret).cumprod()
    years = len(ret) / 365.25
    return {
        "days": len(ret),
        "total": eq.iloc[-1] - 1,
        "ann": eq.iloc[-1] ** (1 / years) - 1,
        "dd": rc.maxdd(eq),
        "sharpe": ret.mean() / ret.std() * np.sqrt(365) if ret.std() > 0 else 0.0,
        "pos_wk": (eq.resample("W").last().pct_change().dropna() > 0).mean(),
    }


def main() -> None:
    n_folds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    legs = rc.collect_legs()
    mat = pd.concat(legs, axis=1)
    mat.columns = range(mat.shape[1])
    port = mat.mean(axis=1, skipna=True).dropna()
    # 只取加密與美股都有資料的重疊期(近 ~4.7 年)以利逐段比較
    port = port.iloc[-int(365 * 4.7):]

    # 用整數索引切段(避免對 Series 直接 array_split 的 FutureWarning)
    bounds = np.linspace(0, len(port), n_folds + 1).astype(int)
    folds = [port.iloc[bounds[i]:bounds[i + 1]] for i in range(n_folds)]
    print(f"分段 walk-forward({n_folds} 段,每段獨立觀察,未放大槓桿=原始 risk2% 口徑)")
    print(f"{'段':>3} {'起':>10} {'迄':>10} {'天':>4} {'總報酬':>8} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'週勝率':>7}")
    sharpes = []
    for i, f in enumerate(folds, 1):
        s = seg_stats(f)
        if not s:
            continue
        sharpes.append(s["sharpe"])
        print(f"{i:>3} {f.index[0].date()!s:>10} {f.index[-1].date()!s:>10} {s['days']:>4} "
              f"{s['total']:>+7.1%} {s['ann']:>+6.1%} {s['dd']:>6.1%} {s['sharpe']:>7.2f} {s['pos_wk']:>6.1%}")

    pos = sum(1 for x in sharpes if x > 0)
    print(f"\n一致性:{len(sharpes)} 段中 {pos} 段 Sharpe>0;"
          f"Sharpe 平均 {np.mean(sharpes):+.2f}、標準差 {np.std(sharpes):.2f}、"
          f"最差 {min(sharpes):+.2f}、最佳 {max(sharpes):+.2f}")
    print("解讀:若多數段為正且最差段不太差 → 跨時間較穩健;若正報酬集中在 1-2 段 → 高度看行情臉色。")


if __name__ == "__main__":
    main()
