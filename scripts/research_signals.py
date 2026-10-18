#!/usr/bin/env python3
"""訊號研究框架:用誠實引擎 + 樣本內/樣本外(時間切分)評估候選訊號。

重點:每個訊號都報「全期 / 前70%(IS) / 後30%(OOS)」三段績效。若只有 IS 好、OOS 崩,
就是過擬合,不採用。所有數字都已扣手續費+滑價(EventBacktester 預設單邊 6bps)。

執行:./venv/bin/python scripts/research_signals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from engine.backtester import EventBacktester  # noqa: E402
from engine.data import load_ohlcv, resample_ohlcv  # noqa: E402
from signals.library import (  # noqa: E402
    donchian_breakout,
    ema_cross,
    ema_ribbon_trend,
    with_trend_filter,
)

DATA = ROOT / "training_data" / "btc_15m_2022_2025.csv"


def _evaluate(df: pd.DataFrame, signal: pd.Series, label: str, **bt_kwargs) -> None:
    """對 全期/IS/OOS 三段分別回測並印出一行摘要。"""
    n = len(df)
    split = int(n * 0.7)
    segments = {
        "全期": (0, n),
        "IS(前70%)": (0, split),
        "OOS(後30%)": (split, n),
    }
    print(f"\n### {label}")
    for seg_name, (a, b) in segments.items():
        seg_df = df.iloc[a:b].reset_index(drop=True)
        seg_sig = signal.iloc[a:b].reset_index(drop=True)
        res = EventBacktester(**bt_kwargs).run(seg_df, seg_sig)
        m = res.metrics
        print(f"  [{seg_name:10s}] {m.summary()}")


def main() -> None:
    raw = load_ohlcv(str(DATA))
    print(f"資料: {raw.timestamp.min()} → {raw.timestamp.max()} ({len(raw)} 根 15m)")

    # 幣安 USDT 永續的典型成本 + 槓桿設定(先用保守值)
    common = dict(
        fee_rate=0.0004,
        slippage_rate=0.0002,
        position_size_ratio=1.0,
        leverage=1.0,
        max_hold_bars=48,
        allow_opposite_exit=True,
        initial_capital=10000.0,
    )

    # ---- 候選 1:Donchian 突破(1h / 4h) ----
    for tf, period, sl, tp in [("1h", 24, 0.03, 0.06), ("4h", 20, 0.05, 0.10)]:
        d = resample_ohlcv(raw, tf)
        sig = donchian_breakout(d, period=period)
        _evaluate(
            d, sig, f"Donchian突破 {tf} period={period} SL={sl:.0%} TP={tp:.0%}",
            stop_loss_pct=sl, take_profit_pct=tp, **common,
        )

    # ---- 候選 2:Donchian 突破 + EMA ribbon 趨勢濾網(只順勢) ----
    for tf, period, sl, tp in [("1h", 24, 0.03, 0.06), ("4h", 20, 0.05, 0.10)]:
        d = resample_ohlcv(raw, tf)
        trend = ema_ribbon_trend(d, spans=(90, 120, 240, 300, 420))
        sig = with_trend_filter(donchian_breakout(d, period=period), trend)
        _evaluate(
            d, sig, f"Donchian突破+趨勢濾網 {tf} period={period} SL={sl:.0%} TP={tp:.0%}",
            stop_loss_pct=sl, take_profit_pct=tp, **common,
        )

    # ---- 候選 3:EMA ribbon 趨勢(狀態型,順勢持有) ----
    for tf, sl, tp in [("4h", 0.06, 0.15), ("1h", 0.04, 0.10)]:
        d = resample_ohlcv(raw, tf)
        sig = ema_ribbon_trend(d, spans=(90, 120, 240, 300, 420))
        _evaluate(
            d, sig, f"EMA ribbon 趨勢 {tf} SL={sl:.0%} TP={tp:.0%}",
            stop_loss_pct=sl, take_profit_pct=tp, **common,
        )

    # ---- 候選 4:EMA 50/200 交叉(1h / 4h) ----
    for tf, sl, tp in [("1h", 0.03, 0.06), ("4h", 0.05, 0.10)]:
        d = resample_ohlcv(raw, tf)
        sig = ema_cross(d, fast=50, slow=200)
        _evaluate(
            d, sig, f"EMA 50/200 交叉 {tf} SL={sl:.0%} TP={tp:.0%}",
            stop_loss_pct=sl, take_profit_pct=tp, **common,
        )


if __name__ == "__main__":
    main()
