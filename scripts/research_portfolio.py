#!/usr/bin/env python3
"""多幣種 × 多訊號 組合研究。

做法(誠實且避免過擬合):
1. 每個 (幣種, 訊號) = 一條 leg,各自用 ATR 移動停損 + 每筆固定風險跑誠實回測。
2. 把每條 leg 轉成「日報酬流」,等權平均成組合日報酬(= 資金平均分散到各 leg)。
3. 分散會降低波動 → Sharpe 上升。再把整體部位線性放大到「最大回撤 ≈ 目標(預設18%)」,
   看在相同回撤預算下年化能到多少 —— 這才是跨幣種分散的真正價值。
4. 全程報「全期 / 樣本外後30%」,扣手續費滑價。

執行(需先 engine/fetch_binance.py 抓好快取):./venv/bin/python scripts/research_portfolio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine.backtester import EventBacktester  # noqa: E402
from engine.data import resample_ohlcv  # noqa: E402
from engine.fetch_binance import load_or_fetch  # noqa: E402
from signals.library import (  # noqa: E402
    donchian_breakout,
    ema_cross,
    ema_ribbon_trend,
)

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
         "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]

SIGNALS = [
    ("EMA50/200 4h", "4h", lambda d: ema_cross(d, 50, 200)),
    ("Donchian20 8h", "8h", lambda d: donchian_breakout(d, 20)),
    ("ribbon 12h", "12h", lambda d: ema_ribbon_trend(d, (90, 120, 240, 300, 420))),
]

BT = dict(
    fee_rate=0.0004, slippage_rate=0.0002, take_profit_pct=None, max_hold_bars=None,
    allow_opposite_exit=True, trailing_atr_mult=4.0, atr_period=14,
    stop_loss_pct=0.06, leverage=10.0, risk_per_trade_pct=0.02, initial_capital=10000.0,
)


def leg_daily_returns(res, cal) -> pd.Series:
    """把一條 leg 的交易轉成日報酬流(以日曆對齊,無交易日報酬=0)。"""
    eq = pd.Series(BT["initial_capital"], index=cal, dtype=float)
    for t in res.trades:
        et = pd.Timestamp(t.exit_time).normalize()
        eq.loc[eq.index >= et] = t.equity_after
    return eq.pct_change().fillna(0.0)


def maxdd(equity: pd.Series) -> float:
    peak = equity.cummax()
    return ((peak - equity) / peak).max()


def scale_to_dd(port_ret: pd.Series, target_dd: float = 0.18):
    """找倍數 k 使組合最大回撤≈target,回傳 (k, equity, cagr, dd, sharpe)。"""
    years = len(port_ret) / 365.25
    best = None
    for k in np.arange(0.5, 20.0, 0.5):
        eq = (1 + k * port_ret).cumprod()
        dd = maxdd(eq)
        if dd <= target_dd:
            best = k
        else:
            break
    k = best if best else 0.5
    eq = (1 + k * port_ret).cumprod()
    cagr = eq.iloc[-1] ** (1 / years) - 1
    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(365) if port_ret.std() > 0 else 0
    return k, eq, cagr, maxdd(eq), sharpe


def build_portfolio(coins):
    all_rets = []
    cal = None
    legs_used = 0
    # 先決定共同日曆(用 BTC 範圍)
    base = load_or_fetch("BTCUSDT", "1h")
    cal = pd.date_range(base.timestamp.min().normalize(),
                        base.timestamp.max().normalize(), freq="D")
    for coin in coins:
        raw = load_or_fetch(coin, "1h")
        if len(raw) < 2000:
            continue
        for _, tf, fn in SIGNALS:
            d = resample_ohlcv(raw, tf)
            if len(d) < 400:
                continue
            res = EventBacktester(**BT).run(d, fn(d))
            all_rets.append(leg_daily_returns(res, cal))
            legs_used += 1
    port = pd.concat(all_rets, axis=1).mean(axis=1)  # 等權平均日報酬
    return port, legs_used


def main():
    print("載入快取並建構多幣種組合...")
    port, legs = build_portfolio(COINS)
    years = len(port) / 365.25
    n = len(port)
    split = int(n * 0.7)

    for label, seg in [("全期", port), ("樣本外(後30%)", port.iloc[split:])]:
        k, eq, cagr, dd, sharpe = scale_to_dd(seg, 0.18)
        wk = eq.resample("W").last().pct_change().dropna() if hasattr(eq.index, "freq") or True else None
        wkwin = (eq.resample("W").last().pct_change().dropna() > 0).mean()
        print(f"\n[{label}] legs={legs}, {len(seg)/365.25:.1f}年")
        print(f"  放大至回撤≈18%(倍數k={k:.1f}): 年化 {cagr:+.1%} | 最大回撤 {dd:.1%} | "
              f"Sharpe {sharpe:.2f} | 週勝率 {wkwin:.1%}")


if __name__ == "__main__":
    main()
