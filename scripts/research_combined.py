#!/usr/bin/env python3
"""跨資產(加密永續 + 美股槓桿ETF/個股)順勢組合研究。

- 加密:1h 資料聚合到 4h/8h/12h,訊號 EMA50/200、Donchian20、ribbon。
- 美股/ETF:日線,訊號 EMA50/200、EMA20/50、Donchian20(日線 ribbon 太慢故不用)。
- 每條 leg 各自用 ATR 移動停損 + 每筆固定風險跑誠實回測,轉成日報酬流。
- 每天只在「當時已上市/有資料」的 leg 間等權平均(避免未上市期間稀釋)。
- 放大整體部位到最大回撤≈目標,報全期 / 樣本外(後30%)。

執行:./venv/bin/python scripts/research_combined.py
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
from engine.fetch_equity import load_or_fetch_daily  # noqa: E402
from signals.library import donchian_breakout, ema_cross, ema_ribbon_trend  # noqa: E402

# 使用者只接受的加密標的:BTC 與 AAVE(其餘山寨相關性高又弱,加了反而降 Sharpe)
CRYPTO = ["BTCUSDT", "AAVEUSDT"]
EQUITY = ["SOXL", "TQQQ", "SOXX", "KORU", "MSTR", "INTC", "NVDA", "SPXL", "TNA", "LABU",
          "FAS", "ERX", "NUGT", "YINN", "UDOW", "UPRO", "RETL", "DRN", "DFEN", "CURE", "DUST"]

CRYPTO_SIGS = [
    ("4h", lambda d: ema_cross(d, 50, 200)),
    ("8h", lambda d: donchian_breakout(d, 20)),
    ("12h", lambda d: ema_ribbon_trend(d, (90, 120, 240, 300, 420))),
]
EQUITY_SIGS = [
    lambda d: ema_cross(d, 50, 200),
    lambda d: ema_cross(d, 20, 50),
    lambda d: donchian_breakout(d, 20),
]

BT_CRYPTO = dict(fee_rate=0.0004, slippage_rate=0.0002, stop_loss_pct=0.06)
BT_EQUITY = dict(fee_rate=0.0002, slippage_rate=0.0003, stop_loss_pct=0.08)
BT_COMMON = dict(take_profit_pct=None, max_hold_bars=None, allow_opposite_exit=True,
                 trailing_atr_mult=4.0, atr_period=14, leverage=10.0,
                 risk_per_trade_pct=0.02, initial_capital=10000.0)


def leg_daily_returns(df, sig, bt_kwargs):
    """跑一條 leg,回傳「以其自身資料範圍為準」的日報酬(範圍外為 NaN)。"""
    res = EventBacktester(**bt_kwargs, **BT_COMMON).run(df, sig)
    cal = pd.date_range(df["timestamp"].min().normalize(),
                        df["timestamp"].max().normalize(), freq="D")
    eq = pd.Series(BT_COMMON["initial_capital"], index=cal, dtype=float)
    for t in res.trades:
        eq.loc[eq.index >= pd.Timestamp(t.exit_time).normalize()] = t.equity_after
    return eq.pct_change()


def maxdd(equity):
    peak = equity.cummax()
    return ((peak - equity) / peak).max()


def scale_and_report(port_ret, label, target_dd=0.18):
    port_ret = port_ret.dropna()
    years = len(port_ret) / 365.25
    best = 0.5
    for k in np.arange(0.5, 25.0, 0.5):
        if maxdd((1 + k * port_ret).cumprod()) <= target_dd:
            best = k
        else:
            break
    eq = (1 + best * port_ret).cumprod()
    cagr = eq.iloc[-1] ** (1 / years) - 1
    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(365) if port_ret.std() > 0 else 0
    wk = eq.resample("W").last().pct_change().dropna()
    wk_target = (wk >= 0.015).mean()  # 達到週 +1.5% 的比例
    print(f"[{label}] {years:.1f}年 k={best:.1f} | 年化 {cagr:+.1%} | 回撤 {maxdd(eq):.1%} | "
          f"Sharpe {sharpe:.2f} | 週勝率 {(wk > 0).mean():.1%} | 週達標(+1.5%) {wk_target:.1%}")


def collect_legs():
    legs = []
    for c in CRYPTO:
        raw = load_or_fetch(c, "1h")
        if len(raw) < 2000:
            continue
        for tf, fn in CRYPTO_SIGS:
            d = resample_ohlcv(raw, tf)
            if len(d) >= 400:
                legs.append(leg_daily_returns(d, fn(d), BT_CRYPTO))
    for s in EQUITY:
        d = load_or_fetch_daily(s)
        if len(d) < 400:
            continue
        for fn in EQUITY_SIGS:
            legs.append(leg_daily_returns(d, fn(d), BT_EQUITY))
    return legs


def main():
    print("建構跨資產組合(加密永續 + 美股槓桿ETF/個股)...")
    legs = collect_legs()
    mat = pd.concat(legs, axis=1)
    mat.columns = range(mat.shape[1])  # 避免重複欄名
    # 等權:每天只在「當時有資料」的 leg 間平均。(風險平價在這種多數為0的事件型
    # 報酬流上會退化——長期無交易的腿波動趨近0使權重爆掉,故不採用。)
    port = mat.mean(axis=1, skipna=True)
    n = len(port)
    split = int(n * 0.7)
    print(f"共 {len(legs)} 條 leg\n")
    scale_and_report(port, "全期")
    scale_and_report(port.iloc[split:], "樣本外(後30%)")
    # 只看近 4 年(加密與美股都有資料的重疊期較具代表性)
    recent = port.iloc[-int(365 * 4):]
    scale_and_report(recent, "近4年")


if __name__ == "__main__":
    main()
