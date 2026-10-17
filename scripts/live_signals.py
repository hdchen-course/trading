#!/usr/bin/env python3
"""Live 訊號儀表板:輸出各幣各訊號的「當前狀態」與「這一根是否剛觸發」。

這是未來 Telegram 通知的內容來源。目前讀 data/cache 的歷史(剛抓的,接近當下);
正式上線時由監控迴圈定期以最新資料呼叫 TrendPortfolio.current_signals()。

執行:./venv/bin/python scripts/live_signals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from engine.data import resample_ohlcv  # noqa: E402
from engine.fetch_binance import load_or_fetch  # noqa: E402
from engine.strategy import TrendPortfolio, default_legs  # noqa: E402

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
         "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]


def main() -> None:
    data = {c: load_or_fetch(c, "1h") for c in COINS}
    latest = max(df["timestamp"].max() for df in data.values())
    print(f"資料截至約 {latest} (UTC) / {latest + pd.Timedelta(hours=8)} (台北)\n")

    port = TrendPortfolio(legs=default_legs(COINS))

    # 1) 當前狀態總表(每腿:多/空/觀望)
    print("=== 當前各訊號狀態 ===")
    header = f"{'幣種':<10}{'EMA50/200(4h)':>16}{'Donchian20(8h)':>16}{'ribbon(12h)':>14}"
    print(header)
    for coin in COINS:
        raw = data[coin]
        states = []
        for leg in [lg for lg in port.legs if lg.coin == coin]:
            d = resample_ohlcv(raw, leg.timeframe)
            s = int(leg.signal_fn(d).iloc[-1])
            states.append({1: "做多", -1: "做空", 0: "觀望"}[s])
        print(f"{coin:<10}{states[0]:>14}{states[1]:>16}{states[2]:>14}")

    # 2) 這一根剛觸發的新訊號(= 會發 Telegram 的)
    events = port.current_signals(data)
    print("\n=== 本根剛觸發的新訊號(未來會推播) ===")
    if not events:
        print("(目前沒有新觸發的訊號 — 多數時間如此,這是正常的低頻特性)")
    else:
        for e in events:
            print(
                f"🔔 {e['coin']} {e['direction']} | {e['signal_name']}({e['timeframe']}) | "
                f"價 {e['price']:.4f} | 建議初始停損 {e['init_stop']:.4f} | {e['trailing_hint']}"
            )


if __name__ == "__main__":
    main()
