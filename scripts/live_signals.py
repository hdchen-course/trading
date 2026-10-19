#!/usr/bin/env python3
"""Live 訊號儀表板(乾跑,不推播、不落帳):

- 當前各腿 gated 訊號「狀態」(做多/做空/觀望)
- 這一根剛觸發的新訊號(= run_live 會推播的)

宇宙與 run_live 一致:加密 BTC/AAVE(幣安 1h)+ 美股/ETF(日線)。讀快取為主,
故反映最近一次抓取的狀態;要當下即時請跑 run_live(會刷新)。

執行:./venv/bin/python scripts/live_signals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.fetch_binance import load_or_fetch  # noqa: E402
from engine.strategy import TrendPortfolio, default_legs, equity_legs  # noqa: E402

CRYPTO = ["BTCUSDT", "AAVEUSDT"]
# 槓桿 ETF 已移除(趨勢 sleeve 樣本外無 edge,每日重置衰減);保留非槓桿股+SOXX。見 run_live 註解。
EQUITIES = ["SOXX", "MSTR", "INTC", "NVDA"]
_STATE = {1: "做多", -1: "做空", 0: "觀望"}


def main() -> None:
    data = {c: load_or_fetch(c, "1h") for c in CRYPTO}
    legs = list(default_legs(CRYPTO))
    try:
        from engine.fetch_equity import load_or_fetch_daily

        for s in EQUITIES:
            d = load_or_fetch_daily(s)  # 讀快取(儀表板用,不強制刷新)
            if len(d):
                data[s] = d
        legs += equity_legs([s for s in EQUITIES if s in data])
    except Exception as exc:
        print(f"⚠️ 美股資料層不可用(僅顯示加密): {exc}")

    port = TrendPortfolio(legs=legs)

    print("=== 當前各訊號狀態(已套 regime + 距離 gate)===")
    print(f"{'標的':<10}{'週期':>5}{'訊號':>12}{'狀態':>6}{'價格':>12}")
    for s in port.current_states(data):
        if s["state"] != 0:  # 只列有部位的,觀望的省略以免洗版
            print(f"{s['coin']:<10}{s['timeframe']:>5}{s['signal_name']:>12}"
                  f"{_STATE[s['state']]:>6}{s['price']:>12.4f}")

    events = port.current_signals(data)
    print("\n=== 本根剛觸發的新訊號(run_live 會推播)===")
    if not events:
        print("(目前沒有新觸發 — 低頻特性,多數時間如此)")
    else:
        for e in events:
            print(f"🔔 {e['coin']} {e['direction']} | {e['signal_name']}({e['timeframe']}) | "
                  f"價 {e['price']:.4f} | 初始停損 {e['init_stop']:.4f} | "
                  f"部位≈權益{e['notional_pct_of_equity']:.0%}")


if __name__ == "__main__":
    main()
