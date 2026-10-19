#!/usr/bin/env python3
"""即時訊號監控入口(一次性執行,外部用 cron/排程定期呼叫,例如每小時一次)。

流程:抓各幣最新 1h 資料 → TrendPortfolio.current_signals() 找「本根新觸發」的訊號
→ 印出 → 若已設定 TELEGRAM_BOT_TOKEN/CHAT_ID 則推播。

執行:./venv/bin/python scripts/run_live.py
設定 Telegram(可選):
    export TELEGRAM_BOT_TOKEN=... ; export TELEGRAM_CHAT_ID=...
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from engine.fetch_binance import fetch_recent  # noqa: E402
from engine.ledger import log_signals  # noqa: E402
from engine.notify import format_signal_events, send_telegram, telegram_configured  # noqa: E402
from engine.strategy import TrendPortfolio, default_legs, equity_legs  # noqa: E402

# 實際交易宇宙:加密只 BTC/AAVE(24h,幣安 1h);美股/ETF(日線)
CRYPTO = ["BTCUSDT", "AAVEUSDT"]
EQUITIES = ["SOXL", "TQQQ", "SOXX", "KORU", "MSTR", "INTC", "NVDA", "SPXL",
            "TNA", "LABU", "FAS", "ERX", "NUGT", "YINN", "UDOW", "UPRO"]


def main() -> None:
    now = pd.Timestamp.utcnow()
    print(f"[{now:%Y-%m-%d %H:%M} UTC] 抓取最新資料並檢查訊號...")
    data = {}
    legs = list(default_legs(CRYPTO))
    for c in CRYPTO:
        try:
            data[c] = fetch_recent(c, "1h", bars=6000)
        except Exception as exc:  # 單一標的失敗不影響其他
            print(f"  ⚠️ {c} 抓取失敗: {exc}")

    # 美股/ETF 日線(yfinance 研究資料源;老機器無 yfinance 時此段略過,加密路徑仍可跑)
    try:
        from engine.fetch_equity import load_or_fetch_daily

        for s in EQUITIES:
            try:
                data[s] = load_or_fetch_daily(s, refresh=True)  # 補上當日最新日線
            except Exception as exc:
                print(f"  ⚠️ {s} 抓取失敗: {exc}")
        legs += equity_legs([s for s in EQUITIES if s in data])
    except Exception as exc:
        print(f"  ⚠️ 美股資料層不可用(略過,僅跑加密): {exc}")

    port = TrendPortfolio(legs=legs)
    events = port.current_signals(data)

    if not events:
        print("目前沒有新觸發的訊號(低頻特性,多數時間如此)。")
        return

    for e in events:
        print(
            f"🔔 {e['coin']} {e['direction']} | {e['signal_name']}({e['timeframe']}) | "
            f"價 {e['price']:.4f} | 初始停損 {e['init_stop']:.4f}"
        )

    # 落地到紙上交易帳本(冪等),供日後實盤 vs 回測比對
    added = log_signals(events)
    print(f"📒 紙上帳本新增 {added} 筆(output/paper_trades.csv)")

    msg = format_signal_events(events)
    if telegram_configured():
        ok = send_telegram(msg)
        print(f"Telegram 推播{'成功' if ok else '失敗'}。")
    else:
        print("(未設定 TELEGRAM_BOT_TOKEN/CHAT_ID,略過推播)")


if __name__ == "__main__":
    main()
