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

# 實際交易宇宙:加密只 BTC/AAVE(24h,幣安 1h);美股(日線,非槓桿 1x 底層)。
# 槓桿 ETF(SOXL/TQQQ/SPXL... 每日重置)已移除:趨勢 sleeve 樣本外無 edge(衰減+假突破)。
# 訊號跑在 1x 底層(QQQ/SPY/SOXX...),要槓桿由使用者用對應工具執行。擴充到流動大型名:
# OOS Sharpe 0.52→0.86、近4年 0.89→1.49(同回撤),OOS 與近4年同步改善=真分散。
CRYPTO = ["BTCUSDT", "AAVEUSDT"]
EQUITIES = ["SOXX", "MSTR", "INTC", "NVDA", "SPY", "QQQ", "SMH", "AMD",
            "GOOGL", "META", "AAPL", "AMZN", "TSLA", "MSFT"]


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

    # 有訓練好的 meta 模型就掛上(對訊號做品質部位傾斜);無則照常
    meta_path = ROOT / "models" / "meta" / "meta_model.pkl"
    port = TrendPortfolio(
        legs=legs, meta_model_path=str(meta_path) if meta_path.exists() else None)
    events = port.current_signals(data)
    exits = port.current_exits(data)  # 時間停損:持倉滿 max_hold_bars 建議平倉(已驗證 edge)
    # 註:VIX>30 風險off overlay 已於 2026-10-21 重審後移除——在現行 6×trail+分批放配置下,
    # 它在全部端錨窗皆略降 Sharpe(砍掉了高期望的「恐慌中順勢多單」),見 engine/risk_overlay.py 說明。

    for e in events:
        print(
            f"🔔 {e['coin']} {e['direction']} | {e['signal_name']}({e['timeframe']}) | "
            f"價 {e['price']:.4f} | 初始停損 {e['init_stop']:.4f} | "
            f"建議部位≈權益{e['notional_pct_of_equity']:.1%}"
            + (f"(品質×{e['quality_mult']})" if e.get('quality_mult', 1.0) != 1.0 else "")
            + (f"\n    ↳ 分批放:{e['scale_out_hint']}" if e.get('scale_out_hint') else "")
        )
    for x in exits:
        print(
            f"⏳ 平倉 {x['coin']} {x['direction']} | {x['signal_name']}({x['timeframe']}) | "
            f"價 {x['price']:.4f} | 持倉 {x['held_bars']} 根(時間停損)"
        )

    if not events and not exits:
        print("目前沒有新觸發的進場訊號或時間平倉(低頻特性,多數時間如此)。")
        return

    if events:
        # 落地到紙上交易帳本(冪等),供日後實盤 vs 回測比對
        added = log_signals(events)
        print(f"📒 紙上帳本新增 {added} 筆(output/paper_trades.csv)")

    # 組合推播訊息(進場 + 時間平倉)
    parts = []
    if events:
        parts.append(format_signal_events(events))
    if exits:
        lines = ["⏳ 時間停損(持倉到期,建議平倉;趨勢續行待新訊號再進):"]
        lines += [f"• {x['coin']} {x['direction']} {x['signal_name']}({x['timeframe']}) "
                  f"持倉 {x['held_bars']} 根 @ {x['price']:.4f}" for x in exits]
        parts.append("\n".join(lines))
    msg = "\n\n".join(parts)

    if telegram_configured():
        ok = send_telegram(msg)
        print(f"Telegram 推播{'成功' if ok else '失敗'}。")
    else:
        print("(未設定 TELEGRAM_BOT_TOKEN/CHAT_ID,略過推播)")


if __name__ == "__main__":
    main()
