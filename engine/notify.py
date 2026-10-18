#!/usr/bin/env python3
"""Telegram 通知(讀環境變數,未設定時安全略過,不寫死任何密鑰)。

設定方式(在 shell 或 .env 匯出):
    export TELEGRAM_BOT_TOKEN="123456:ABC..."
    export TELEGRAM_CHAT_ID="123456789"

未設定時 send_telegram() 直接回傳 False(不報錯),方便本機測試與老機離線跑。
只依賴標準庫 urllib,無第三方依賴。
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import List


def telegram_configured() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))


def send_telegram(message: str, timeout: int = 10) -> bool:
    """送一則訊息到 Telegram。成功回 True;未設定或失敗回 False(不拋例外)。"""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(
        {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    ).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=timeout) as resp:
            return json.loads(resp.read()).get("ok", False)
    except Exception:
        return False


def format_signal_events(events: List[dict]) -> str:
    """把 current_signals() 的事件清單格式化成一則中文通知訊息。"""
    if not events:
        return ""
    lines = ["📡 <b>順勢訊號觸發</b>"]
    for e in events:
        arrow = "🟢 做多" if e["direction"] == "LONG" else "🔴 做空"
        size_line = ""
        if e.get("notional_pct_of_equity") is not None:
            size_line = (
                f"\n　建議部位:名目約權益 {e['notional_pct_of_equity']:.0%}"
                f"(停損被打到最多賠 {e.get('risk_pct', 0):.0%})"
            )
        lines.append(
            f"\n{arrow} <b>{e['coin']}</b> ({e['signal_name']} {e['timeframe']})\n"
            f"　價格 {e['price']:.4f}\n"
            f"　建議初始停損 {e['init_stop']:.4f}"
            f"{size_line}\n"
            f"　出場:{e['trailing_hint']}"
        )
    return "".join(lines)
