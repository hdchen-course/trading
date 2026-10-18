#!/usr/bin/env python3
"""Telegram 通知模組測試:格式化正確、未設定時安全略過(不拋例外、不送網路)。"""

import engine.notify as notify


def test_not_configured_returns_false(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert notify.telegram_configured() is False
    # 未設定時應直接回 False,不嘗試連網、不拋例外
    assert notify.send_telegram("hi") is False


def test_format_empty_events():
    assert notify.format_signal_events([]) == ""


def test_format_long_and_short():
    events = [
        {"coin": "BTCUSDT", "signal_name": "EMA50/200", "timeframe": "4h",
         "direction": "LONG", "price": 50000.0, "init_stop": 47000.0,
         "trailing_hint": "4.0×ATR 移動停損"},
        {"coin": "ETHUSDT", "signal_name": "ribbon", "timeframe": "12h",
         "direction": "SHORT", "price": 3000.0, "init_stop": 3180.0,
         "trailing_hint": "4.0×ATR 移動停損"},
    ]
    msg = notify.format_signal_events(events)
    assert "BTCUSDT" in msg and "做多" in msg
    assert "ETHUSDT" in msg and "做空" in msg
    assert "50000" in msg
