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


def test_format_includes_scale_out_and_quality():
    """推播須含分批放階梯與品質傾斜(與 run_live 主控台一致,避免漏掉出場計畫)。"""
    events = [{
        "coin": "BTCUSDT", "signal_name": "EMA50/200", "timeframe": "4h",
        "direction": "LONG", "price": 50000.0, "init_stop": 47000.0,
        "trailing_hint": "6.0×ATR 移動停損", "risk_pct": 0.02,
        "notional_pct_of_equity": 0.137, "quality_mult": 0.414,
        "scale_out_hint": "到 60000.0000(+20%)減碼 33%;到 70000.0000(+40%)減碼 33%,其餘 6.0×ATR 移動停損",
    }]
    msg = notify.format_signal_events(events)
    assert "分批放" in msg
    assert "減碼 33%" in msg
    assert "品質×0.414" in msg
