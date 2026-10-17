#!/usr/bin/env python3
"""TrendPortfolio.current_signals 的轉換邏輯測試:只在「新翻多/空」那根才通知。"""

import numpy as np
import pandas as pd

from engine.strategy import Leg, TrendPortfolio


def _raw(n=60):
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    price = np.linspace(100, 110, n)
    return pd.DataFrame(
        {"timestamp": ts, "open": price, "high": price + 1, "low": price - 1,
         "close": price, "volume": 1.0}
    )


def _portfolio_with_tail(last, prev):
    """造一個固定訊號(最後兩根為 prev,last)的 Leg,其餘為 0。"""
    def fn(d):
        s = pd.Series(0, index=d.index, dtype=int)
        if len(s) >= 2:
            s.iloc[-2] = prev
            s.iloc[-1] = last
        return s

    leg = Leg(coin="BTCUSDT", timeframe="1h", name="test", signal_fn=fn)
    return TrendPortfolio(legs=[leg], stop_loss_pct=0.06)


def test_fresh_long_triggers_event():
    p = _portfolio_with_tail(last=1, prev=0)
    ev = p.current_signals({"BTCUSDT": _raw()})
    assert len(ev) == 1
    assert ev[0]["direction"] == "LONG"
    assert ev[0]["init_stop"] < ev[0]["price"]  # 多單停損在下方


def test_continuing_signal_no_event():
    """訊號延續(last==prev)不應重複通知。"""
    p = _portfolio_with_tail(last=1, prev=1)
    assert p.current_signals({"BTCUSDT": _raw()}) == []


def test_flip_to_short_triggers_event():
    p = _portfolio_with_tail(last=-1, prev=1)
    ev = p.current_signals({"BTCUSDT": _raw()})
    assert len(ev) == 1
    assert ev[0]["direction"] == "SHORT"
    assert ev[0]["init_stop"] > ev[0]["price"]  # 空單停損在上方


def test_no_signal_no_event():
    p = _portfolio_with_tail(last=0, prev=0)
    assert p.current_signals({"BTCUSDT": _raw()}) == []


def test_missing_coin_is_skipped():
    p = _portfolio_with_tail(last=1, prev=0)
    assert p.current_signals({}) == []  # 沒資料不報錯
