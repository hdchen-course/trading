#!/usr/bin/env python3
"""總經風險 overlay(VIX 美股做多風險off)測試:no-lookahead + 只濾美股做多。"""

import pandas as pd

from engine.risk_overlay import (
    filter_equity_longs,
    is_risk_off_now,
    vix_risk_off,
)


def _vix(vals):
    ts = pd.date_range("2024-01-01", periods=len(vals), freq="D")
    return pd.DataFrame({"timestamp": ts, "close": vals})


def test_vix_risk_off_is_lagged_no_lookahead():
    """判定用的是『落後 lag 天』的 VIX:尖峰當天不觸發,隔天才觸發。"""
    s = pd.Series([15, 15, 40, 15, 15], dtype=float)  # index2 當天 VIX=40
    ro = vix_risk_off(s, threshold=30.0, lag_days=1)
    assert ro.iloc[2] == False  # 尖峰當天還不知道(用昨收 15)  # noqa: E712
    assert ro.iloc[3] == True   # 隔天才用到 40 → 風險off  # noqa: E712
    assert ro.iloc[0] == False  # 無昨日資料 → 不干預  # noqa: E712


def test_is_risk_off_now_uses_prior_close():
    assert is_risk_off_now(_vix([15, 15, 35]), threshold=30.0) is False  # 今=35但昨=15
    assert is_risk_off_now(_vix([15, 35, 20]), threshold=30.0) is True   # 昨=35>30
    assert is_risk_off_now(_vix([35]), threshold=30.0) is False          # 資料不足不干預


def test_filter_equity_longs_only_blocks_equity_longs():
    events = [
        {"coin": "SPY", "direction": "LONG", "timeframe": "1d"},    # 美股多 → 濾
        {"coin": "SPY", "direction": "SHORT", "timeframe": "1d"},   # 美股空 → 留
        {"coin": "BTCUSDT", "direction": "LONG", "timeframe": "4h"},  # 加密多 → 留
    ]
    kept, suppressed = filter_equity_longs(events, risk_off=True)
    assert len(suppressed) == 1 and suppressed[0]["coin"] == "SPY"
    assert {e["coin"] for e in kept} == {"SPY", "BTCUSDT"}  # 空單+加密都留
    assert len(kept) == 2


def test_filter_noop_when_not_risk_off():
    events = [{"coin": "SPY", "direction": "LONG", "timeframe": "1d"}]
    kept, suppressed = filter_equity_longs(events, risk_off=False)
    assert kept == events and suppressed == []
