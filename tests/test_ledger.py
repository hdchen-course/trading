#!/usr/bin/env python3
"""紙上交易帳本測試:正確落地 + 冪等去重(重複執行不重複記錄)。"""

import pandas as pd

from engine.ledger import log_signals


def _event(coin="BTCUSDT", direction="LONG", t="2026-10-19 00:00:00"):
    return {
        "coin": coin, "timeframe": "4h", "signal_name": "EMA50/200",
        "direction": direction, "price": 50000.0, "time": t,
        "init_stop": 47000.0, "notional_pct_of_equity": 0.33,
    }


def test_log_and_dedup(tmp_path):
    path = str(tmp_path / "ledger.csv")
    assert log_signals([_event()], path) == 1          # 首次寫入
    assert log_signals([_event()], path) == 0          # 同訊號再跑 → 不重複
    # 不同方向/時間 → 視為新訊號
    assert log_signals([_event(direction="SHORT")], path) == 1
    assert log_signals([_event(t="2026-10-19 04:00:00")], path) == 1
    df = pd.read_csv(path)
    assert len(df) == 3


def test_empty_events_noop(tmp_path):
    path = str(tmp_path / "ledger.csv")
    assert log_signals([], path) == 0
