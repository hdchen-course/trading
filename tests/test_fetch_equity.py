#!/usr/bin/env python3
"""merge_ohlcv 測試:合併快取與新抓資料,去重留最新、排序(離線純函式)。"""

import pandas as pd

from engine.fetch_equity import merge_ohlcv


def _df(rows):
    return pd.DataFrame(
        [{"timestamp": pd.Timestamp(t), "open": o, "high": h, "low": lo,
          "close": c, "volume": v} for (t, o, h, lo, c, v) in rows]
    )


def test_merge_appends_and_dedups_keeping_latest():
    old = _df([("2026-01-01", 1, 1, 1, 10, 100), ("2026-01-02", 1, 1, 1, 11, 100)])
    # 新資料:重疊 01-02(收盤改為 12 = 修正值)+ 新增 01-03
    new = _df([("2026-01-02", 1, 1, 1, 12, 100), ("2026-01-03", 1, 1, 1, 13, 100)])
    out = merge_ohlcv(old, new)
    assert len(out) == 3
    assert list(out["timestamp"].dt.strftime("%Y-%m-%d")) == ["2026-01-01", "2026-01-02", "2026-01-03"]
    # 01-02 應保留新抓的值 12(keep='last')
    assert out.loc[out["timestamp"] == pd.Timestamp("2026-01-02"), "close"].iloc[0] == 12


def test_merge_handles_empty():
    old = _df([("2026-01-01", 1, 1, 1, 10, 100)])
    assert len(merge_ohlcv(None, old)) == 1
    assert len(merge_ohlcv(old, None)) == 1
    assert len(merge_ohlcv(old, _df([]))) == 1
