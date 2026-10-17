#!/usr/bin/env python3
"""資料層與訊號層測試:重點是「無前視偏誤」的性質驗證。

無前視的定義:把資料截斷到前 k 根後重算,前 k 根的指標/訊號必須與用完整資料算出的
前 k 根完全相同(除了 warmup 期的 NaN)。若未來資料會改變過去的值,就是前視偏誤。
"""

import numpy as np
import pandas as pd
import pytest

from engine.data import atr, donchian, ema, resample_ohlcv
from signals.library import donchian_breakout, ema_cross, ema_ribbon_trend, with_trend_filter


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close,
         "volume": rng.uniform(1, 100, n)}
    )


# ---------- 無前視:截斷資料不改變過去訊號 ----------
@pytest.mark.parametrize(
    "fn",
    [
        lambda d: donchian_breakout(d, period=48),
        lambda d: ema_cross(d, fast=20, slow=60),
        lambda d: ema_ribbon_trend(d, spans=(20, 40, 80)),
    ],
)
def test_signals_have_no_lookahead(fn):
    df = _synthetic(500)
    full = fn(df)
    k = 300
    truncated = fn(df.iloc[:k].copy())
    # 前 k 根訊號必須一致(未來資料不得影響過去)
    pd.testing.assert_series_equal(
        full.iloc[:k].reset_index(drop=True),
        truncated.reset_index(drop=True),
        check_names=False,
    )


@pytest.mark.parametrize("ind", ["ema", "atr", "donchian"])
def test_indicators_have_no_lookahead(ind):
    df = _synthetic(500)
    k = 300
    if ind == "ema":
        full = ema(df["close"], 50)
        trunc = ema(df["close"].iloc[:k], 50)
    elif ind == "atr":
        full = atr(df, 14)
        trunc = atr(df.iloc[:k], 14)
    else:
        full = donchian(df, 48)[0]
        trunc = donchian(df.iloc[:k].copy(), 48)[0]
    pd.testing.assert_series_equal(
        full.iloc[:k].reset_index(drop=True),
        trunc.reset_index(drop=True),
        check_names=False,
    )


def test_donchian_excludes_current_bar():
    """donchian 上軌在第 i 根應等於第 i-period..i-1 根的最高 high(不含當根)。"""
    df = _synthetic(100)
    period = 10
    upper, lower = donchian(df, period)
    i = 50
    expected_upper = df["high"].iloc[i - period : i].max()
    assert upper.iloc[i] == pytest.approx(expected_upper)


def test_signals_values_in_set():
    df = _synthetic(400)
    for s in [
        donchian_breakout(df, 48),
        ema_cross(df, 20, 60),
        ema_ribbon_trend(df, (20, 40, 80)),
    ]:
        assert set(pd.unique(s)).issubset({-1, 0, 1})


def test_with_trend_filter_keeps_only_aligned():
    idx = range(5)
    entry = pd.Series([1, -1, 1, -1, 0], index=idx)
    trend = pd.Series([1, 1, -1, -1, 1], index=idx)
    out = with_trend_filter(entry, trend)
    # 第0:1==1留;第1:-1 vs 1 濾掉;第2:1 vs -1 濾掉;第3:-1==-1留;第4:0
    assert list(out) == [1, 0, 0, -1, 0]


def test_resample_aggregation():
    """15m → 1h 聚合:open=first,high=max,low=min,close=last,volume=sum。"""
    ts = pd.date_range("2024-01-01 00:00", periods=4, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [10, 11, 12, 13],
            "high": [15, 16, 14, 17],
            "low": [9, 8, 11, 12],
            "close": [11, 12, 13, 14],
            "volume": [1, 2, 3, 4],
        }
    )
    out = resample_ohlcv(df, "1h")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["open"] == 10
    assert row["high"] == 17
    assert row["low"] == 8
    assert row["close"] == 14
    assert row["volume"] == 10
