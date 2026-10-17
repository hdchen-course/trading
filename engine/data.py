#!/usr/bin/env python3
"""資料載入與指標輔助(全部 trailing,無前視偏誤)。

提供:
- load_ohlcv:讀本地 BTC 15m CSV,回傳標準化 OHLCV DataFrame。
- resample_ohlcv:把 15m 聚合到任意較大週期(1h/4h...),正確用 OHLC 聚合規則。
- ema / atr / donchian / rolling_return:常用指標,全部只用過去資料(無 center、無 shift(-n))。

設計原則:這些函式產生的任何欄位,在第 i 根只會用到第 i 根(含)以前的資料,
確保丟進 EventBacktester 時不會有前視偏誤。
"""

from __future__ import annotations

import pandas as pd

_OHLC_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

# pandas resample 規則對照(分鐘/小時/天)
_RULE = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
    "1d": "1D",
}


def load_ohlcv(path: str) -> pd.DataFrame:
    """讀取本地 CSV(需含 timestamp, open, high, low, close, volume),時間排序後回傳。

    timestamp 一律解析為 datetime(視為 UTC),索引重設為 0..n-1。
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[cols].sort_values("timestamp").drop_duplicates("timestamp")
    return df.reset_index(drop=True)


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """把 OHLCV 聚合到指定週期。timeframe 例如 '1h'、'4h'。

    聚合規則:open=第一筆、high=最高、low=最低、close=最後、volume=加總。
    空桶(無成交)會被移除。回傳的 timestamp 為每根「開始時間」。
    """
    if timeframe not in _RULE:
        raise ValueError(f"不支援的週期: {timeframe},可用: {list(_RULE)}")
    indexed = df.set_index("timestamp")
    out = indexed.resample(_RULE[timeframe], label="left", closed="left").agg(_OHLC_AGG)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out.reset_index()


def ema(series: pd.Series, span: int) -> pd.Series:
    """指數移動平均(trailing)。"""
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """平均真實區間 ATR(trailing,用 Wilder 平滑近似為 EMA)。"""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def donchian(df: pd.DataFrame, period: int):
    """唐奇安通道上下軌(用「前一根為止」的過去 period 根,避免把當根納入 → 無前視)。

    回傳 (upper, lower):第 i 根的 upper = 第 i-period..i-1 根的最高 high。
    這樣「當根 close 突破 upper」才是乾淨的突破訊號(不含當根自己)。
    """
    upper = df["high"].rolling(period).max().shift(1)
    lower = df["low"].rolling(period).min().shift(1)
    return upper, lower


def rolling_return(series: pd.Series, period: int) -> pd.Series:
    """過去 period 根的報酬率(trailing)。"""
    return series.pct_change(period)
