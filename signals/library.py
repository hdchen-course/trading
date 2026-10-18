#!/usr/bin/env python3
"""訊號產生器實作(純 pandas,不依賴 TA-Lib)。

全部 trailing:第 i 根的訊號只看第 i 根(含)以前。配合 EventBacktester 於下一根 open 成交。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.data import donchian, ema


def donchian_breakout(df: pd.DataFrame, period: int = 48) -> pd.Series:
    """唐奇安通道突破(順勢):

    - 當根 close 向上突破「前 period 根的最高 high」→ +1 做多。
    - 當根 close 向下跌破「前 period 根的最低 low」→ -1 做空。
    donchian() 的上下軌已 shift(1),不含當根 → 無前視。
    """
    upper, lower = donchian(df, period)
    close = df["close"]
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[close > upper] = 1
    signal[close < lower] = -1
    return signal


def ema_cross(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    """EMA 黃金/死亡交叉(只在「交叉發生的那根」發訊號,避免每根重複):

    - fast EMA 由下往上穿越 slow EMA → +1。
    - fast EMA 由上往下穿越 slow EMA → -1。
    """
    fast_ema = ema(df["close"], fast)
    slow_ema = ema(df["close"], slow)
    diff = fast_ema - slow_ema
    prev_diff = diff.shift(1)
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[(prev_diff <= 0) & (diff > 0)] = 1
    signal[(prev_diff >= 0) & (diff < 0)] = -1
    return signal


def ema_ribbon_trend(
    df: pd.DataFrame, spans=(90, 120, 240, 300, 420)
) -> pd.Series:
    """多條 EMA「排列一致」趨勢(使用者慣用的 EMA ribbon,spans 預設 90/120/240/300/420):

    - 所有 EMA 由快到慢嚴格遞減(短>長)且價格在最快 EMA 之上 → 多頭排列,回傳 +1。
    - 所有 EMA 由快到慢嚴格遞增(短<長)且價格在最快 EMA 之下 → 空頭排列,回傳 -1。
    - 否則 0(排列未對齊)。
    注意:這是「狀態」型訊號(排列成立期間每根都 +1/-1),EventBacktester 會在首次
    進場後持有到退場,不會每根重複開倉。
    """
    spans = sorted(spans)  # 由快(小)到慢(大)
    emas = [ema(df["close"], s) for s in spans]
    close = df["close"]

    bull = pd.Series(True, index=df.index)
    bear = pd.Series(True, index=df.index)
    for i in range(len(emas) - 1):
        bull &= emas[i] > emas[i + 1]  # 快 > 慢
        bear &= emas[i] < emas[i + 1]  # 快 < 慢
    bull &= close > emas[0]
    bear &= close < emas[0]

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[bull] = 1
    signal[bear] = -1
    return signal


def with_trend_filter(
    entry: pd.Series, trend: pd.Series
) -> pd.Series:
    """只保留「與趨勢同向」的進場訊號。

    entry/trend 皆為 {-1,0,1} 序列。回傳:entry 方向與 trend 相同才保留,否則 0。
    用法:把較快的進場訊號(如突破)用較慢的趨勢(如 ribbon)過濾,降低逆勢假突破。
    """
    entry = entry.reindex_like(trend).fillna(0).astype(int)
    aligned = np.where(entry == trend, entry, 0)
    return pd.Series(aligned, index=trend.index, dtype=int)
