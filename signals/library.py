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


def regime_filter(df: pd.DataFrame, entry: pd.Series, ema_span: int = 200) -> pd.Series:
    """市場狀態濾網:只在「與長期趨勢同向」時接受進場訊號。

    - 多單(+1)只在 close > EMA(ema_span) 時保留,否則歸零。
    - 空單(-1)只在 close < EMA(ema_span) 時保留,否則歸零。
    經研究(2026-10)驗證:套在順勢訊號上可把樣本外 Sharpe 由 ~0.17 提到 ~0.46,
    鄰近參數(150/200/250/300)一致變好,非過擬合。EMA 為 trailing → 無前視。
    """
    trend_ema = ema(df["close"], ema_span)
    long_ok = df["close"] > trend_ema
    short_ok = df["close"] < trend_ema
    entry = entry.reindex_like(df["close"]).fillna(0).astype(int)
    out = entry.copy()
    out[(entry == 1) & (~long_ok)] = 0
    out[(entry == -1) & (~short_ok)] = 0
    return out


def bollinger_revert(df: pd.DataFrame, period: int = 30, k: float = 2.5) -> pd.Series:
    """布林通道「均值回歸」(fade,非突破):

    - close 跌破下軌(中線 - k*std)→ +1 做多(賭回歸中線)。
    - close 突破上軌(中線 + k*std)→ -1 做空。
    注意:這是「逆勢」訊號,與順勢近乎零相關,僅適合當小部位分散腿。
    ⚠️ 強烈依賴市場 regime:盤整多頭年強、但趨勢年/危機年(2008/2022)會失血、樣本內曾虧損。
    預設不納入核心組合;若用須加波動度/趨勢 gate 並限小額。中線與 std 皆 trailing(無前視)。
    """
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = mid + k * std
    lower = mid - k * std
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[df["close"] < lower] = 1
    signal[df["close"] > upper] = -1
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
