#!/usr/bin/env python3
"""市場結構特徵庫(market-structure features)——把「盤面形態」量化成一組無前視特徵,
供更複雜的情境判斷機制(walk-forward 驗證的模型)使用,取代單一門檻的機械規則。

設計原則:
1. **無前視(no-lookahead)是鐵律**:每個特徵在第 i 根的值,只能用到第 i 根(含)以前的資料。
   「波段/結構」類特徵本質上要等 right 根後才能「確認」,因此一律把確認往後 shift(right),
   確保在時間 t 只引用「到 t 為止已確認」的結構。tests/test_features.py 用「截斷不變性」
   性質測試驗證(把資料截到 i+1 再算,末值必須與用完整資料算的第 i 值相同)。
2. **純 pandas/numpy**(可攜核心,老 Mac 也能跑;不依賴 TA-Lib)。
3. **precision 重於 recall**:這些特徵是給「預設不動、高把握才出手」的模型用的——
   寧可漏接機會(0 成本)也不接爛單(真虧)。模型應輸出信心分數、低於門檻就 abstain。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.data import atr as _atr, ema as _ema


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI(trailing)。0-100,>70 超買、<30 超賣。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0)  # avg_loss=0(一路漲)→ RSI=100


def roc(close: pd.Series, period: int = 20) -> pd.Series:
    """變動率動能(trailing):過去 period 根的報酬。"""
    return close.pct_change(period)


def dist_from_ema(df: pd.DataFrame, span: int = 200, atr_period: int = 14) -> pd.Series:
    """收盤距 EMA 的距離,以 ATR 為單位(signed)。>0 在均線上方、<0 下方;量級=乖離程度。"""
    close = df["close"]
    a = _atr(df, atr_period)
    return (close - _ema(close, span)) / a.replace(0.0, np.nan)


def ema_stack_score(df: pd.DataFrame, spans=(20, 50, 100, 200)) -> pd.Series:
    """多均線排列分數(trailing):相鄰 EMA 由快到慢是否多頭堆疊。
    回傳 [-1, 1]:+1=完全多頭排列(EMA_fast>...>EMA_slow)、-1=完全空頭、0=糾結。
    量化「趨勢階段/一致性」。"""
    close = df["close"]
    emas = [_ema(close, s) for s in spans]
    score = pd.Series(0.0, index=df.index)
    for a, b in zip(emas[:-1], emas[1:]):
        score += np.sign(a - b)  # 快在慢上方 +1、下方 -1
    return score / (len(spans) - 1)


def atr_percentile(df: pd.DataFrame, atr_period: int = 14, lookback: int = 100) -> pd.Series:
    """波動度 regime(trailing):當前 ATR 在過去 lookback 根 ATR 分布中的百分位 [0,1]。
    高=高波動環境、低=低波動(盤整)。"""
    a = _atr(df, atr_period)
    return a.rolling(lookback).apply(
        lambda w: (w.iloc[-1] >= w).mean(), raw=False
    )


def _swing_prices(series: pd.Series, is_high: bool, left: int, right: int) -> pd.Series:
    """內部:回傳「到 t 為止最近一個『已確認』的 swing 價格」(無前視)。
    swing 定義:第 i 根為區域極值(中心窗 left+right+1);確認延遲 right 根後(shift(right))
    才可知,再 ffill 持有。"""
    win = left + right + 1
    roll = series.rolling(win, center=True)
    extreme = roll.max() if is_high else roll.min()
    is_swing = series == extreme
    # swing 在第 i 根成形,但要到第 i+right 根才確認 → 往後移 right,再 ffill
    confirmed = series.where(is_swing).shift(right).ffill()
    return confirmed


def recent_swing_high(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.Series:
    """到 t 為止最近『已確認』的波段高點價(無前視)=當前上方最近的結構壓力參考。"""
    return _swing_prices(df["high"], True, left, right)


def recent_swing_low(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.Series:
    """到 t 為止最近『已確認』的波段低點價(無前視)=當前下方最近的結構支撐參考。"""
    return _swing_prices(df["low"], False, left, right)


def dist_to_resistance(df: pd.DataFrame, left: int = 3, right: int = 3,
                       atr_period: int = 14) -> pd.Series:
    """收盤到最近已確認波段高(壓力)的距離,ATR 單位(≥0 表示在壓力下方多遠;負=已突破)。"""
    a = _atr(df, atr_period)
    return (recent_swing_high(df, left, right) - df["close"]) / a.replace(0.0, np.nan)


def dist_to_support(df: pd.DataFrame, left: int = 3, right: int = 3,
                    atr_period: int = 14) -> pd.Series:
    """收盤到最近已確認波段低(支撐)的距離,ATR 單位(≥0 表示在支撐上方多遠;負=已跌破)。"""
    a = _atr(df, atr_period)
    return (df["close"] - recent_swing_low(df, left, right)) / a.replace(0.0, np.nan)


def trend_age(df: pd.DataFrame, span: int = 200) -> pd.Series:
    """regime 年齡(trailing):自「收盤相對 EMA 的正負號」上次翻轉以來的根數。
    量化趨勢持續多久(配合 max_hold 的『開太久會吐回』直覺)。"""
    close = df["close"]
    side = np.sign(close - _ema(close, span))
    flip = side != side.shift(1)
    grp = flip.cumsum()
    return grp.groupby(grp).cumcount().astype(float)


def htf_ma_distance(df: pd.DataFrame, rule: str, span: int, atr_period: int = 14) -> pd.Series:
    """收盤距「高週期均線」的距離(ATR 單位,signed),無前視。
    使用者直覺:低週期的長均線 ≈ 高週期的軸線(如 2h 300EMA≈月線),這些高週期均線常是
    結構性壓力/支撐。rule 例如 '1D'(日線軸)、'1W'(週線軸)。

    無前視做法:把 df 依 timestamp resample 到 rule、對高週期收盤算 EMA,再 **shift(1) 一根高週期**
    (只用『已收完』的高週期 bar),用 reindex(method='ffill') 映回低週期——低週期第 t 根只會拿到
    「在 t 之前已收盤」的高週期均線值。需要 df 有 timestamp 欄。
    """
    if "timestamp" not in df.columns:
        raise ValueError("htf_ma_distance 需要 df 含 timestamp 欄")
    s = df.set_index(pd.DatetimeIndex(df["timestamp"]))["close"]
    htf_close = s.resample(rule, label="left", closed="left").last()
    ma = htf_close.ewm(span=span, adjust=False).mean().shift(1)  # 只用已收完的高週期 bar
    ma_base = ma.reindex(s.index, method="ffill")  # 每根低週期取「≤當下」最近的已知高週期均線
    a = _atr(df, atr_period)
    dist = (df["close"].to_numpy() - ma_base.to_numpy()) / a.replace(0.0, np.nan).to_numpy()
    return pd.Series(dist, index=df.index)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """組裝一組預設特徵(全部無前視),回傳與 df 同索引的特徵 DataFrame。
    模型/研究由此取用;欄位可擴充。"""
    return pd.DataFrame({
        "rsi14": rsi(df["close"], 14),
        "roc20": roc(df["close"], 20),
        "dist_ema200": dist_from_ema(df, 200),
        "dist_ema50": dist_from_ema(df, 50),
        "ema_stack": ema_stack_score(df, (20, 50, 100, 200)),
        "atr_pct100": atr_percentile(df, 14, 100),
        "dist_resist": dist_to_resistance(df),
        "dist_support": dist_to_support(df),
        "trend_age200": trend_age(df, 200),
        "dist_daily_axis": htf_ma_distance(df, "1D", 50),   # 日線軸(EMA50)
        "dist_weekly_axis": htf_ma_distance(df, "1W", 20),  # 週線軸(EMA20)
    }, index=df.index)
