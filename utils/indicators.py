#!/usr/bin/env python3
"""
Crypto Trading System Utilities

統一的技術指標計算函數，確保所有模組使用相同的計算邏輯。
"""

import pandas as pd
import numpy as np
import talib
import warnings

warnings.filterwarnings("ignore")


def calculate_amplitude(high, low, close):
    """計算振幅"""
    return (high - low) / close


def calculate_volume_trend(df, window_size):
    """計算成交量趨勢 (窗口內前半段 vs 後半段)"""
    half_window = window_size // 2
    vol_recent = (
        df["volume"].rolling(window=half_window).mean().iloc[window_size - 1 :].values
    )
    vol_prev = (
        df["volume"]
        .rolling(window=half_window)
        .mean()
        .shift(half_window)
        .iloc[window_size - 1 :]
        .values
    )

    volume_trend = np.ones(len(df))
    volume_trend[window_size - 1 :] = np.where(vol_prev > 0, vol_recent / vol_prev, 1.0)
    return volume_trend


def calculate_price_change_pct(df, window_size):
    """計算窗口內價格變化百分比"""
    close_start = df["close"].iloc[: -window_size + 1].values
    close_end = df["close"].iloc[window_size - 1 :].values
    price_change_window = (close_end - close_start) / close_start * 100

    price_change_pct = np.zeros(len(df))
    price_change_pct[window_size - 1 :] = price_change_window
    return price_change_pct


def calculate_price_atr_ratio_1h(df):
    """計算1小時前價格變化相對ATR比值"""
    atr = talib.ATR(
        df["high"].values, df["low"].values, df["close"].values, timeperiod=14
    )
    price_1h_ago = df["close"].shift(4)
    price_diff_1h = df["close"] - price_1h_ago
    return price_diff_1h / atr


def calculate_technical_indicators(df):
    """計算所有基本技術指標"""
    df = df.copy()

    # 基本指標
    df["amplitude"] = calculate_amplitude(df["high"], df["low"], df["close"])

    # MACD (12, 48, 9)
    exp1 = df["close"].ewm(span=12).mean()
    exp2 = df["close"].ewm(span=48).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # 成交量變化率
    df["volume_change"] = df["volume"].pct_change()

    # SAR
    df["sar"] = talib.SAR(df["high"].values, df["low"].values)
    df["price_above_sar"] = df["close"] > df["sar"]

    # RSI
    df["rsi"] = talib.RSI(df["close"].values, timeperiod=14)

    # ATR
    df["atr"] = talib.ATR(
        df["high"].values, df["low"].values, df["close"].values, timeperiod=14
    )

    # 100EMA
    df["ema_100"] = talib.EMA(df["close"].values, timeperiod=100)

    # 50EMA
    df["ema_50"] = talib.EMA(df["close"].values, timeperiod=50)

    # 布林帶
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(df["close"].values)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )

    return df


def calculate_second_derivative(series):
    """計算序列的二階導數 (加速度)"""
    first_derivative = series.diff()
    second_derivative = first_derivative.diff()
    return second_derivative


def calculate_z_score(series, window=200):
    """計算z-score標準化"""
    rolling_mean = series.rolling(window=window, min_periods=20).mean()
    rolling_std = series.rolling(window=window, min_periods=20).std()
    z_score = (series - rolling_mean) / rolling_std
    return z_score.fillna(0)


def analyze_cumulative_metrics(df, window_size=16):
    """分析累計技術指標"""
    n = len(df)
    if n < window_size:
        raise ValueError(f"Data length {n} is less than window size {window_size}")

    results = {
        "period": range(window_size, n + 1),
        "timestamp": df["timestamp"].iloc[window_size - 1 :].values,
        "close_price": df["close"].iloc[window_size - 1 :].values,
        "ema_100": df["ema_100"].iloc[window_size - 1 :].values,
        "ema_50": df["ema_50"].iloc[window_size - 1 :].values,
        "avg_amplitude": df["amplitude"]
        .rolling(window=window_size)
        .mean()
        .iloc[window_size - 1 :]
        .values,
        "avg_volume": df["volume"]
        .rolling(window=window_size)
        .mean()
        .iloc[window_size - 1 :]
        .values,
        "avg_macd": df["macd"]
        .rolling(window=window_size)
        .mean()
        .iloc[window_size - 1 :]
        .values,
        "avg_rsi": df["rsi"]
        .rolling(window=window_size)
        .mean()
        .iloc[window_size - 1 :]
        .values,
        "price_above_sar_ratio": df["price_above_sar"]
        .rolling(window=window_size)
        .mean()
        .iloc[window_size - 1 :]
        .values,
        "avg_bb_position": df["bb_position"]
        .rolling(window=window_size)
        .mean()
        .iloc[window_size - 1 :]
        .values,
        "volatility": df["close"]
        .pct_change()
        .rolling(window=window_size)
        .std()
        .iloc[window_size - 1 :]
        .values,
    }

    # 計算窗口內價格變化百分比
    results["price_change_pct"] = calculate_price_change_pct(df, window_size)[
        window_size - 1 :
    ]

    # 計算1小時前價格變化相對ATR比值
    price_atr_ratio_1h = calculate_price_atr_ratio_1h(df)
    results["price_atr_ratio_1h"] = price_atr_ratio_1h.iloc[window_size - 1 :].values

    # 計算成交量趨勢
    volume_trend = calculate_volume_trend(df, window_size)
    results["volume_trend"] = volume_trend[window_size - 1 :]

    # 計算二階導數 (加速度) 並標準化為z-score
    macd_series = pd.Series(results["avg_macd"])
    price_series = pd.Series(results["close_price"])
    volume_trend_series = pd.Series(results["volume_trend"])
    atr_series = pd.Series(
        df["atr"].rolling(window=window_size).mean().iloc[window_size - 1 :].values
    )

    # 計算原始二階導數
    macd_2nd_raw = calculate_second_derivative(macd_series)
    price_2nd_raw = calculate_second_derivative(price_series)
    vol_2nd_raw = calculate_second_derivative(volume_trend_series)
    atr_2nd_raw = calculate_second_derivative(atr_series)

    # 標準化為z-score (200期滾動)
    results["macd_2nd"] = calculate_z_score(macd_2nd_raw, window=200).values
    results["price_2nd"] = calculate_z_score(price_2nd_raw, window=200).values
    results["vol_2nd"] = calculate_z_score(vol_2nd_raw, window=200).values
    results["atr_2nd"] = calculate_z_score(atr_2nd_raw, window=200).values

    # 把原始df中對應時間點的所有欄位複製到results中（除了timestamp和已存在的欄位）
    df_slice = df.iloc[window_size - 1:]
    for col in df.columns:
        if col != 'timestamp' and col not in results:
            results[col] = df_slice[col].values

    return pd.DataFrame(results)
