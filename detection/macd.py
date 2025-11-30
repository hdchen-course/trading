#!/usr/bin/env python3
"""
MACD Detection Methods
MACD 相關檢測方法
"""

import numpy as np
from .base import DetectionMethod
from utils.indicators import analyze_cumulative_metrics, calculate_price_atr_ratio_1h


class MacdReversalDetection(DetectionMethod):
    """4小時滾動窗口MACD反轉檢測"""

    def __init__(self):
        super().__init__("MACD Reversal")

    def detect(self, df):
        window_size = 16
        n = len(df)
        if n < window_size:
            return np.zeros(n, dtype=bool)

        # 檢查必要的欄位是否存在
        required_cols = ['macd_2nd', 'price_2nd', 'atr_2nd', 'price_atr_ratio_1h', 'price_change_pct']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Please call analyze_cumulative_metrics() before detection."
            )
        
        # 直接使用傳入的 df（已包含累計指標）
        analysis_df = df

        # Round到小數點第二位
        macd_2nd_rounded = np.round(analysis_df["macd_2nd"], 2)
        price_2nd_rounded = np.round(analysis_df["price_2nd"], 2)
        atr_2nd_rounded = np.round(analysis_df["atr_2nd"], 2)

        # 當期條件
        curr_macd_abs = np.abs(macd_2nd_rounded)
        curr_price_abs = np.abs(price_2nd_rounded)
        curr_atr = atr_2nd_rounded

        condition1 = (
            (curr_macd_abs >= 0.5)
            & (curr_macd_abs <= 1.5)
            & (curr_price_abs >= 0.5)
            & (curr_price_abs <= 1.5)
            & (curr_atr > -0.5)
            & (analysis_df["price_atr_ratio_1h"] > 0)
        )

        # 當期價格變化百分比 > 1%
        condition2 = analysis_df["price_change_pct"] > 1

        # 添加 price_2nd 和 macd_2nd 相反方向的條件
        condition3 = ((macd_2nd_rounded > 0) & (price_2nd_rounded < 0)) | (
            (macd_2nd_rounded < 0) & (price_2nd_rounded > 0)
        )

        signals = condition1 & condition2 & condition3

        # 轉換回原始df的索引格式用於返回
        signal_indices = []
        if signals.any():
            signal_positions = signals[signals].index
            for pos in signal_positions:
                # analysis_df的索引對應原始df的索引需要加上window_size-1
                original_idx = pos + window_size - 1
                signal_indices.append(original_idx)

        # 創建與原始df長度匹配的布爾數組
        result_mask = np.zeros(len(df), dtype=bool)
        for idx in signal_indices:
            if idx < len(df):
                result_mask[idx] = True

        return result_mask

    def get_color(self):
        return "purple"


class BTCMACDBuyDetection(DetectionMethod):
    """BTC MACD買入信號檢測"""

    def __init__(self):
        super().__init__("BTC MACD Buy")

    def detect(self, df):
        n = len(df)
        window_size = 16

        if n < window_size + 1:
            return np.zeros(n, dtype=bool)

        # 檢查必要的欄位是否存在
        required_cols = ['macd_2nd', 'ema_100', 'ema_50', 'close_price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Please call analyze_cumulative_metrics() before detection."
            )
        
        # 直接使用傳入的 df（已包含累計指標）
        analysis_df = df
        signals = np.zeros(n, dtype=bool)

        # 向量化操作
        macd_2nd = analysis_df["macd_2nd"].values
        prev_macd_2nd = np.roll(macd_2nd, 1)

        # 條件：前一期 macd_2nd <= -1，當期 macd_2nd 比前一期增長超過50%
        # 增長50%意思是：curr = prev + |prev| * 0.5
        # 加上 EMA100 > EMA50 > close_price (下跌趨勢中的反彈)
        # 加上價格比48小時高點低超過1%
        ema_condition = (analysis_df["ema_100"] > analysis_df["ema_50"]) & (
            analysis_df["ema_50"] > analysis_df["close_price"]
        )

        # 計算24小時高點和24小時低點 (使用 analysis_df 中的數據)
        high_24h = analysis_df["close_price"].rolling(window=96, min_periods=1).max()
        low_24h = analysis_df["close_price"].rolling(window=96, min_periods=1).min()
        price_drop_condition = analysis_df["close_price"] < high_24h * 0.99
        price_not_too_high_condition = analysis_df["close_price"] < low_24h * 1.005

        condition = (
            (prev_macd_2nd <= -1)
            & (macd_2nd >= prev_macd_2nd + np.abs(prev_macd_2nd) * 0.5)
            & ema_condition.values
            & price_drop_condition.values
            & price_not_too_high_condition.values
        )

        # 跳過第一個元素（因為 roll 後第一個元素無效）
        condition[0] = False

        # 映射到原始 DataFrame 索引
        valid_indices = np.arange(window_size, window_size + len(condition))
        signals[valid_indices[condition]] = True

        return signals

    def get_color(self):
        return "green"


class MacdVolumeDetection(DetectionMethod):
    """MACD Volume檢測"""

    def __init__(self):
        super().__init__("MACD Volume")

    def detect(self, df):
        n = len(df)
        window_size = 16  # ROLLING_WINDOW_SIZE = 4 * 4

        if n < window_size + 1:
            return np.zeros(n, dtype=bool)

        # Calculate MACD (12, 48, 9)
        exp1 = df["close"].ewm(span=12).mean()
        exp2 = df["close"].ewm(span=48).mean()
        macd = exp1 - exp2

        # 4小時滾動平均MACD
        macd_rolling = macd.rolling(window=window_size).mean()

        # Volume trend - exact copy from crypto_analyze.py
        half_window = window_size // 2
        vol_recent = (
            df["volume"]
            .rolling(window=half_window)
            .mean()
            .iloc[window_size - 1 :]
            .values
        )  # 窗口後半段平均
        vol_prev = (
            df["volume"]
            .rolling(window=half_window)
            .mean()
            .shift(half_window)
            .iloc[window_size - 1 :]
            .values
        )  # 窗口前半段平均

        # Create full length array
        volume_trend = np.ones(len(df))
        volume_trend[window_size - 1 :] = np.where(
            vol_prev > 0, vol_recent / vol_prev, 1.0
        )

        # SAR calculation using talib
        import talib

        sar = talib.SAR(df["high"].values, df["low"].values)

        # Price above SAR ratio
        price_above_sar = df["close"] > sar
        price_above_sar_ratio = price_above_sar.rolling(window=window_size).mean()

        # Price ATR ratio condition: (current price - price 1h ago) / ATR > 1.2
        price_atr_ratio = calculate_price_atr_ratio_1h(df)
        price_condition = price_atr_ratio > 1.2

        # Price change percentage (same as crypto_analyze.py - vectorized)
        close_start = df["close"].iloc[: -window_size + 1].values  # 窗口開始價格
        close_end = df["close"].iloc[window_size - 1 :].values  # 窗口結束價格
        price_change_window = (close_end - close_start) / close_start * 100

        # Create full length array
        price_change_pct = np.zeros(len(df))
        price_change_pct[window_size - 1 :] = price_change_window

        # Conditions
        macd_condition = macd_rolling < -90
        volume_condition = volume_trend > 1.2  # Changed from < 1 to > 1.2
        sar_condition = price_above_sar_ratio > 0.8
        price_atr_condition = price_condition
        price_change_condition = price_change_pct > 0

        # Valid range
        valid_mask = np.zeros(n, dtype=bool)
        valid_mask[window_size:] = True

        import pandas as pd

        signals = (
            valid_mask
            & macd_condition
            & volume_condition
            & sar_condition
            & price_atr_condition
            & price_change_condition
            & ~pd.isna(macd_rolling)
            & ~pd.isna(volume_trend)
            & ~pd.isna(price_above_sar_ratio)
        )

        return signals

    def get_color(self):
        return "blue"
