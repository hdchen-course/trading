#!/usr/bin/env python3
"""
Peak Valley Detection Methods
波峰波谷檢測方法

⚠️ 重要說明：
這個檢測器是用於「事後分析」和「自動標籤」的工具，不是用於實時交易！

用途：
1. 自動標記歷史數據中的真實反轉點（Ground Truth）
2. 為機器學習模型生成訓練標籤
3. 評估其他檢測方法的準確度和召回率

特點：
- 使用 scipy.signal.find_peaks 找出價格的局部極值
- 需要完整的歷史數據才能準確識別波峰波谷
- 無法在實時交易中使用（因為需要看到未來的價格才能確認極值）

實時交易應使用：
- TripleConfirmationReversalDetector
- MLReversalDetector
- 或其他前瞻性檢測方法
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .base import DetectionMethod


class PeakValleyDetection(DetectionMethod):
    """
    波峰波谷檢測 - 使用 scipy.signal 自動找 A 轉 V 轉點

    ⚠️ 這是事後分析工具，用於自動標記真實反轉點（Ground Truth）
    不適用於實時交易，因為需要看到未來價格才能確認極值點
    """

    def __init__(self, distance=20, prominence=None, height_threshold=0.02):
        """
        初始化波峰波谷檢測

        Args:
            distance: 波峰/波谷之間的最小距離 (數據點數)
            prominence: 最小突出度 (價格變化幅度)
            height_threshold: 高度閾值 (相對價格變化百分比)
        """
        super().__init__("Peak Valley Detection")
        self.distance = distance
        self.prominence = prominence
        self.height_threshold = height_threshold

    def detect(self, df: pd.DataFrame) -> np.ndarray:
        """
        檢測波峰波谷信號

        Args:
            df: 包含 OHLCV 數據的 DataFrame

        Returns:
            布爾數組，True 表示該點為反轉點
        """
        if len(df) < self.distance * 2:
            return np.zeros(len(df), dtype=bool)

        prices = df["close"].values

        # 動態計算prominence (如果未設定)
        # 使用滾動窗口計算局部標準差，避免長時間序列的問題
        if self.prominence is None:
            # 使用較短的窗口（例如 500 個數據點，約 5 天的 15 分鐘數據）
            # 來計算局部波動性，而不是整個數據集的標準差
            window_size = min(500, len(prices) // 4)  # 最多使用 1/4 的數據作為窗口
            if window_size < 100:
                window_size = len(prices)  # 如果數據太少，使用全部

            # 計算滾動標準差的中位數作為 prominence
            rolling_std = (
                pd.Series(prices).rolling(window=window_size, min_periods=1).std()
            )
            prominence = rolling_std.median() * 0.5
        else:
            prominence = self.prominence

        # 找波谷 (A轉V的轉折點 - 買入信號)
        valleys, valley_props = find_peaks(
            -prices, distance=self.distance, prominence=prominence
        )

        # 找波峰 (V轉A的轉折點 - 賣出信號)
        peaks, peak_props = find_peaks(
            prices, distance=self.distance, prominence=prominence
        )

        # 創建布林數組
        result = np.zeros(len(df), dtype=bool)

        # 標記顯著的波谷
        for valley_idx in valleys:
            if valley_idx < len(df):
                valley_price = prices[valley_idx]

                # 計算相對高度變化
                if valley_idx > 0 and valley_idx < len(prices) - 1:
                    left_price = prices[max(0, valley_idx - self.distance // 2)]
                    right_price = prices[
                        min(len(prices) - 1, valley_idx + self.distance // 2)
                    ]
                    height_change = min(
                        (left_price - valley_price) / valley_price,
                        (right_price - valley_price) / valley_price,
                    )

                    # 只保留顯著的波谷
                    if height_change >= self.height_threshold:
                        result[valley_idx] = True

        # 標記顯著的波峰
        for peak_idx in peaks:
            if peak_idx < len(df):
                peak_price = prices[peak_idx]

                # 計算相對高度變化
                if peak_idx > 0 and peak_idx < len(prices) - 1:
                    left_price = prices[max(0, peak_idx - self.distance // 2)]
                    right_price = prices[
                        min(len(prices) - 1, peak_idx + self.distance // 2)
                    ]
                    height_change = min(
                        (peak_price - left_price) / left_price,
                        (peak_price - right_price) / right_price,
                    )

                    # 只保留顯著的波峰
                    if height_change >= self.height_threshold:
                        result[peak_idx] = True

        # 打印反轉點信息（簡化版，不依賴 analyze_cumulative_metrics）
        reversal_indices = [i for i, is_reversal in enumerate(result) if is_reversal]

        if reversal_indices:
            print(f"{'時間':<20} {'價格':<10}")
            print("-" * 30)

            for idx in reversal_indices:
                timestamp = df.iloc[idx].get(
                    "timestamp",
                    df.index[idx] if hasattr(df.index, "__getitem__") else "N/A",
                )
                close_price = round(float(df.iloc[idx]["close"]), 2)
                print(f"{str(timestamp):<20} {close_price:<10}")

        return result

    def get_color(self):
        return "red"

    def get_marker(self):
        return "D"  # 菱形標記
