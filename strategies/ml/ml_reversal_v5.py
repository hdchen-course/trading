#!/usr/bin/env python3
"""
ML Reversal V5 策略

基於 Random Forest Top 16 特徵的交易策略
使用全部 138 個特徵訓練後精選的 16 個最優特徵

訓練結果：
- Random Forest (Top 16): 60.15% 準確率, ROC AUC 0.6184
- 特徵數: 16 個（從 138 個全特徵中精選）
- 訓練數據: 2024 年完整數據

16 個核心特徵：
- price_2nd, amplitude, volume, rsi, volatility
- price_atr_ratio_1h, macd_2nd, price_change_3
- avg_volume, volume_change, stochrsi_d
- avg_amplitude, atr_2nd
"""

import pandas as pd
from ..base import BaseStrategy, StrategyConfig
from detection.ml_reversal_v5 import MLReversalV5Detector


class MLReversalV5Strategy(BaseStrategy):
    """
    ML Reversal V5 策略 - Random Forest Top 16

    使用 Random Forest 模型，16 個精選特徵
    基於峰值/谷值分開處理的標籤生成邏輯
    """

    def __init__(self, config: StrategyConfig = None):
        """
        初始化策略

        Args:
            config: 策略配置，可包含以下參數：
                - prob_threshold: 概率閾值（默認0.5）
                - signal_type: 信號類型 'buy', 'sell', 或 'both'（默認'both'）
                - stochrsi_threshold_high: StochRSI 高閾值（默認90）
                - stochrsi_threshold_low: StochRSI 低閾值（默認10）
        """
        if config is None:
            config = StrategyConfig(
                name="ML Reversal V5",
                params={
                    "prob_threshold": 0.5,
                    "signal_type": "both",
                    "stochrsi_threshold_high": 90,
                    "stochrsi_threshold_low": 10,
                },
            )
        super().__init__(config)

        # 創建檢測器
        self.detector = MLReversalV5Detector()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信號

        Args:
            df: 包含 OHLCV 數據和所有技術指標的 DataFrame

        Returns:
            pd.Series: 交易信號序列（1=做多, -1=做空, 0=無信號）
        """
        # 執行檢測
        prob_threshold = self.params.get("prob_threshold", 0.5)
        stochrsi_high = self.params.get("stochrsi_threshold_high", 90)
        stochrsi_low = self.params.get("stochrsi_threshold_low", 10)

        signal_array = self.detector.detect(
            df,
            prob_threshold=prob_threshold,
            stochrsi_threshold_high=stochrsi_high,
            stochrsi_threshold_low=stochrsi_low,
            debug=False,
        )

        # 轉換為 pd.Series
        signals = pd.Series(0, index=df.index)

        signal_type = self.params.get("signal_type", "both")

        for idx in range(len(signal_array)):
            if signal_array[idx] == 1:  # 做空信號
                if signal_type in ["sell", "both"]:
                    signals.iloc[idx] = -1
            elif signal_array[idx] == 2:  # 做多信號
                if signal_type in ["buy", "both"]:
                    signals.iloc[idx] = 1

        return signals

    def get_description(self) -> str:
        """返回策略描述"""
        return (
            f"ML Reversal V5 策略\n"
            f"模型: Random Forest (Top 16 特徵)\n"
            f"準確率: 60.15%, ROC AUC: 0.6184\n"
            f"概率閾值: {self.params.get('prob_threshold', 0.5)}\n"
            f"信號類型: {self.params.get('signal_type', 'both')}\n"
            f"StochRSI 高閾值: {self.params.get('stochrsi_threshold_high', 90)}\n"
            f"StochRSI 低閾值: {self.params.get('stochrsi_threshold_low', 10)}"
        )
