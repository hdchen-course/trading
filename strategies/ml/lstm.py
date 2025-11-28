#!/usr/bin/env python3
"""
LSTM 策略
"""

import pandas as pd

from ..base import BaseStrategy, StrategyConfig
from detection import LSTMDetection
from utils.indicators import calculate_technical_indicators


class LSTMStrategy(BaseStrategy):
    """
    LSTM 反轉策略

    基於 LSTM 模型的反轉信號
    """

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(name="LSTM", params={})
        super().__init__(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # 檢查必要的欄位是否存在
        required_cols = ['avg_macd', 'avg_rsi', 'volatility', 'macd_2nd', 'price_2nd', 'vol_2nd', 'atr_2nd']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Please call calculate_technical_indicators() and analyze_cumulative_metrics() before generating signals."
            )
        
        detector = LSTMDetection()
        signal_mask = detector.detect(df)

        signals = pd.Series(0, index=df.index)
        signals[signal_mask] = -1

        return signals

    def get_description(self) -> str:
        return "LSTM 反轉策略 (雙向 LSTM 模型預測反轉點)"
