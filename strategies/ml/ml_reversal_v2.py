#!/usr/bin/env python3
"""
ML Reversal V2 策略

基於新的 MLReversalDetector 檢測器的交易策略
使用訓練好的 RandomForest 模型預測反轉入場時機
"""

import pandas as pd
from ..base import BaseStrategy, StrategyConfig
from detection.ml_reversal import MLReversalDetector


class MLReversalV2Strategy(BaseStrategy):
    """
    ML Reversal V2 策略

    基於 MLReversalDetector 的反轉信號
    使用右側入場邏輯：在反轉點後 1-2 根 K 線確認後入場

    特點：
    - 使用 10 個特徵的 RandomForest 模型
    - 右側交易：等待反轉確認後才入場
    - 支持買入和賣出信號

    信號類型：
    - 買入信號 (1): V轉（下降反轉後上漲）
    - 賣出信號 (-1): A轉（上升反轉後下跌）
    """

    def __init__(self, config: StrategyConfig = None):
        """
        初始化策略

        Args:
            config: 策略配置，可包含以下參數：
                - distance: 局部極值的最小距離（默認20）
                - window_size: 滾動窗口大小（默認16）
                - signal_type: 信號類型 'buy', 'sell', 或 'both'（默認'both'）
        """
        if config is None:
            config = StrategyConfig(
                name="ML Reversal V2",
                params={
                    "distance": 20,
                    "window_size": 16,
                    "signal_type": "both",  # 'buy', 'sell', 或 'both'
                },
            )
        super().__init__(config)

        # 創建檢測器
        self.detector = MLReversalDetector(
            extreme_window=self.params.get("distance", 10),
            window_size=self.params.get("window_size", 16),
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信號

        Args:
            df: 包含 OHLCV 數據和所有技術指標的 DataFrame
                必須已經調用 calculate_technical_indicators() 和 analyze_cumulative_metrics()

        Returns:
            pd.Series: 交易信號序列
                      1 = 買入信號（V轉）
                     -1 = 賣出信號（A轉）
                      0 = 無信號
        """
        # 檢查必要的欄位是否存在
        required_cols = ["price_2nd", "vol_2nd", "macd_2nd", "atr_2nd"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Please call calculate_technical_indicators() and analyze_cumulative_metrics() before generating signals."
            )

        # 執行檢測（返回信號類型：0=無信號, 1=A_turn做空, 2=V_turn做多）
        detector_signals = self.detector.detect(df)

        # 初始化信號序列
        signals = pd.Series(0, index=df.index)

        # 獲取信號類型配置
        signal_type = self.params.get("signal_type", "both")

        # 直接使用 detector 的預測結果，轉換為策略信號
        for idx in range(len(df)):
            prediction = detector_signals[idx]

            if prediction == 0:
                continue

            # prediction = 1: A_turn（做空）-> 賣出信號 (-1)
            # prediction = 2: V_turn（做多）-> 買入信號 (1)
            if prediction == 1:  # A_turn
                if signal_type in ["sell", "both"]:
                    signals.iloc[idx] = -1
            elif prediction == 2:  # V_turn
                if signal_type in ["buy", "both"]:
                    signals.iloc[idx] = 1

        return signals

    def get_description(self) -> str:
        """返回策略描述"""
        signal_type = self.params.get("signal_type", "both")
        signal_desc = {"buy": "買入信號", "sell": "賣出信號", "both": "買入+賣出信號"}

        return (
            f"ML Reversal V2 策略 "
            f"(distance={self.params.get('distance', 20)}, "
            f"window={self.params.get('window_size', 16)}, "
            f"{signal_desc.get(signal_type, '雙向信號')})"
        )
