#!/usr/bin/env python3
"""
Triple Confirmation Reversal 策略

基於三重確認反轉檢測器的交易策略
使用 Price_2nd, Vol_2nd, MACD_2nd 三個二階導數特徵
"""

import pandas as pd

from ..base import BaseStrategy, StrategyConfig
from detection.triple_confirmation_reversal import TripleConfirmationReversalDetector


class TripleConfirmationStrategy(BaseStrategy):
    """
    Triple Confirmation Reversal 策略

    基於 TripleConfirmationReversalDetector 的反轉信號

    特點：
    - 精確度：81.2%
    - 覆蓋率：40.4%
    - 每天約1.2個訊號
    - 使用兩階段檢測：局部極值 + 三重確認過濾

    信號類型：
    - 買入信號 (1): 檢測到谷底反轉（價格即將上漲）
    - 賣出信號 (-1): 檢測到峰頂反轉（價格即將下跌）
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
                name="Triple Confirmation Reversal",
                params={
                    "distance": 20,
                    "window_size": 16,
                    "signal_type": "both",  # 'buy', 'sell', 或 'both'
                },
            )
        super().__init__(config)

        # 創建檢測器
        self.detector = TripleConfirmationReversalDetector(
            extreme_window=self.params.get("distance", 20),
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
                      1 = 買入信號（谷底反轉）
                     -1 = 賣出信號（峰頂反轉）
                      0 = 無信號
        """
        # 檢查必要的欄位是否存在
        required_cols = ["price_2nd", "vol_2nd", "macd_2nd"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Please call calculate_technical_indicators() and analyze_cumulative_metrics() before generating signals."
            )

        # 執行檢測（返回信號類型：0=無信號, 1=峰值做空, 2=谷值做多）
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

            # prediction = 1: 峰值（做空）-> 賣出信號 (-1)
            # prediction = 2: 谷值（做多）-> 買入信號 (1)
            if prediction == 1:  # 峰值
                if signal_type in ["sell", "both"]:
                    signals.iloc[idx] = -1
            elif prediction == 2:  # 谷值
                if signal_type in ["buy", "both"]:
                    signals.iloc[idx] = 1

        return signals

    def get_description(self) -> str:
        """返回策略描述"""
        signal_type = self.params.get("signal_type", "both")
        signal_desc = {"buy": "買入信號", "sell": "賣出信號", "both": "買入+賣出信號"}

        return (
            f"Triple Confirmation Reversal 策略 "
            f"(distance={self.params.get('distance', 20)}, "
            f"window={self.params.get('window_size', 16)}, "
            f"{signal_desc.get(signal_type, '雙向信號')})"
        )


class TripleConfirmationBuyStrategy(BaseStrategy):
    """
    Triple Confirmation 買入策略

    只在谷底反轉時產生買入信號
    適合做多策略
    """

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="Triple Confirmation Buy",
                params={"distance": 20, "window_size": 16, "signal_type": "buy"},
            )
        else:
            # 強制設置為只買入
            config.params["signal_type"] = "buy"

        super().__init__(config)

        self.detector = TripleConfirmationReversalDetector(
            distance=self.params.get("distance", 20),
            window_size=self.params.get("window_size", 16),
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """生成買入信號"""
        # 檢查必要的欄位是否存在
        required_cols = ["price_2nd", "vol_2nd", "macd_2nd"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Please call calculate_technical_indicators() and analyze_cumulative_metrics() before generating signals."
            )

        # 執行檢測（返回信號類型：0=無信號, 1=峰值做空, 2=谷值做多）
        detector_signals = self.detector.detect(df)

        signals = pd.Series(0, index=df.index)

        # 只在谷值（prediction=2）時產生買入信號
        for idx in range(len(df)):
            if detector_signals[idx] == 2:  # 谷值
                signals.iloc[idx] = 1

        return signals

    def get_description(self) -> str:
        return "Triple Confirmation Buy 策略（只做多）"


class TripleConfirmationSellStrategy(BaseStrategy):
    """
    Triple Confirmation 賣出策略

    只在峰頂反轉時產生賣出信號
    適合做空或止盈策略
    """

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="Triple Confirmation Sell",
                params={"distance": 20, "window_size": 16, "signal_type": "sell"},
            )
        else:
            # 強制設置為只賣出
            config.params["signal_type"] = "sell"

        super().__init__(config)

        self.detector = TripleConfirmationReversalDetector(
            distance=self.params.get("distance", 20),
            window_size=self.params.get("window_size", 16),
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """生成賣出信號"""
        # 檢查必要的欄位是否存在
        required_cols = ["price_2nd", "vol_2nd", "macd_2nd"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Please call calculate_technical_indicators() and analyze_cumulative_metrics() before generating signals."
            )

        # 執行檢測（返回信號類型：0=無信號, 1=峰值做空, 2=谷值做多）
        detector_signals = self.detector.detect(df)

        signals = pd.Series(0, index=df.index)

        # 只在峰值（prediction=1）時產生賣出信號
        for idx in range(len(df)):
            if detector_signals[idx] == 1:  # 峰值
                signals.iloc[idx] = -1

        return signals

    def get_description(self) -> str:
        return "Triple Confirmation Sell 策略（只做空）"
