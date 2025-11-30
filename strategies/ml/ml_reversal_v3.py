#!/usr/bin/env python3
"""
ML Reversal V3 策略

基於新的 MLReversalV3Detector 檢測器的交易策略
使用訓練好的 XGBoost 模型（Top 15 特徵）預測反轉入場時機

訓練邏輯：
- 使用極值檢測找反轉點
- 基於止損止盈的標籤生成
- 樣本平衡 1:1
- 測試準確率 65.17%
- ROC AUC 0.7083
"""

import pandas as pd
from ..base import BaseStrategy, StrategyConfig
from detection.ml_reversal_v3 import MLReversalV3Detector


class MLReversalV3Strategy(BaseStrategy):
    """
    ML Reversal V3 策略

    基於 MLReversalV3Detector 的反轉信號
    使用極值檢測 + XGBoost 模型預測入場點

    特點：
    - 使用 Top 15 特徵的 XGBoost 模型
    - 基於極值檢測的反轉點識別
    - 支持概率閾值調整
    - 支持買入和賣出信號

    信號類型：
    - 買入信號 (1): 谷值反轉後上漲
    - 賣出信號 (-1): 峰值反轉後下跌
    """

    def __init__(self, config: StrategyConfig = None):
        """
        初始化策略

        Args:
            config: 策略配置，可包含以下參數：
                - prob_threshold: 預測概率閾值（默認0.5）
                - signal_type: 信號類型 'buy', 'sell', 或 'both'（默認'both'）
        """
        if config is None:
            config = StrategyConfig(
                name="ML Reversal V3",
                params={
                    "prob_threshold": 0.5,
                    "signal_type": "both",  # 'buy', 'sell', 或 'both'
                },
            )
        super().__init__(config)

        # 創建檢測器
        self.detector = MLReversalV3Detector(
            prob_threshold=self.params.get("prob_threshold", 0.5)
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信號

        Args:
            df: 包含 OHLCV 數據和所有技術指標的 DataFrame
                必須已經調用 calculate_technical_indicators()

        Returns:
            pd.Series: 交易信號序列
                      1 = 買入信號（谷值反轉）
                     -1 = 賣出信號（峰值反轉）
                      0 = 無信號
        """
        # 檢查必要的欄位是否存在（Top 15 特徵需要的基礎欄位）
        required_cols = [
            "ema_50",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "price_change_pct",
            "low",
            "open",
            "close",
            "sar",
            "avg_amplitude",
            "macd_signal",
            "atr",
            "volume_sma",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Please call calculate_technical_indicators() before generating signals."
            )

        # 執行檢測
        signal_mask = self.detector.detect(df)

        # 初始化信號序列
        signals = pd.Series(0, index=df.index)

        # 獲取信號索引
        signal_indices = signal_mask.nonzero()[0]

        if len(signal_indices) == 0:
            return signals

        # 判斷每個信號是買入還是賣出
        signal_type = self.params.get("signal_type", "both")

        for idx in signal_indices:
            # 需要至少3個點來判斷趨勢
            if idx < 3 or idx >= len(df) - 1:
                continue

            # 檢查信號點前後的價格變化
            # 入場點是 [t+1] 或 [t+2]，所以我們看入場點前的極值點
            price_before = df["close"].iloc[idx - 3 : idx].mean()
            price_at = df["close"].iloc[idx]

            # 谷值反轉：價格從低點反彈 -> 買入信號
            if price_at > price_before:
                if signal_type in ["buy", "both"]:
                    signals.iloc[idx] = 1

            # 峰值反轉：價格從高點回落 -> 賣出信號
            elif price_at < price_before:
                if signal_type in ["sell", "both"]:
                    signals.iloc[idx] = -1

        return signals

    def get_description(self) -> str:
        """返回策略描述"""
        signal_type = self.params.get("signal_type", "both")
        prob_threshold = self.params.get("prob_threshold", 0.5)

        signal_desc = {"buy": "買入信號", "sell": "賣出信號", "both": "買入+賣出信號"}

        return (
            f"ML Reversal V3 策略 "
            f"(prob_threshold={prob_threshold:.2f}, "
            f"{signal_desc.get(signal_type, '雙向信號')})"
        )


class MLReversalV3BuyStrategy(BaseStrategy):
    """
    ML Reversal V3 買入策略

    只在谷值反轉時產生買入信號
    適合做多策略
    """

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="ML Reversal V3 Buy",
                params={"prob_threshold": 0.5, "signal_type": "buy"},
            )
        else:
            # 強制設置為只買入
            config.params["signal_type"] = "buy"

        super().__init__(config)

        self.detector = MLReversalV3Detector(
            prob_threshold=self.params.get("prob_threshold", 0.5)
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """生成買入信號"""
        # 檢查必要的欄位
        required_cols = [
            "ema_50",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "price_change_pct",
            "low",
            "open",
            "close",
            "sar",
            "avg_amplitude",
            "macd_signal",
            "atr",
            "volume_sma",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Please call calculate_technical_indicators() before generating signals."
            )

        signal_mask = self.detector.detect(df)

        signals = pd.Series(0, index=df.index)
        signal_indices = signal_mask.nonzero()[0]

        for idx in signal_indices:
            if idx < 3 or idx >= len(df) - 1:
                continue

            price_before = df["close"].iloc[idx - 3 : idx].mean()
            price_at = df["close"].iloc[idx]

            # 只在谷值反轉時產生買入信號
            if price_at > price_before:
                signals.iloc[idx] = 1

        return signals

    def get_description(self) -> str:
        prob_threshold = self.params.get("prob_threshold", 0.5)
        return f"ML Reversal V3 Buy 策略（只做多，prob={prob_threshold:.2f}）"


class MLReversalV3SellStrategy(BaseStrategy):
    """
    ML Reversal V3 賣出策略

    只在峰值反轉時產生賣出信號
    適合做空或止盈策略
    """

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="ML Reversal V3 Sell",
                params={"prob_threshold": 0.5, "signal_type": "sell"},
            )
        else:
            # 強制設置為只賣出
            config.params["signal_type"] = "sell"

        super().__init__(config)

        self.detector = MLReversalV3Detector(
            prob_threshold=self.params.get("prob_threshold", 0.5)
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """生成賣出信號"""
        # 檢查必要的欄位
        required_cols = [
            "ema_50",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "price_change_pct",
            "low",
            "open",
            "close",
            "sar",
            "avg_amplitude",
            "macd_signal",
            "atr",
            "volume_sma",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Please call calculate_technical_indicators() before generating signals."
            )

        signal_mask = self.detector.detect(df)

        signals = pd.Series(0, index=df.index)
        signal_indices = signal_mask.nonzero()[0]

        for idx in signal_indices:
            if idx < 3 or idx >= len(df) - 1:
                continue

            price_before = df["close"].iloc[idx - 3 : idx].mean()
            price_at = df["close"].iloc[idx]

            # 只在峰值反轉時產生賣出信號
            if price_at < price_before:
                signals.iloc[idx] = -1

        return signals

    def get_description(self) -> str:
        prob_threshold = self.params.get("prob_threshold", 0.5)
        return f"ML Reversal V3 Sell 策略（只做空，prob={prob_threshold:.2f}）"
