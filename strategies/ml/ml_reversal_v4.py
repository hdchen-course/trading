#!/usr/bin/env python3
"""
ML Reversal V4 策略

基於 V4 多模型集成檢測器的交易策略
使用 RF + LightGBM + CatBoost 加權投票

訓練結果：
- Random Forest: 63.88% 準確率, ROC AUC 0.6901
- LightGBM: 59.73% 準確率, ROC AUC 0.6504
- CatBoost: 60.73% 準確率, ROC AUC 0.6495

加權投票：
- RF 權重: 35.2%
- LGB 權重: 33.2%
- CAT 權重: 33.1%
"""

import pandas as pd
from ..base import BaseStrategy, StrategyConfig
from detection.ml_reversal_v4 import MLReversalV4Detector


class MLReversalV4Strategy(BaseStrategy):
    """
    ML Reversal V4 策略 - 三模型加權投票

    使用 Random Forest + LightGBM + CatBoost 加權投票
    權重基於各模型的 ROC AUC
    """

    def __init__(self, config: StrategyConfig = None):
        """
        初始化策略

        Args:
            config: 策略配置，可包含以下參數：
                - prob_threshold: 加權概率閾值（默認0.5）
                - signal_type: 信號類型 'buy', 'sell', 或 'both'（默認'both'）
        """
        if config is None:
            config = StrategyConfig(
                name="ML Reversal V4",
                params={
                    "prob_threshold": 0.5,
                    "signal_type": "both",
                },
            )
        super().__init__(config)

        # 創建檢測器
        self.detector = MLReversalV4Detector()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信號（使用加權投票）

        Args:
            df: 包含 OHLCV 數據和所有技術指標的 DataFrame

        Returns:
            pd.Series: 交易信號序列
        """
        # 執行檢測
        prob_threshold = self.params.get("prob_threshold", 0.5)
        signal_mask = self.detector.detect(df, prob_threshold=prob_threshold)

        # 初始化信號序列
        signals = pd.Series(0, index=df.index)

        # 獲取信號索引
        signal_indices = signal_mask.nonzero()[0]

        if len(signal_indices) == 0:
            return signals

        # 判斷每個信號是買入還是賣出
        signal_type = self.params.get("signal_type", "both")

        for idx in signal_indices:
            if idx < 3 or idx >= len(df) - 1:
                continue

            price_before = df["close"].iloc[idx - 3 : idx].mean()
            price_at = df["close"].iloc[idx]

            # 谷值反轉：買入信號
            if price_at > price_before:
                if signal_type in ["buy", "both"]:
                    signals.iloc[idx] = 1

            # 峰值反轉：賣出信號
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
            f"ML Reversal V4 策略 (RF+LGB+CAT 加權投票, "
            f"prob={prob_threshold:.2f}, "
            f"{signal_desc.get(signal_type, '雙向信號')})"
        )
