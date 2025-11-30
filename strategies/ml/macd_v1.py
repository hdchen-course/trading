#!/usr/bin/env python3
"""
MACD V1 策略

基於 MACD 交叉點的機器學習策略
使用訓練好的 Random Forest 模型預測交叉後的反轉概率

訓練邏輯：
- 檢測 MACD 黃金交叉/死亡交叉
- 過濾條件：黃金交叉時 EMA100 > EMA50，死亡交叉時 EMA100 < EMA50
- 入場點：[t+1]（交叉後一根K線）
- 止損：1小時內 -0.18%
- 止盈：2.5小時內 >= 0.4%
- 測試準確率：56.01%
- ROC AUC：0.5808
"""

import pandas as pd
from ..base import BaseStrategy, StrategyConfig
from detection.macd_v1 import MACDV1Detector


class MACDV1Strategy(BaseStrategy):
    """
    MACD V1 策略

    基於 MACD 交叉點的機器學習預測
    使用 Random Forest 模型預測交叉後的反轉概率

    特點：
    - 檢測 MACD 黃金交叉/死亡交叉
    - 使用 EMA50/EMA100 關係過濾信號
    - 支持概率閾值調整
    - 支持買入和賣出信號

    信號類型：
    - 買入信號 (1): 黃金交叉（MACD 線穿越信號線向上）
    - 賣出信號 (-1): 死亡交叉（MACD 線穿越信號線向下）
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
                name="MACD V1",
                params={
                    "prob_threshold": 0.5,
                    "signal_type": "both",  # 'buy', 'sell', 或 'both'
                },
            )
        super().__init__(config)

        # 創建檢測器
        self.detector = MACDV1Detector(
            model_path="models/macd/macd_model_v1_rf_top5.pkl"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信號

        Args:
            df: 包含 OHLCV 數據和所有技術指標的 DataFrame
                必須已經調用 calculate_technical_indicators()

        Returns:
            pd.Series: 交易信號序列
                      1 = 買入信號（黃金交叉）
                     -1 = 賣出信號（死亡交叉）
                      0 = 無信號
        """
        # 檢查必要的欄位是否存在
        required_cols = ["macd", "macd_signal", "ema_50", "ema_100", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Please call calculate_technical_indicators() before generating signals."
            )

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
            # 需要至少2個點來判斷交叉方向
            if idx < 1:
                continue

            # 檢查 MACD 交叉方向
            macd_prev = df["macd"].iloc[idx - 1]
            macd_curr = df["macd"].iloc[idx]
            signal_prev = df["macd_signal"].iloc[idx - 1]
            signal_curr = df["macd_signal"].iloc[idx]

            # 黃金交叉：MACD 從下方穿越信號線 -> 買入信號
            if macd_prev < signal_prev and macd_curr > signal_curr:
                if signal_type in ["buy", "both"]:
                    signals.iloc[idx] = 1

            # 死亡交叉：MACD 從上方穿越信號線 -> 賣出信號
            elif macd_prev > signal_prev and macd_curr < signal_curr:
                if signal_type in ["sell", "both"]:
                    signals.iloc[idx] = -1

        return signals

    def get_description(self) -> str:
        """返回策略描述"""
        signal_type = self.params.get("signal_type", "both")
        prob_threshold = self.params.get("prob_threshold", 0.5)

        signal_desc = {"buy": "買入信號", "sell": "賣出信號", "both": "買入+賣出信號"}

        return (
            f"MACD V1 策略 "
            f"(prob_threshold={prob_threshold:.2f}, "
            f"{signal_desc.get(signal_type, '雙向信號')})"
        )
