#!/usr/bin/env python3
"""
ML Reversal V6 策略

基於 Random Forest 12 特徵 + SAR 距離過濾的交易策略

訓練結果：
- Random Forest: 70.22% 準確率, ROC AUC 0.7864
- 特徵數: 12 個（變化率特徵）
- 訓練數據: 2024 年完整數據
- 正樣本召回率: 66%

12 個核心特徵（變化率）：
1. price_2nd_change_t_minus_1_to_t
2. price_2nd_change_t_to_entry
3. macd_2nd_change_t_minus_1_to_t
4. vol_2nd_change_t_minus_1_to_t
5. vol_2nd_change_t_to_entry
6. macd_histogram_change_t_to_entry
7. stochrsi_k_change_t_minus_1_to_t
8. sar_distance_pct_change_t_minus_1_to_t
9. sar_distance_pct_change_t_to_entry
10. price_change_3
11. price_change_5
12. price_atr_ratio_1h

過濾條件：
Peak（做空）：
- 價格在 bb_middle 之上
- sar_distance_pct > 1.22%
- [t-1] -> [t] 價格變化 < 0.5%
- entry 價格 < peak 價格

Valley（做多）：
- 價格在 bb_middle 之下
- sar_distance_pct < -1.22%
- [t-1] -> [t] 價格變化 < 0.5%
- entry 價格 > valley 價格
"""

import pandas as pd
from ..base import BaseStrategy, StrategyConfig
from detection.ml_reversal_v6 import MLReversalV6Detector


class MLReversalV6Strategy(BaseStrategy):
    """
    ML Reversal V6 策略 - Random Forest + SAR 距離過濾

    使用 Random Forest 模型，12 個變化率特徵
    加入 SAR 距離和反轉確認過濾條件
    """

    def __init__(self, config: StrategyConfig = None):
        """
        初始化策略

        Args:
            config: 策略配置，可包含以下參數：
                - prob_threshold: 概率閾值（默認0.5）
                - signal_type: 信號類型 'buy', 'sell', 或 'both'（默認'both'）
        """
        if config is None:
            config = StrategyConfig(
                name="ML Reversal V6",
                params={
                    "prob_threshold": 0.5,
                    "signal_type": "both",
                },
            )
        super().__init__(config)

        # 創建檢測器
        prob_threshold = self.params.get("prob_threshold", 0.5)
        self.detector = MLReversalV6Detector(prob_threshold=prob_threshold)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信號

        Args:
            df: 包含 OHLCV 數據和所有技術指標的 DataFrame

        Returns:
            pd.Series: 交易信號序列（1=做多, -1=做空, 0=無信號）
        """
        # 執行檢測
        signal_array = self.detector.detect(df)

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
            f"ML Reversal V6 策略\n"
            f"模型: Random Forest (12 變化率特徵)\n"
            f"準確率: 70.22%, ROC AUC: 0.7864\n"
            f"概率閾值: {self.params.get('prob_threshold', 0.5)}\n"
            f"信號類型: {self.params.get('signal_type', 'both')}\n"
            f"過濾: SAR 距離 + 反轉確認 + 波動過濾"
        )
