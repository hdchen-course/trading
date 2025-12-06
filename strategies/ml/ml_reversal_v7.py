"""
ML Reversal V7 Strategy - PyTorch LSTM
基於 LSTM 的反轉交易策略

使用 PyTorch LSTM 模型預測反轉點並執行交易

策略邏輯：
1. 使用 MLReversalV7Detector 檢測反轉信號
2. 信號 = 1: 做空（peak 後入場）
3. 信號 = 2: 做多（valley 後入場）
4. 止盈：1.0%
5. 止損：0.18%
6. 最大持倉時間：16 根 K 線（4 小時）
"""

import pandas as pd
from ..base import BaseStrategy, StrategyConfig
from detection.ml_reversal_v7 import MLReversalV7Detector


class MLReversalV7Strategy(BaseStrategy):
    """ML Reversal V7 策略 - PyTorch LSTM"""

    def __init__(self, config: StrategyConfig):
        """
        初始化策略

        Args:
            config: 策略配置
                - prob_threshold: 預測概率閾值（默認 0.5）
                - model_path: LSTM 模型路徑（可選）
                - take_profit_pct: 止盈百分比（默認 1.0）
                - stop_loss_pct: 止損百分比（默認 0.18）
                - max_holding_bars: 最大持倉 K 線數（默認 16）
        """
        super().__init__(config)

        # 從配置中獲取參數
        prob_threshold = config.params.get("prob_threshold", 0.5)
        model_path = config.params.get(
            "model_path", "models/reversal/reversal_model_v7_lstm.pth"
        )
        self.take_profit_pct = config.params.get("take_profit_pct", 1.0)
        self.stop_loss_pct = config.params.get("stop_loss_pct", 0.18)
        self.max_holding_bars = config.params.get("max_holding_bars", 16)

        # 初始化檢測器
        self.detector = MLReversalV7Detector(
            model_path=model_path, prob_threshold=prob_threshold
        )

        print("✅ ML Reversal V7 策略初始化")
        print(f"   概率閾值: {prob_threshold}")
        print(f"   止盈: {self.take_profit_pct}%")
        print(f"   止損: {self.stop_loss_pct}%")
        print(f"   最大持倉: {self.max_holding_bars} 根 K 線")

    def generate_signals(self, df):
        """
        生成交易信號

        Args:
            df: 包含 OHLCV 和技術指標的 DataFrame

        Returns:
            signals: 信號 Series
            - 0: 無信號
            - 1: 做空信號
            - 2: 做多信號
        """
        signals_array = self.detector.detect(df)
        return pd.Series(signals_array, index=df.index)

    def should_enter(self, df, idx, position):
        """
        判斷是否應該入場

        Args:
            df: DataFrame
            idx: 當前索引
            position: 當前持倉狀態

        Returns:
            (should_enter, direction)
            - should_enter: 是否入場
            - direction: 'long' 或 'short'
        """
        # 如果已有持倉，不入場
        if position is not None:
            return False, None

        # 獲取信號
        signals = self.generate_signals(df.iloc[: idx + 1])
        signal = signals[idx]

        if signal == 1:
            return True, "short"
        elif signal == 2:
            return True, "long"
        else:
            return False, None

    def should_exit(self, df, idx, position):
        """
        判斷是否應該出場

        Args:
            df: DataFrame
            idx: 當前索引
            position: 當前持倉信息

        Returns:
            (should_exit, reason)
        """
        if position is None:
            return False, None

        entry_price = position["entry_price"]
        entry_idx = position["entry_idx"]
        direction = position["direction"]
        current_price = df.iloc[idx]["close"]

        # 計算盈虧百分比
        if direction == "long":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:  # short
            pnl_pct = (entry_price - current_price) / entry_price * 100

        # 止盈
        if pnl_pct >= self.take_profit_pct:
            return True, "take_profit"

        # 止損
        if pnl_pct <= -self.stop_loss_pct:
            return True, "stop_loss"

        # 最大持倉時間
        holding_bars = idx - entry_idx
        if holding_bars >= self.max_holding_bars:
            return True, "max_holding"

        return False, None

    def get_description(self) -> str:
        """返回策略描述"""
        prob_threshold = self.config.params.get("prob_threshold", 0.5)
        return (
            f"ML Reversal V7 策略 (PyTorch LSTM)\n"
            f"模型: LSTM (1層, hidden_size=32, seq_len=30)\n"
            f"特徵數: 44 個（移除 Bottom 5 有害特徵）\n"
            f"測試 AUC: 0.8520, 準確率: 77.84%\n"
            f"概率閾值: {prob_threshold}\n"
            f"止盈: {self.take_profit_pct}%, 止損: {self.stop_loss_pct}%\n"
            f"最大持倉: {self.max_holding_bars} 根 K 線"
        )
