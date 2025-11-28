#!/usr/bin/env python3
"""
MACD 相關策略

包含所有基於 MACD 指標的交易策略
"""

import pandas as pd
import pandas_ta as ta

from ..base import BaseStrategy, StrategyConfig
from detection import MacdReversalDetection, BTCMACDBuyDetection, MacdVolumeDetection
from utils.indicators import calculate_technical_indicators


class MACDStrategy(BaseStrategy):
    """
    MACD 交叉策略

    基於 MACD 指標的金叉死叉信號:
    - 金叉 (MACD > Signal): 買入信號
    - 死叉 (MACD < Signal): 賣出信號
    """

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="MACD Cross",
                params={"fast": 12, "slow": 32, "signal": 9},
            )
        super().__init__(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        macd = ta.macd(
            df["close"],
            fast=self.params["fast"],
            slow=self.params["slow"],
            signal=self.params["signal"],
        )

        signals = pd.Series(0, index=df.index)

        macd_line = macd.iloc[:, 0]
        signal_line = macd.iloc[:, 1]

        # 金叉條件
        buy_condition = (macd_line > signal_line) & (
            macd_line.shift(1) <= signal_line.shift(1)
        )
        signals[buy_condition] = 1

        # 死叉條件
        sell_condition = (macd_line < signal_line) & (
            macd_line.shift(1) >= signal_line.shift(1)
        )
        signals[sell_condition] = -1

        return signals

    def get_description(self) -> str:
        return (
            f"MACD({self.params['fast']},{self.params['slow']},"
            f"{self.params['signal']}) 交叉策略"
        )


class MACDVolumeStrategy(BaseStrategy):
    """
    MACD + 成交量確認策略

    結合 MACD 金叉和成交量確認:
    - 買入: MACD 金叉 + 成交量 > 均量 * 倍數
    - 賣出: MACD 死叉
    """

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="MACD Volume",
                params={
                    "fast": 10,
                    "slow": 21,
                    "signal": 5,
                    "volume_period": 20,
                    "volume_multiplier": 1.2,
                },
            )
        super().__init__(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # 計算 MACD
        macd = ta.macd(
            df["close"],
            fast=self.params["fast"],
            slow=self.params["slow"],
            signal=self.params["signal"],
        )

        # 計算成交量均線
        volume_sma = ta.sma(df["volume"], length=self.params["volume_period"])

        signals = pd.Series(0, index=df.index)

        macd_line = macd.iloc[:, 0]
        signal_line = macd.iloc[:, 1]

        # 金叉條件
        golden_cross = (macd_line > signal_line) & (
            macd_line.shift(1) <= signal_line.shift(1)
        )

        # 成交量確認
        volume_confirm = df["volume"] > volume_sma * self.params["volume_multiplier"]

        # 買入信號
        buy_condition = golden_cross & volume_confirm
        signals[buy_condition] = 1

        # 死叉條件
        sell_condition = (macd_line < signal_line) & (
            macd_line.shift(1) >= signal_line.shift(1)
        )
        signals[sell_condition] = -1

        return signals

    def get_description(self) -> str:
        return (
            f"MACD({self.params['fast']},{self.params['slow']},"
            f"{self.params['signal']}) + Volume"
        )


class MacdReversalStrategy(BaseStrategy):
    """
    MACD Reversal 策略

    基於 MacdReversalDetection 的反轉信號
    """

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(name="MACD Reversal", params={})
        super().__init__(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # 檢查必要的欄位是否存在
        required_cols = ['macd_2nd', 'price_2nd', 'atr_2nd', 'price_atr_ratio_1h', 'price_change_pct']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Please call calculate_technical_indicators() and analyze_cumulative_metrics() before generating signals."
            )
        
        detector = MacdReversalDetection()
        signal_mask = detector.detect(df)

        signals = pd.Series(0, index=df.index)
        signals[signal_mask] = -1

        return signals

    def get_description(self) -> str:
        return "MACD Reversal 反轉策略"


class MacdVolumeStrategy(BaseStrategy):
    """
    MACD Volume 策略

    基於 MacdVolumeDetection 的賣出信號
    """

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(name="MACD Volume", params={})
        super().__init__(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = calculate_technical_indicators(df)
        
        # MacdVolumeDetection 不需要 analyze_cumulative_metrics
        detector = MacdVolumeDetection()
        signal_mask = detector.detect(df)

        signals = pd.Series(0, index=df.index)
        signals[signal_mask] = -1

        return signals

    def get_description(self) -> str:
        return "MACD Volume 策略"


class BTCMACDBuyStrategy(BaseStrategy):
    """
    BTC MACD 買入策略

    基於 BTCMACDBuyDetection 的買入信號
    """

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(name="BTC MACD Buy", params={})
        super().__init__(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = calculate_technical_indicators(df)
        detector = BTCMACDBuyDetection()
        signal_mask = detector.detect(df)

        signals = pd.Series(0, index=df.index)
        signals[signal_mask] = 1

        return signals

    def get_description(self) -> str:
        return "BTC MACD Buy 策略"
