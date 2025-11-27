#!/usr/bin/env python3
"""
K線型態策略
"""

import pandas as pd
import pandas_ta as ta

from ..base import BaseStrategy, StrategyConfig


class CandlestickStrategy(BaseStrategy):
    """K線型態策略 (反轉 + 趨勢持續型態)"""

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="Candlestick Patterns",
                params={"threshold": 160},
            )
        super().__init__(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)

        patterns = {}

        # 看漲型態
        patterns["hammer"] = ta.cdl_pattern(
            df["open"], df["high"], df["low"], df["close"], name="hammer"
        )
        patterns["engulfing"] = ta.cdl_pattern(
            df["open"], df["high"], df["low"], df["close"], name="engulfing"
        )
        patterns["morningstar"] = ta.cdl_pattern(
            df["open"], df["high"], df["low"], df["close"], name="morningstar"
        )
        patterns["dragonflydoji"] = ta.cdl_pattern(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            name="dragonflydoji",
        )
        patterns["3whitesoldiers"] = ta.cdl_pattern(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            name="3whitesoldiers",
        )
        patterns["closingmarubozu"] = ta.cdl_pattern(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            name="closingmarubozu",
        )

        # 看跌型態
        patterns["shootingstar"] = ta.cdl_pattern(
            df["open"], df["high"], df["low"], df["close"], name="shootingstar"
        )
        patterns["darkcloudcover"] = ta.cdl_pattern(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            name="darkcloudcover",
        )
        patterns["eveningstar"] = ta.cdl_pattern(
            df["open"], df["high"], df["low"], df["close"], name="eveningstar"
        )
        patterns["gravestonedoji"] = ta.cdl_pattern(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            name="gravestonedoji",
        )
        patterns["3blackcrows"] = ta.cdl_pattern(
            df["open"], df["high"], df["low"], df["close"], name="3blackcrows"
        )

        # 生成信號
        for pattern_name, pattern_data in patterns.items():
            if pattern_data is not None:
                if pattern_name in [
                    "hammer",
                    "morningstar",
                    "dragonflydoji",
                    "3whitesoldiers",
                ]:
                    mask = pattern_data.iloc[:, 0] >= self.params["threshold"]
                    signals[mask] = 1
                elif pattern_name == "engulfing":
                    threshold = self.params["threshold"]
                    neg_threshold = -self.params["threshold"]
                    buy_mask = pattern_data.iloc[:, 0] >= threshold
                    sell_mask = pattern_data.iloc[:, 0] <= neg_threshold
                    signals[buy_mask] = 1
                    signals[sell_mask] = -1
                elif pattern_name == "closingmarubozu":
                    threshold = self.params["threshold"]
                    neg_threshold = -self.params["threshold"]
                    buy_mask = pattern_data.iloc[:, 0] >= threshold
                    sell_mask = pattern_data.iloc[:, 0] <= neg_threshold
                    signals[buy_mask] = 1
                    signals[sell_mask] = -1
                else:
                    neg_threshold = -self.params["threshold"]
                    sell_mask = pattern_data.iloc[:, 0] <= neg_threshold
                    signals[sell_mask] = -1

        return signals

    def get_description(self) -> str:
        return (
            f"Candlestick Patterns (reversal: hammer, engulfing, stars; "
            f"continuation: 3 soldiers/crows, "
            f"strength >= {self.params['threshold']})"
        )
