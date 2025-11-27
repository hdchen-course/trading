#!/usr/bin/env python3
"""
成交量策略
"""

import pandas as pd
import pandas_ta as ta

from ..base import BaseStrategy, StrategyConfig


class VolumeStrategy(BaseStrategy):
    """成交量策略 (量價配合)"""

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="Volume Breakout",
                params={
                    "volume_ma_period": 20,
                    "volume_threshold": 2.4,
                    "price_change_threshold": 0.025,
                },
            )
        super().__init__(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)

        period = self.params["volume_ma_period"]
        volume_ma = ta.sma(df["volume"], length=period)
        price_change = df["close"].pct_change()

        threshold = self.params["volume_threshold"]
        volume_surge = df["volume"] > (volume_ma * threshold)

        buy_condition = volume_surge & (
            price_change > self.params["price_change_threshold"]
        )
        signals[buy_condition] = 1

        sell_condition = volume_surge & (
            price_change < -self.params["price_change_threshold"]
        )
        signals[sell_condition] = -1

        return signals

    def get_description(self) -> str:
        return (
            f"Volume Breakout (vol>{self.params['volume_threshold']}x avg + "
            f"price change>{self.params['price_change_threshold']:.1%})"
        )
