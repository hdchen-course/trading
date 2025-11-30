#!/usr/bin/env python3
"""
Bollinger Bands 相關策略
"""

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import talib

from ..base import BaseStrategy, StrategyConfig


class BollingerStrategy(BaseStrategy):
    """布林通道策略"""

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                    name="Bollinger Bands",
                    params={"period": 32, "std": 3.2, "multiplier": 3.2},
                    )
        super().__init__(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        std_val = self.params["std"]
        bb = ta.bbands(
                df["close"],
                length=self.params["period"],
                lower_std=std_val,
                upper_std=std_val,
                )
        signals = pd.Series(0, index=df.index)

        period = self.params["period"]
        std_val = self.params["std"]

        upper_band = bb[f"BBU_{period}_{std_val}_{std_val}"]
        lower_band = bb[f"BBL_{period}_{std_val}_{std_val}"]

        signals[df["close"] <= lower_band] = 1
        signals[df["close"] >= upper_band] = -1

        return signals

    def get_description(self) -> str:
        return (
                f"Bollinger Bands({self.params['period']},{self.params['std']}) "
                f"standard deviation bands"
                )


class BollingerExtremeStrategy(BaseStrategy):
    """布林極限策略"""

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            params = {"period": 32, "std": 3.2}
            config = StrategyConfig(name="Bollinger Extreme", params=params)
        super().__init__(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        std_val = self.params["std"]
        bb = ta.bbands(
                df["close"],
                length=self.params["period"],
                lower_std=std_val,
                upper_std=std_val,
                )
        signals = pd.Series(0, index=df.index)

        period = self.params["period"]
        signals[df["close"] < bb[f"BBL_{period}_{std_val}_{std_val}"]] = 1
        signals[df["close"] > bb[f"BBU_{period}_{std_val}_{std_val}"]] = -1

        return signals

    def get_description(self) -> str:
        return (
                f"Bollinger Extreme({self.params['period']},{self.params['std']}) "
                f"extreme breakout strategy"
                )
