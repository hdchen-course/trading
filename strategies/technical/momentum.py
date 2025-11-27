#!/usr/bin/env python3
"""
動量指標策略 (RSI, StochRSI, ATR, SAR)
"""

import pandas as pd
import pandas_ta as ta

from ..base import BaseStrategy, StrategyConfig


class StochRSIStrategy(BaseStrategy):
    """隨機 RSI 策略"""

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="Stochastic RSI",
                params={
                    "rsi_period": 14,
                    "stoch_period": 14,
                    "k_period": 3,
                    "d_period": 3,
                    "oversold": 5,
                    "overbought": 95,
                },
            )
        super().__init__(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        stochrsi = ta.stochrsi(
            df["close"],
            length=self.params["rsi_period"],
            rsi_length=self.params["stoch_period"],
            k=self.params["k_period"],
            d=self.params["d_period"],
        )

        signals = pd.Series(0, index=df.index)

        k_line = stochrsi[
            f'STOCHRSIk_{self.params["rsi_period"]}_'
            f'{self.params["stoch_period"]}_{self.params["k_period"]}_'
            f'{self.params["d_period"]}'
        ]

        signals[k_line < self.params["oversold"]] = 1
        signals[k_line > self.params["overbought"]] = -1

        return signals

    def get_description(self) -> str:
        return (
            f"Stochastic RSI({self.params['rsi_period']}) < "
            f"{self.params['oversold']} buy, > "
            f"{self.params['overbought']} sell"
        )


class ATRStrategy(BaseStrategy):
    """ATR 波動策略"""

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="ATR Volatility",
                params={"atr_period": 14, "multiplier": 2.0},
            )
        super().__init__(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        atr = ta.atr(
            df["high"],
            df["low"],
            df["close"],
            length=self.params["atr_period"],
        )
        sma = ta.sma(df["close"], length=self.params["atr_period"])

        signals = pd.Series(0, index=df.index)

        upper_band = sma + (atr * self.params["multiplier"])
        lower_band = sma - (atr * self.params["multiplier"])

        signals[df["close"] <= lower_band] = 1
        signals[df["close"] >= upper_band] = -1

        return signals

    def get_description(self) -> str:
        return (
            f"ATR({self.params['atr_period']}) "
            f"{self.params['multiplier']}x volatility channel"
        )


class SARStrategy(BaseStrategy):
    """拋物線 SAR 策略"""

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="Parabolic SAR",
                params={"start": 0.02, "increment": 0.02, "maximum": 0.2},
            )
        super().__init__(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sar = ta.psar(
            df["high"],
            df["low"],
            df["close"],
            af0=self.params["start"],
            af=self.params["increment"],
            max_af=self.params["maximum"],
        )

        signals = pd.Series(0, index=df.index)

        buy_condition = (df["close"] > sar["PSARl_0.02_0.2"]) & (
            sar["PSARl_0.02_0.2"].notna()
        )
        sell_condition = (df["close"] < sar["PSARs_0.02_0.2"]) & (
            sar["PSARs_0.02_0.2"].notna()
        )

        signals[buy_condition] = 1
        signals[sell_condition] = -1

        return signals

    def get_description(self) -> str:
        return (
            f"Parabolic SAR({self.params['start']},"
            f"{self.params['increment']},{self.params['maximum']})"
        )
