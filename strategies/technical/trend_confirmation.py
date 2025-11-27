#!/usr/bin/env python3
"""
趨勢確認策略
"""

import pandas as pd
import pandas_ta as ta

from ..base import BaseStrategy, StrategyConfig


class TrendConfirmationStrategy(BaseStrategy):
    """趨勢確認策略 - 多條件驗證趨勢有效性"""

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="Trend Confirmation",
                params={
                    "confirmation_hours": 2,
                    "bb_period": 20,
                    "bb_std": 2.0,
                },
            )
        super().__init__(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)

        bb = ta.bbands(
            df["close"],
            length=self.params["bb_period"],
            std=self.params["bb_std"],
        )
        bb_period = self.params["bb_period"]
        bb_std = self.params["bb_std"]
        bb_upper = bb[f"BBU_{bb_period}_{bb_std}_{bb_std}"]
        bb_lower = bb[f"BBL_{bb_period}_{bb_std}_{bb_std}"]

        rsi = ta.rsi(df["close"], length=14)
        volume_ma = ta.sma(df["volume"], length=20)
        volume_ratio = df["volume"] / volume_ma

        bb_position = (df["close"] - bb_lower) / (bb_upper - bb_lower)

        time_window = int(self.params["confirmation_hours"] * 4)

        price_change_rate = df["low"].pct_change(time_window)
        high_change_rate = df["high"].pct_change(time_window)
        slope_threshold_pct = 0.0048 / time_window

        rising_lows = price_change_rate > slope_threshold_pct
        falling_highs = high_change_rate < -slope_threshold_pct

        kline_up = df["close"] > df["open"]
        kline_down = df["close"] < df["open"]
        up_count = kline_up.rolling(time_window).sum()
        down_count = kline_down.rolling(time_window).sum()
        up_down_ratio = up_count / (down_count + 1e-10)
        down_up_ratio = down_count / (up_count + 1e-10)
        strong_uptrend = up_down_ratio >= 2.0
        strong_downtrend = down_up_ratio >= 2.0

        rsi_buy_filter = rsi < 55
        rsi_sell_filter = rsi > 63

        volume_buy_confirm = volume_ratio > 1.2
        volume_sell_confirm = volume_ratio > 1.2

        bb_buy_position = bb_position < 0.6
        bb_sell_position = bb_position > 0.7

        buy_conditions = (
            rising_lows
            & strong_uptrend
            & rsi_buy_filter
            & volume_buy_confirm
            & bb_buy_position
        )

        sell_conditions = (
            falling_highs
            & strong_downtrend
            & rsi_sell_filter
            & volume_sell_confirm
            & bb_sell_position
        )

        signals[buy_conditions] = 1
        signals[sell_conditions] = -1
        signals.iloc[:time_window] = 0

        return signals

    def get_description(self) -> str:
        hours = self.params["confirmation_hours"]
        return (
            f"Trend Confirmation: {hours}h trend slopes, K-line ratio≥2, "
            f"RSI<55 buy/RSI>63 sell, volume>1.2x, BB position filter"
        )
