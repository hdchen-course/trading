#!/usr/bin/env python3
"""修復動態停損測試"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import logging
from crypto_backtest import CryptoBacktester
from crypto_strategies import BaseStrategy, StrategyConfig, CoinConfig
from main import TradingConfig


class MockStrategy(BaseStrategy):
    def __init__(self, buy_signals, sell_signals):
        super().__init__(StrategyConfig("Mock", {}))
        self.buy_signals = buy_signals
        self.sell_signals = sell_signals

    def generate_signals(self, df):
        signals = pd.Series(0, index=df.index)
        for i in self.buy_signals:
            if i < len(signals):
                signals.iloc[i] = 1
        for i in self.sell_signals:
            if i < len(signals):
                signals.iloc[i] = -1
        return signals

    def get_description(self):
        return "Mock Strategy for Testing"


def test_fixed_cases():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config = TradingConfig()

    # 做多停利測試
    print("=== 做多停利測試 ===")
    dates = pd.date_range("2024-01-01", periods=120, freq="15min")
    prices = (
        [100] * 51
        + [
            100.1,
            100.2,
            100.3,
            100.4,
            100.5,
            100.6,
            100.7,
            100.75,
            100.8,
            100.9,
            101.0,
            101.1,
            101.2,
            101.3,
            101.4,
            101.5,
        ]
        + [101.5] * 53
    )

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
        }
    )

    strategy = MockStrategy(buy_signals=[51], sell_signals=[])
    coin_config = CoinConfig("Test", "TESTUSDT", [strategy], 10)

    backtester = CryptoBacktester(config)
    trades, _, _ = backtester.backtest_strategy_vectorized(
        df, coin_config, "TEST"
    )

    print(f"交易數量: {len(trades)}")
    if len(trades) > 0:
        trade = trades[0]
        print(f"進場: {trade['entry_price']:.3f}")
        print(f"出場: {trade['exit_price']:.3f}")
        print(f"原因: {trade['exit_reason']}")
        print(f"盈虧: {trade['pnl']:.2%}")
    else:
        print("沒有交易記錄")

        # 檢查信號
        signals = strategy.generate_signals(df)
        print(f"信號總數: {signals.sum()}")
        print(
            f"第51根信號: {signals.iloc[51] if len(signals) > 51 else 'N/A'}"
        )


if __name__ == "__main__":
    test_fixed_cases()
