#!/usr/bin/env python3
"""調試動態停損"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
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


def debug_trailing_stop():
    config = TradingConfig()
    backtester = CryptoBacktester(config)

    # 做多情境2測試數據 (修正版)
    dates = pd.date_range("2024-02-01", periods=80, freq="15min")
    prices = (
        [100] * 50
        + [100, 102, 104, 106, 108, 110]  # 漸漲到110 (10%獲利)
        + [108, 106, 104, 103, 102.8]  # 回落觸發停損
        + [102.8] * 19
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

    strategy = MockStrategy(buy_signals=[50], sell_signals=[])
    coin_config = CoinConfig("Test", "TESTUSDT", [strategy], 10)

    trades, _, _ = backtester.backtest_strategy_vectorized(
        df, coin_config, "TEST"
    )

    print("=== 調試結果 ===")
    print(f"交易數量: {len(trades)}")

    if len(trades) > 0:
        trade = trades[0]
        print(f"進場價格: {trade['entry_price']}")
        print(f"出場價格: {trade['exit_price']}")
        print(f"出場原因: {trade['exit_reason']}")
        print(f"盈虧: {trade['pnl']:.2%}")
        print(f"收益: {trade['return']:.2%}")

        # 計算預期的停損價格
        print(f"\n=== 預期計算 ===")
        print(f"初始停損: {100 * (1 - 0.018/10):.2f}")  # 98.2
        print(f"4%獲利時停損應移至: {100 * (1 + 0.02):.2f}")  # 102
        print(f"6%獲利時停損應移至: {100 * (1 + 0.03):.2f}")  # 103
        print(f"最終價格102.5，應該觸發停損")

    # 檢查價格序列
    print(f"\n=== 價格序列 (50-70) ===")
    for i in range(50, min(70, len(prices))):
        print(f"{i}: {prices[i]}")


if __name__ == "__main__":
    debug_trailing_stop()
