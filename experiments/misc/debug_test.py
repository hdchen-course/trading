#!/usr/bin/env python3

import pandas as pd
import sys

sys.path.append(".")

from crypto_backtest import CryptoBacktester
from config import BacktestConfig, CoinConfig
from strategy.base_strategy import BaseStrategy, StrategyConfig


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


# 創建測試數據
dates = pd.date_range("2024-01-01", periods=80, freq="15min")
prices = (
    [100] * 50
    + [100, 100.2, 100.4, 100.5, 100.6, 100.75, 100.9, 101.0, 101.2, 101.5]
    + [101.5] * 20
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

# 創建策略和配置
strategy = MockStrategy(buy_signals=[49], sell_signals=[])
coin_config = CoinConfig("Test", "TESTUSDT", [strategy], 10)

config = BacktestConfig(
    initial_capital=10000,
    stop_loss_pct=1.8,
    take_profit_pct=10.0,
    enable_trailing_stop=True,
)

# 運行回測
backtester = CryptoBacktester(config)
trades, _, _ = backtester.backtest_strategy_vectorized(df, coin_config, "TEST")

# 檢查結果
signals = strategy.generate_signals(df)
print(f"信號數量: {len(signals)}")
print(f"買入信號位置: {[i for i, s in enumerate(signals) if s == 1]}")
print(f"數據長度: {len(df)}")
print(f"價格範圍: {df['close'].min():.1f}-{df['close'].max():.1f}")
print(f"交易數量: {len(trades)}")

if len(trades) > 0:
    trade = trades[0]
    print(
        f"交易詳情: 進場{trade['entry_price']:.1f} -> 出場{trade['exit_price']:.1f}"
    )
    print(f"出場原因: {trade['exit_reason']}")
else:
    print("沒有交易記錄")

    # 檢查回測器內部邏輯
    print("\n檢查回測器邏輯:")
    print(f"從第50根開始處理: {df.iloc[50]['close']}")
    print(f"第49根信號: {signals.iloc[49]}")
    print(f"第50根信號: {signals.iloc[50]}")
