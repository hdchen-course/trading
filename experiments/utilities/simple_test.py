#!/usr/bin/env python3

import pandas as pd
import sys

sys.path.append(".")

from crypto_backtest import CryptoBacktester
from crypto_strategies import CoinConfig, StrategyConfig, BaseStrategy
from main import TradingConfig


class SimpleStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(StrategyConfig("Simple", {}))

    def generate_signals(self, df):
        signals = pd.Series(0, index=df.index)
        if len(signals) > 51:
            signals.iloc[51] = 1  # 買入信號在第51根
        return signals

    def get_description(self):
        return "Simple Test Strategy"


def main():
    # 創建簡單的測試數據
    dates = pd.date_range("2024-01-01", periods=70, freq="15min")
    prices = [100] * 50 + [
        100.1,
        100.2,
        100.3,
        100.4,
        100.5,
        100.6,
        100.7,
        100.8,
        100.9,
        101.0,
        101.1,
        101.2,
        101.3,
        101.4,
        101.5,
        101.6,
        101.7,
        101.8,
        101.9,
        102.0,
    ]

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
    strategy = SimpleStrategy()
    coin_config = CoinConfig("Test", "TESTUSDT", [strategy], 10)
    config = TradingConfig()

    print(f"配置檢查:")
    print(f"  停損: {config.stop_loss_pct}")
    print(f"  停利: {config.take_profit_pct}")
    print(f"  啟用停損: {config.enable_stop_loss}")
    print(f"  啟用停利: {config.enable_take_profit}")

    # 檢查信號
    signals = strategy.generate_signals(df)
    print(f"\n信號檢查:")
    print(f"  信號數量: {len(signals)}")
    print(f"  買入信號位置: {[i for i, s in enumerate(signals) if s == 1]}")

    # 檢查價格
    print(f"\n價格檢查:")
    print(f"  數據長度: {len(df)}")
    print(f"  第51根價格: {df.iloc[51]['close']}")
    print(f"  最高價格: {max(prices)}")

    # 計算預期停利價格
    entry_price = df.iloc[51]["close"]
    actual_take_profit = config.take_profit_pct / coin_config.leverage
    expected_tp = entry_price * (1 + actual_take_profit)
    print(f"  進場價格: {entry_price}")
    print(f"  槓桿後停利比例: {actual_take_profit:.4f}")
    print(f"  預期停利價格: {expected_tp:.3f}")
    print(f"  是否能達到: {max(prices) >= expected_tp}")

    # 運行回測
    print(f"\n回測結果:")
    backtester = CryptoBacktester(config)
    trades, _, _ = backtester.backtest_strategy_vectorized(
        df, coin_config, "TEST"
    )

    print(f"  交易數量: {len(trades)}")
    if len(trades) > 0:
        trade = trades[0]
        print(
            f"  交易詳情: 進場{trade['entry_price']:.1f} -> 出場{trade['exit_price']:.3f}"
        )
        print(f"  出場原因: {trade['exit_reason']}")
        print(f"  盈虧: {trade['pnl']:.2f}")
    else:
        print("  沒有交易記錄")


if __name__ == "__main__":
    main()
