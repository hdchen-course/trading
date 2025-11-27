#!/usr/bin/env python3

import pandas as pd
import sys

sys.path.append(".")

from crypto_backtest import CryptoBacktester
from crypto_strategies import CoinConfig, StrategyConfig, BaseStrategy
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


# 創建測試數據
dates = pd.date_range("2024-01-01", periods=100, freq="15min")
prices = (
    [100] * 50
    + [100, 100.2, 100.4, 100.5, 100.6, 100.75, 100.9, 101.0, 101.2, 101.5]
    + [101.5] * 40
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
strategy = MockStrategy(buy_signals=[51], sell_signals=[])
coin_config = CoinConfig("Test", "TESTUSDT", [strategy], 10)
config = TradingConfig()

# 手動實現回測邏輯來調試
signals = strategy.generate_signals(df)
trades = []
position = None
capital = config.initial_capital

# 槓桿後的停損停利比例
actual_stop_loss = config.stop_loss_pct / coin_config.leverage
actual_take_profit = config.take_profit_pct / coin_config.leverage

print(f"開始回測調試:")
print(f"停損比例: {actual_stop_loss:.4f}")
print(f"停利比例: {actual_take_profit:.4f}")

for i in range(50, len(df)):
    current_price = df.iloc[i]["close"]
    timestamp = df.iloc[i]["timestamp"]
    signal = signals.iloc[i]

    if i <= 55:  # 只打印前幾根的詳細信息
        print(
            f"第{i}根: 價格={current_price:.1f}, 信號={signal}, 持倉={position is not None}"
        )

    if signal == 1 and position is None:  # 買入信號 (做多)
        position = {
            "entry_price": current_price,
            "entry_time": timestamp,
            "entry_idx": i,
            "stop_loss_price": current_price * (1 - actual_stop_loss),
            "take_profit_price": current_price * (1 + actual_take_profit),
            "position_type": "long",
            "original_stop_loss": actual_stop_loss,
            "original_take_profit": actual_take_profit,
            "level": 0,
            "last_adjusted_level": -1,
        }
        print(
            f"  -> 開倉做多: 進場={current_price:.1f}, 停損={position['stop_loss_price']:.3f}, 停利={position['take_profit_price']:.3f}"
        )

    elif position is not None:
        # 檢查平倉條件
        exit_reason = None
        exit_price = None

        # 停損檢查
        if config.enable_stop_loss and (
            (
                position["position_type"] == "long"
                and current_price <= position["stop_loss_price"]
            )
            or (
                position["position_type"] == "short"
                and current_price >= position["stop_loss_price"]
            )
        ):
            exit_reason = "stop_loss"
            exit_price = position["stop_loss_price"]

        # 停利檢查
        elif config.enable_take_profit and (
            (
                position["position_type"] == "long"
                and current_price >= position["take_profit_price"]
            )
            or (
                position["position_type"] == "short"
                and current_price <= position["take_profit_price"]
            )
        ):
            exit_reason = "take_profit"
            exit_price = position["take_profit_price"]

        if exit_reason:
            # 計算盈虧
            if position["position_type"] == "long":
                pnl_pct = (exit_price - position["entry_price"]) / position[
                    "entry_price"
                ]
            else:
                pnl_pct = (position["entry_price"] - exit_price) / position[
                    "entry_price"
                ]

            pnl = capital * pnl_pct * coin_config.leverage

            trade = {
                "entry_time": position["entry_time"],
                "exit_time": timestamp,
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "position_type": position["position_type"],
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": exit_reason,
                "leverage": coin_config.leverage,
            }

            trades.append(trade)
            print(
                f"  -> 平倉: 出場={exit_price:.3f}, 原因={exit_reason}, 盈虧={pnl:.2f}"
            )
            position = None

print(f"\n最終結果: {len(trades)} 筆交易")
if trades:
    for i, trade in enumerate(trades):
        print(
            f"交易{i+1}: {trade['entry_price']:.1f} -> {trade['exit_price']:.3f} ({trade['exit_reason']})"
        )
