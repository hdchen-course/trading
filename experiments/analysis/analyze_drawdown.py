#!/usr/bin/env python3
"""
分析SidewaysRangeStrategy的回撤分布
"""

import pandas as pd
import numpy as np
from crypto_backtest import CryptoBacktester
from crypto_strategies import SidewaysRangeStrategy, StrategyConfig, CoinConfig
from main import TradingConfig


def calculate_max_drawdown(trades):
    """計算最大回撤"""
    if not trades:
        return 0

    # 計算累積收益曲線 (從初始資金開始)
    initial_capital = 10000  # 假設初始資金
    cumulative_values = [initial_capital]

    for trade in trades:
        new_value = cumulative_values[-1] + trade["pnl"]
        cumulative_values.append(new_value)

    if len(cumulative_values) <= 1:
        return 0

    # 計算最大回撤
    peak = cumulative_values[0]
    max_drawdown = 0

    for value in cumulative_values:
        if value > peak:
            peak = value
        if peak > 0:
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown


def main():
    print("📊 分析SidewaysRangeStrategy回撤分布...")

    # 初始化回測器
    config = TradingConfig()
    backtester = CryptoBacktester(config)

    # 獲取數據 (3個月)
    df = backtester.get_binance_data("BTCUSDT", "15m", 3 * 30 * 24 * 4)
    if df.empty:
        print("❌ 無法獲取數據")
        return

    print(
        f"📊 數據期間: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}"
    )

    # 測試幾個代表性的參數組合
    test_configs = [
        {
            "bb_period": 40,
            "bb_std": 1.5,
            "min_bb_width": 1.0,
            "max_bb_width": 2.5,
        },
        {
            "bb_period": 50,
            "bb_std": 1.5,
            "min_bb_width": 1.2,
            "max_bb_width": 2.0,
        },
        {
            "bb_period": 30,
            "bb_std": 2.0,
            "min_bb_width": 0.8,
            "max_bb_width": 2.0,
        },
        {
            "bb_period": 40,
            "bb_std": 2.0,
            "min_bb_width": 1.0,
            "max_bb_width": 2.0,
        },
        {
            "bb_period": 50,
            "bb_std": 2.0,
            "min_bb_width": 1.0,
            "max_bb_width": 2.0,
        },
    ]

    results = []

    for i, params in enumerate(test_configs, 1):
        print(
            f"\n測試配置 {i}: BB({params['bb_period']}, {params['bb_std']}) 振幅({params['min_bb_width']}-{params['max_bb_width']}%)"
        )

        # 創建策略
        strategy = SidewaysRangeStrategy(
            StrategyConfig(
                "Test",
                {
                    "bb_period": params["bb_period"],
                    "bb_std": params["bb_std"],
                    "atr_period": 14,
                    "max_atr_pct": 0.30,
                    "min_bb_width": params["min_bb_width"],
                    "max_bb_width": params["max_bb_width"],
                    "entry_threshold": 0.995,
                    "volume_period": 20,
                    "volume_threshold": 1.0,
                },
            )
        )

        coin_config = CoinConfig("Test", "TESTUSDT", [strategy], 10)

        try:
            trades, total_return, win_rate = (
                backtester.backtest_strategy_vectorized(
                    df, coin_config, "TEST"
                )
            )

            if trades:
                max_drawdown = calculate_max_drawdown(trades)
                sharpe_ratio = (
                    (total_return * np.sqrt(252))
                    / (
                        np.std([t["return_pct"] for t in trades])
                        * np.sqrt(252)
                    )
                    if trades
                    else 0
                )

                result = {
                    "config": f"BB({params['bb_period']},{params['bb_std']}) {params['min_bb_width']}-{params['max_bb_width']}%",
                    "trades": len(trades),
                    "total_return": total_return,
                    "win_rate": win_rate,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                }

                results.append(result)

                print(f"  交易次數: {len(trades)}")
                print(f"  總收益率: {total_return:.2%}")
                print(f"  勝率: {win_rate:.1%}")
                print(f"  夏普比率: {sharpe_ratio:.2f}")
                print(f"  最大回撤: {max_drawdown:.2%}")

                # 分析交易分布
                profits = [t["pnl"] for t in trades if t["pnl"] > 0]
                losses = [t["pnl"] for t in trades if t["pnl"] < 0]

                if profits and losses:
                    print(f"  平均獲利: ${np.mean(profits):.2f}")
                    print(f"  平均虧損: ${np.mean(losses):.2f}")
                    print(f"  最大單筆獲利: ${max(profits):.2f}")
                    print(f"  最大單筆虧損: ${min(losses):.2f}")

        except Exception as e:
            print(f"  ❌ 測試失敗: {e}")

    # 總結
    if results:
        print("\n" + "=" * 80)
        print("📈 回撤分析總結:")
        print("=" * 80)

        drawdowns = [r["max_drawdown"] for r in results]
        print(f"最小回撤: {min(drawdowns):.2%}")
        print(f"最大回撤: {max(drawdowns):.2%}")
        print(f"平均回撤: {np.mean(drawdowns):.2%}")

        # 按回撤排序
        results.sort(key=lambda x: x["max_drawdown"])

        print(f"\n🏆 回撤最低的配置:")
        best = results[0]
        print(f"配置: {best['config']}")
        print(f"回撤: {best['max_drawdown']:.2%}")
        print(f"收益: {best['total_return']:.2%}")
        print(f"夏普: {best['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    main()
