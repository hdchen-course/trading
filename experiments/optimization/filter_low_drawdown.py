#!/usr/bin/env python3
"""
篩選最大回撤低於5%的SidewaysRangeStrategy參數組合
"""

import pandas as pd
import numpy as np
from crypto_backtest import CryptoBacktester
from crypto_config import get_test_config
from crypto_strategies import SidewaysRangeStrategy, StrategyConfig, CoinConfig
import itertools


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
    print("🔍 尋找最大回撤 < 8% 的參數組合...")

    # 初始化回測器
    from main import TradingConfig

    config = TradingConfig()
    backtester = CryptoBacktester(config)

    # 獲取數據
    df = backtester.get_binance_data("BTCUSDT", "15m", 6 * 30 * 24 * 4)
    if df.empty:
        print("❌ 無法獲取數據")
        return

    print(
        f"📊 數據期間: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}"
    )

    # 參數範圍 (聚焦在表現較好的範圍)
    bb_periods = [30, 40, 50]
    bb_stds = [1.5, 2.0, 2.5]
    min_widths = [0.8, 1.0, 1.2]
    max_widths = [2.0, 2.5, 3.0]
    entry_thresholds = [0.995]
    volume_thresholds = [1.0]

    total_combinations = (
        len(bb_periods)
        * len(bb_stds)
        * len(min_widths)
        * len(max_widths)
        * len(entry_thresholds)
        * len(volume_thresholds)
    )
    print(f"🧮 總共測試 {total_combinations} 個參數組合")

    valid_results = []
    count = 0

    for (
        bb_period,
        bb_std,
        min_width,
        max_width,
        entry_threshold,
        volume_threshold,
    ) in itertools.product(
        bb_periods,
        bb_stds,
        min_widths,
        max_widths,
        entry_thresholds,
        volume_thresholds,
    ):
        if min_width >= max_width:
            continue

        count += 1

        # 創建策略
        strategy = SidewaysRangeStrategy(
            StrategyConfig(
                "Test",
                {
                    "bb_period": bb_period,
                    "bb_std": bb_std,
                    "min_bb_width": min_width,
                    "max_bb_width": max_width,
                    "entry_threshold": entry_threshold,
                    "volume_threshold": volume_threshold,
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

            if len(trades) >= 50:  # 至少50筆交易
                max_drawdown = calculate_max_drawdown(trades)

                if max_drawdown < 0.08:  # 回撤 < 8%
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
                        "bb_period": bb_period,
                        "bb_std": bb_std,
                        "min_bb_width": min_width,
                        "max_bb_width": max_width,
                        "entry_threshold": entry_threshold,
                        "volume_threshold": volume_threshold,
                        "trades": len(trades),
                        "total_return": total_return,
                        "win_rate": win_rate,
                        "sharpe_ratio": sharpe_ratio,
                        "max_drawdown": max_drawdown,
                        "avg_return_per_trade": (
                            total_return / len(trades) if trades else 0
                        ),
                    }

                    valid_results.append(result)

                    print(
                        f"#{len(valid_results):3d} | BB({bb_period},{bb_std}) | 振幅({min_width:.1f}-{max_width:.1f}%) | "
                        f"交易:{len(trades):3d} | 收益:{total_return:6.2%} | 勝率:{win_rate:5.1%} | "
                        f"夏普:{sharpe_ratio:5.2f} | 回撤:{max_drawdown:5.2%}"
                    )

        except Exception as e:
            continue

        # 進度顯示
        if count % 20 == 0:
            print(
                f"\n進度: {count}/{total_combinations} ({count/total_combinations*100:.1f}%) - 找到 {len(valid_results)} 個低回撤組合"
            )

    print(f"\n找到 {len(valid_results)} 個最大回撤 < 8% 的參數組合")

    if valid_results:
        # 按夏普比率排序
        valid_results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

        print("\n前10名最佳組合 (按夏普比率排序):")
        print("-" * 120)

        for i, result in enumerate(valid_results[:10], 1):
            print(f"\n第 {i:2d} 名:")
            print(f"  布林通道: BB({result['bb_period']}, {result['bb_std']})")
            print(
                f"  振幅範圍: {result['min_bb_width']:.1f}% - {result['max_bb_width']:.1f}%"
            )
            print(f"  進場門檻: {result['entry_threshold']}")
            print(f"  成交量門檻: {result['volume_threshold']}")
            print(f"  交易次數: {result['trades']}")
            print(f"  總收益率: {result['total_return']:.2%}")
            print(f"  勝率: {result['win_rate']:.1%}")
            print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {result['max_drawdown']:.2%}")
            print(f"  平均每筆收益: {result['avg_return_per_trade']:.3%}")
    else:
        print("❌ 沒有找到符合條件的組合")


if __name__ == "__main__":
    main()
