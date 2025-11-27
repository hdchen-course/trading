#!/usr/bin/env python3
"""
篩選最佳結果：收益>20% 且 夏普>=2
"""

import pandas as pd
import itertools
from dataclasses import dataclass
from crypto_backtest import CryptoBacktester
from crypto_strategies import SidewaysRangeStrategy, StrategyConfig, CoinConfig


@dataclass
class OptimizeConfig:
    timeframe_minutes: int = 15
    initial_capital: float = 10000.0
    stop_loss_pct: float = 0.018
    take_profit_pct: float = 0.10
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    binance_base_url: str = "https://api.binance.com"

    @property
    def timeframe_str(self) -> str:
        return "15m"


def test_top_combinations():
    """測試頂級組合"""
    df = pd.read_csv("data/btc_6months_15m.csv", parse_dates=["timestamp"])
    config = OptimizeConfig()
    backtester = CryptoBacktester(config)

    # 從輸出中提取的高收益組合
    top_combinations = [
        # BB(30,2.0) 振幅(0.8-3.0%) - #292
        {
            "bb_period": 30,
            "bb_std": 2.0,
            "atr_period": 14,
            "max_atr_pct": 0.30,
            "min_bb_width": 0.8,
            "max_bb_width": 3.0,
            "entry_threshold": 0.995,
            "volume_period": 20,
            "volume_threshold": 1.0,
        },
        # BB(30,1.5) 振幅(0.8-2.0%) - #191
        {
            "bb_period": 30,
            "bb_std": 1.5,
            "atr_period": 14,
            "max_atr_pct": 0.30,
            "min_bb_width": 0.8,
            "max_bb_width": 2.0,
            "entry_threshold": 0.995,
            "volume_period": 20,
            "volume_threshold": 0.8,
        },
        # BB(30,1.5) 振幅(1.0-2.0%) - #235
        {
            "bb_period": 30,
            "bb_std": 1.5,
            "atr_period": 14,
            "max_atr_pct": 0.30,
            "min_bb_width": 1.0,
            "max_bb_width": 2.0,
            "entry_threshold": 0.995,
            "volume_period": 20,
            "volume_threshold": 1.0,
        },
        # BB(30,2.0) 振幅(0.8-2.0%) - #346
        {
            "bb_period": 30,
            "bb_std": 2.0,
            "atr_period": 14,
            "max_atr_pct": 0.30,
            "min_bb_width": 0.8,
            "max_bb_width": 2.0,
            "entry_threshold": 0.995,
            "volume_period": 20,
            "volume_threshold": 0.8,
        },
        # BB(50,1.5) 振幅(1.0-2.0%) - #887
        {
            "bb_period": 50,
            "bb_std": 1.5,
            "atr_period": 14,
            "max_atr_pct": 0.30,
            "min_bb_width": 1.0,
            "max_bb_width": 2.0,
            "entry_threshold": 0.995,
            "volume_period": 20,
            "volume_threshold": 0.8,
        },
        # BB(50,2.0) 振幅(1.0-2.0%) - #887
        {
            "bb_period": 50,
            "bb_std": 2.0,
            "atr_period": 14,
            "max_atr_pct": 0.30,
            "min_bb_width": 1.0,
            "max_bb_width": 2.0,
            "entry_threshold": 0.995,
            "volume_period": 20,
            "volume_threshold": 0.8,
        },
    ]

    results = []

    print("測試頂級組合 (收益>20% 且 夏普>=2):")
    print("=" * 80)

    for i, params in enumerate(top_combinations):
        # 創建策略
        strategy_config = StrategyConfig(
            name=f"Top Combo #{i+1}", params=params, weight=1.0
        )
        strategy = SidewaysRangeStrategy(strategy_config)
        coin_config = CoinConfig("Bitcoin", "BTCUSDT", [strategy], 1.0)

        # 執行回測
        trades, total_return, win_rate = (
            backtester.backtest_strategy_vectorized(df, coin_config, "BTCUSDT")
        )

        if trades:
            import numpy as np

            returns = [t["leveraged_return_pct"] for t in trades]
            returns_array = np.array(returns)
            sharpe_ratio = (
                np.mean(returns_array)
                / (np.std(returns_array) + 1e-8)
                * np.sqrt(252)
            )

            # 計算最大回撤
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))

            result = {
                "combo": i + 1,
                "params": params,
                "trades": len(trades),
                "total_return": total_return,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "avg_return_per_trade": total_return / len(trades),
            }
            results.append(result)

            print(f"組合 #{i+1}:")
            print(
                f"  BB({params['bb_period']},{params['bb_std']}) 振幅({params['min_bb_width']}-{params['max_bb_width']}%)"
            )
            print(f"  交易次數: {len(trades)}")
            print(f"  總收益率: {total_return:.2%}")
            print(f"  勝率: {win_rate:.1%}")
            print(f"  夏普比率: {sharpe_ratio:.2f}")
            print(f"  最大回撤: {max_drawdown:.2%}")
            print(f"  平均每筆: {total_return/len(trades):.3%}")
            print("-" * 50)

    # 按總收益率排序
    results.sort(key=lambda x: x["total_return"], reverse=True)

    print("\n按總收益率排序的前3名:")
    print("=" * 50)
    for i, result in enumerate(results[:3]):
        params = result["params"]
        print(f"\n第 {i+1} 名:")
        print(
            f"  BB({params['bb_period']},{params['bb_std']}) 振幅({params['min_bb_width']}-{params['max_bb_width']}%)"
        )
        print(f"  總收益率: {result['total_return']:.2%}")
        print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
        print(f"  勝率: {result['win_rate']:.1%}")
        print(f"  交易次數: {result['trades']}")
        print(f"  最大回撤: {result['max_drawdown']:.2%}")


if __name__ == "__main__":
    test_top_combinations()
