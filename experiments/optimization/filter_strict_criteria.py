#!/usr/bin/env python3
"""
嚴格篩選：回撤<4.5%, 交易>=100, 收益>=20%, 夏普>=1.8
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


def find_strict_combinations():
    """找出符合嚴格條件的組合"""
    df = pd.read_csv("data/btc_6months_15m.csv", parse_dates=["timestamp"])
    config = OptimizeConfig()
    backtester = CryptoBacktester(config)

    # 縮小搜索範圍，專注於可能符合條件的參數
    param_grid = {
        "bb_period": [30, 40, 50],
        "bb_std": [1.5, 2.0, 2.5],
        "atr_period": [14],
        "max_atr_pct": [0.25, 0.30, 0.35],
        "min_bb_width": [0.8, 1.0, 1.2],
        "max_bb_width": [2.0, 2.5, 3.0],
        "entry_threshold": [0.995, 0.998],
        "volume_period": [20],
        "volume_threshold": [0.8, 1.0],
    }

    param_combinations = list(itertools.product(*param_grid.values()))
    total = len(param_combinations)

    print(f"嚴格篩選條件:")
    print(f"- 最大回撤 < 4.5%")
    print(f"- 交易次數 >= 100")
    print(f"- 總收益率 >= 20%")
    print(f"- 夏普比率 >= 1.8")
    print(f"\n開始搜索 {total} 種組合...")
    print("=" * 80)

    qualified_results = []

    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_grid.keys(), params))

        # 創建策略
        strategy_config = StrategyConfig(
            name="Strict Filter", params=param_dict, weight=1.0
        )
        strategy = SidewaysRangeStrategy(strategy_config)
        coin_config = CoinConfig("Bitcoin", "BTCUSDT", [strategy], 1.0)

        # 執行回測
        trades, total_return, win_rate = (
            backtester.backtest_strategy_vectorized(df, coin_config, "BTCUSDT")
        )

        if trades and len(trades) >= 100 and total_return >= 0.20:
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

            # 檢查是否符合所有條件
            if max_drawdown < 0.045 and sharpe_ratio >= 1.8:
                result = {
                    "params": param_dict,
                    "trades": len(trades),
                    "total_return": total_return,
                    "win_rate": win_rate,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "avg_return_per_trade": total_return / len(trades),
                }
                qualified_results.append(result)

                print(f"✅ 符合條件 #{len(qualified_results)}:")
                print(
                    f"   BB({param_dict['bb_period']},{param_dict['bb_std']}) 振幅({param_dict['min_bb_width']}-{param_dict['max_bb_width']}%)"
                )
                print(
                    f"   交易:{len(trades)} | 收益:{total_return:.2%} | 勝率:{win_rate:.1%} | 夏普:{sharpe_ratio:.2f} | 回撤:{max_drawdown:.2%}"
                )

        if (i + 1) % 50 == 0:
            print(
                f"進度: {i+1}/{total} ({(i+1)/total*100:.1f}%) - 找到 {len(qualified_results)} 個符合條件"
            )

    return qualified_results


def main():
    results = find_strict_combinations()

    if results:
        # 按綜合評分排序 (收益率 * 夏普比率 / 回撤)
        results.sort(
            key=lambda x: (
                x["total_return"] * x["sharpe_ratio"] / x["max_drawdown"]
            ),
            reverse=True,
        )

        print(f"\n🎯 找到 {len(results)} 個符合嚴格條件的組合")
        print("=" * 80)

        for i, result in enumerate(results):
            params = result["params"]
            score = (
                result["total_return"]
                * result["sharpe_ratio"]
                / result["max_drawdown"]
            )

            print(f"\n第 {i+1} 名 (綜合評分: {score:.1f}):")
            print(f"  布林通道: BB({params['bb_period']}, {params['bb_std']})")
            print(
                f"  振幅範圍: {params['min_bb_width']}% - {params['max_bb_width']}%"
            )
            print(f"  進場門檻: {params['entry_threshold']}")
            print(f"  成交量門檻: {params['volume_threshold']}")
            print(f"  交易次數: {result['trades']}")
            print(f"  總收益率: {result['total_return']:.2%}")
            print(f"  勝率: {result['win_rate']:.1%}")
            print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {result['max_drawdown']:.2%}")
            print(f"  平均每筆: {result['avg_return_per_trade']:.3%}")
    else:
        print("\n❌ 沒有找到符合所有嚴格條件的組合")
        print("建議放寬部分條件重新搜索")


if __name__ == "__main__":
    main()
