#!/usr/bin/env python3
"""
SidewaysRangeStrategy 優化腳本 v2
專注於震蕩區間交易，設置最小振幅要求
"""

import pandas as pd
import itertools
import sys
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


def optimize_parameters():
    """優化參數，專注震蕩交易"""
    df = pd.read_csv("data/btc_6months_15m.csv", parse_dates=["timestamp"])
    config = OptimizeConfig()

    # 更精細的參數範圍
    param_grid = {
        "bb_period": [20, 30, 40, 50],
        "bb_std": [1.5, 2.0, 2.5],
        "atr_period": [14],
        "max_atr_pct": [0.25, 0.30, 0.35],  # 震蕩判斷
        "min_bb_width": [0.8, 1.0, 1.2],  # 最小振幅要求
        "max_bb_width": [2.0, 2.5, 3.0],  # 最大振幅
        "entry_threshold": [0.995, 0.998],  # 進場門檻
        "volume_period": [20],
        "volume_threshold": [0.8, 1.0],  # 成交量確認
    }

    results = []
    param_combinations = list(itertools.product(*param_grid.values()))
    total = len(param_combinations)

    print(f"開始優化，共 {total} 種參數組合...")
    print("實時結果 (只顯示有效交易的組合):")
    print("-" * 80)

    backtester = CryptoBacktester(config)

    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_grid.keys(), params))

        # 創建策略
        strategy_config = StrategyConfig(
            name="Sideways Range V2", params=param_dict, weight=1.0
        )
        strategy = SidewaysRangeStrategy(strategy_config)
        coin_config = CoinConfig("Bitcoin", "BTCUSDT", [strategy], 1.0)

        # 執行回測
        trades, total_return, win_rate = (
            backtester.backtest_strategy_vectorized(df, coin_config, "BTCUSDT")
        )

        # 篩選條件：至少5筆交易，收益率>0.5%
        if len(trades) >= 5 and total_return > 0.005:
            # 計算夏普比率
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
            max_drawdown = np.min(drawdown)

            result = {
                "params": param_dict,
                "trades": len(trades),
                "total_return": total_return,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": abs(max_drawdown),
                "avg_return_per_trade": total_return / len(trades),
            }
            results.append(result)

            # 實時輸出
            print(
                f"#{len(results):3d} | "
                f"BB({param_dict['bb_period']},{param_dict['bb_std']}) | "
                f"振幅({param_dict['min_bb_width']}-{param_dict['max_bb_width']}%) | "
                f"交易:{len(trades):2d} | "
                f"收益:{total_return:6.2%} | "
                f"勝率:{win_rate:5.1%} | "
                f"夏普:{sharpe_ratio:5.2f}"
            )
            sys.stdout.flush()

        # 進度提示
        if (i + 1) % 50 == 0:
            print(
                f"\n進度: {i+1}/{total} ({(i+1)/total*100:.1f}%) - 找到 {len(results)} 個有效組合"
            )

    return results


def main():
    print("SidewaysRangeStrategy 震蕩交易優化 v2")
    print("=" * 50)
    print("篩選條件: 至少5筆交易 + 收益率>0.5% + 最小振幅0.8%+")
    print("=" * 50)

    results = optimize_parameters()

    if results:
        # 按夏普比率排序
        results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

        print(f"\n找到 {len(results)} 個有效參數組合")
        print("\n前5名最佳組合:")
        print("-" * 100)

        for i, result in enumerate(results[:5]):
            params = result["params"]
            print(f"\n第 {i+1} 名:")
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
            print(f"  平均每筆收益: {result['avg_return_per_trade']:.3%}")
    else:
        print("未找到符合條件的參數組合，建議放寬篩選條件")


if __name__ == "__main__":
    main()
