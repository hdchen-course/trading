#!/usr/bin/env python3
"""
SidewaysRangeStrategy 參數優化腳本
使用網格搜索找出最佳參數組合
"""

import pandas as pd
import itertools
from dataclasses import dataclass
from crypto_backtest import CryptoBacktester
from crypto_strategies import SidewaysRangeStrategy, StrategyConfig, CoinConfig


@dataclass
class OptimizeConfig:
    """優化配置"""

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


def load_data():
    """載入BTC數據"""
    return pd.read_csv("data/btc_6months_15m.csv", parse_dates=["timestamp"])


def optimize_parameters():
    """優化SidewaysRangeStrategy參數"""
    df = load_data()
    config = OptimizeConfig()

    # 參數範圍
    param_grid = {
        "bb_period": [20, 30, 40, 50, 60],
        "bb_std": [1.5, 2.0, 2.5],
        "atr_period": [14],  # 固定ATR週期
        "max_atr_pct": [0.20, 0.25, 0.30, 0.35],
        "max_bb_width": [1.0, 1.5, 2.0],
        "entry_threshold": [0.990, 0.995, 0.998],
        "volume_period": [20],  # 固定成交量週期
        "volume_threshold": [0.6, 0.8, 1.0],
    }

    best_result = None
    best_params = None

    # 網格搜索
    param_combinations = list(itertools.product(*param_grid.values()))
    total = len(param_combinations)

    print(f"開始優化，共 {total} 種參數組合...")

    backtester = CryptoBacktester(config)

    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_grid.keys(), params))

        # 創建策略
        strategy_config = StrategyConfig(
            name="Sideways Range Optimization", params=param_dict, weight=1.0
        )
        strategy = SidewaysRangeStrategy(strategy_config)

        # 創建幣種配置
        coin_config = CoinConfig("Bitcoin", "BTCUSDT", [strategy], 1.0)

        # 執行回測
        trades, total_return, win_rate = (
            backtester.backtest_strategy_vectorized(df, coin_config, "BTCUSDT")
        )

        # 計算夏普比率
        if trades:
            returns = [t["leveraged_return_pct"] for t in trades]
            if returns:
                import numpy as np

                returns_array = np.array(returns)
                sharpe_ratio = (
                    np.mean(returns_array)
                    / (np.std(returns_array) + 1e-8)
                    * np.sqrt(252)
                )
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        # 評估結果
        if sharpe_ratio > 0 and total_return > 0:
            if (
                best_result is None
                or sharpe_ratio > best_result["sharpe_ratio"]
            ):
                best_result = {
                    "total_return": total_return,
                    "win_rate": win_rate,
                    "sharpe_ratio": sharpe_ratio,
                    "total_trades": len(trades),
                }
                best_params = param_dict
                print(
                    f"新最佳結果! 夏普比率: {sharpe_ratio:.3f}, 收益率: {total_return:.2%}"
                )

        if (i + 1) % 10 == 0:
            print(f"進度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
            if best_result:
                print(f"目前最佳: 夏普比率 {best_result['sharpe_ratio']:.3f}")

    return best_params, best_result


def main():
    """主函數"""
    print("SidewaysRangeStrategy 參數優化")
    print("=" * 50)

    best_params, best_result = optimize_parameters()

    if best_params:
        print("\n最佳參數組合:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        print(f"\n最佳結果:")
        print(f"  總收益率: {best_result['total_return']:.2%}")
        print(f"  夏普比率: {best_result['sharpe_ratio']:.3f}")
        print(f"  勝率: {best_result['win_rate']:.2%}")
        print(f"  交易次數: {best_result['total_trades']}")
    else:
        print("未找到有效的參數組合")


if __name__ == "__main__":
    main()
