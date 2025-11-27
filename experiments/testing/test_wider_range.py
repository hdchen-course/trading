#!/usr/bin/env python3
"""
測試擴大振幅範圍到0.8%-2.5%的效果
"""

import pandas as pd
import numpy as np
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


def test_wider_range():
    """測試擴大振幅範圍的效果"""
    df = pd.read_csv("data/btc_6months_15m.csv", parse_dates=["timestamp"])
    config = OptimizeConfig()
    backtester = CryptoBacktester(config)

    # 原始最佳參數 (1.0-2.5%)
    original_params = {
        "bb_period": 40,
        "bb_std": 1.5,
        "atr_period": 14,
        "max_atr_pct": 0.30,
        "min_bb_width": 1.0,
        "max_bb_width": 2.5,
        "entry_threshold": 0.995,
        "volume_period": 20,
        "volume_threshold": 0.8,
    }

    # 擴大振幅範圍 (0.8-2.5%)
    wider_params = {
        "bb_period": 40,
        "bb_std": 1.5,
        "atr_period": 14,
        "max_atr_pct": 0.30,
        "min_bb_width": 0.8,  # 從1.0降到0.8
        "max_bb_width": 2.5,
        "entry_threshold": 0.995,
        "volume_period": 20,
        "volume_threshold": 0.8,
    }

    results = []

    for name, params in [
        ("原始(1.0-2.5%)", original_params),
        ("擴大(0.8-2.5%)", wider_params),
    ]:
        # 創建策略
        strategy_config = StrategyConfig(name=name, params=params, weight=1.0)
        strategy = SidewaysRangeStrategy(strategy_config)
        coin_config = CoinConfig("Bitcoin", "BTCUSDT", [strategy], 1.0)

        # 執行回測
        trades, total_return, win_rate = (
            backtester.backtest_strategy_vectorized(df, coin_config, "BTCUSDT")
        )

        if trades:
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
                "name": name,
                "params": params,
                "trades": len(trades),
                "total_return": total_return,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "avg_return_per_trade": total_return / len(trades),
            }
            results.append(result)

    # 顯示對比結果
    print("振幅範圍對比測試")
    print("=" * 80)

    for result in results:
        print(f"\n📊 {result['name']}:")
        print(
            f"  振幅範圍: {result['params']['min_bb_width']}% - {result['params']['max_bb_width']}%"
        )
        print(f"  交易次數: {result['trades']}")
        print(f"  總收益率: {result['total_return']:.2%}")
        print(f"  勝率: {result['win_rate']:.1%}")
        print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {result['max_drawdown']:.2%}")
        print(f"  平均每筆: {result['avg_return_per_trade']:.3%}")

    # 計算改善幅度
    if len(results) == 2:
        original = results[0]
        wider = results[1]

        print(f"\n📈 擴大振幅範圍的影響:")
        print(
            f"  交易次數變化: {wider['trades'] - original['trades']:+d} ({(wider['trades']/original['trades']-1)*100:+.1f}%)"
        )
        print(
            f"  收益率變化: {(wider['total_return'] - original['total_return'])*100:+.2f}% ({(wider['total_return']/original['total_return']-1)*100:+.1f}%)"
        )
        print(
            f"  勝率變化: {(wider['win_rate'] - original['win_rate'])*100:+.1f}%"
        )
        print(
            f"  夏普比率變化: {wider['sharpe_ratio'] - original['sharpe_ratio']:+.2f}"
        )
        print(
            f"  回撤變化: {(wider['max_drawdown'] - original['max_drawdown'])*100:+.2f}%"
        )

        # 綜合評估
        print(f"\n🎯 綜合評估:")
        if (
            wider["total_return"] > original["total_return"]
            and wider["max_drawdown"] <= original["max_drawdown"]
        ):
            print("  ✅ 擴大振幅範圍效果很好！收益提升且風險未增加")
        elif wider["total_return"] > original["total_return"]:
            print("  ⚠️  擴大振幅範圍提升了收益，但風險也增加了")
        else:
            print("  ❌ 擴大振幅範圍沒有帶來明顯好處")


if __name__ == "__main__":
    test_wider_range()
