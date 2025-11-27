#!/usr/bin/env python3
"""
SidewaysRangeStrategy 完整參數優化 (包含回撤分析)
"""

import pandas as pd
import numpy as np
from crypto_backtest import CryptoBacktester
from crypto_strategies import SidewaysRangeStrategy, StrategyConfig, CoinConfig
from main import TradingConfig
import itertools
from datetime import datetime


def calculate_max_drawdown(trades):
    """計算最大回撤"""
    if not trades:
        return 0

    # 計算累積收益曲線
    initial_capital = 10000
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
    print("🔍 SidewaysRangeStrategy 完整參數優化 (包含回撤)")

    # 初始化回測器
    config = TradingConfig()
    backtester = CryptoBacktester(config)

    # 讀取本地6個月數據
    try:
        df = pd.read_csv("data/btc_6months_15m.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        print(f"✅ 成功讀取本地6個月數據: {len(df)} 條記錄")
    except FileNotFoundError:
        print("❌ 本地數據文件不存在，嘗試獲取線上數據")
        df = backtester.get_binance_data("BTCUSDT", "15m", 6 * 30 * 24 * 4)
        if df.empty:
            print("❌ 無法獲取數據")
            return

    print(
        f"📊 數據期間: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}"
    )

    # 參數範圍
    bb_periods = [20, 30, 40, 50, 60]
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

    # 準備輸出文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/sideways_optimization_with_drawdown_{timestamp}.csv"

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
                    "atr_period": 14,
                    "max_atr_pct": 0.30,
                    "min_bb_width": min_width,
                    "max_bb_width": max_width,
                    "entry_threshold": entry_threshold,
                    "volume_period": 20,
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

            if len(trades) >= 10:  # 至少10筆交易
                max_drawdown = calculate_max_drawdown(trades)

                # 計算夏普比率
                if trades:
                    returns = [t["return_pct"] for t in trades]
                    sharpe_ratio = (
                        (np.mean(returns) * np.sqrt(252))
                        / (np.std(returns) * np.sqrt(252))
                        if np.std(returns) > 0
                        else 0
                    )
                else:
                    sharpe_ratio = 0

                # 計算綜合評分 (高收益、低回撤、高夏普、高交易次數)
                # 標準化各指標到0-1範圍，然後加權平均
                normalized_return = min(total_return / 1.0, 1.0)  # 100%為滿分
                normalized_sharpe = min(sharpe_ratio / 5.0, 1.0)  # 5.0為滿分
                normalized_trades = min(len(trades) / 300, 1.0)  # 300筆為滿分
                normalized_drawdown = max(
                    0, 1 - max_drawdown / 0.2
                )  # 20%回撤為0分

                # 加權評分: 收益30%, 夏普30%, 回撤25%, 交易次數15%
                composite_score = (
                    normalized_return * 0.30
                    + normalized_sharpe * 0.30
                    + normalized_drawdown * 0.25
                    + normalized_trades * 0.15
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
                    "composite_score": composite_score,
                }

                valid_results.append(result)

                print(
                    f"#{len(valid_results):3d} | BB({bb_period},{bb_std}) | 振幅({min_width:.1f}-{max_width:.1f}%) | "
                    f"交易:{len(trades):3d} | 收益:{total_return:6.2%} | 勝率:{win_rate:5.1%} | "
                    f"夏普:{sharpe_ratio:5.2f} | 回撤:{max_drawdown:5.2%} | 平均:{total_return/len(trades):5.3%}"
                )

        except Exception as e:
            continue

        # 進度顯示
        if count % 50 == 0:
            print(
                f"\n進度: {count}/{total_combinations} ({count/total_combinations*100:.1f}%) - 找到 {len(valid_results)} 個有效組合"
            )

    print(f"\n找到 {len(valid_results)} 個有效參數組合")

    if valid_results:
        # 保存到CSV文件
        df_results = pd.DataFrame(valid_results)
        df_results.to_csv(output_file, index=False)
        print(f"📄 結果已保存至: {output_file}")

        # 按不同指標排序並顯示前10名
        print("\n" + "=" * 120)
        print("🏅 綜合評分排序 - 前10名 (高收益+低回撤+高夏普+高交易次數):")
        print("=" * 120)

        sorted_by_composite = sorted(
            valid_results, key=lambda x: x["composite_score"], reverse=True
        )
        for i, result in enumerate(sorted_by_composite[:10], 1):
            print(
                f"\n第 {i:2d} 名 (綜合評分: {result['composite_score']:.3f}):"
            )
            print(f"  布林通道: BB({result['bb_period']}, {result['bb_std']})")
            print(
                f"  振幅範圍: {result['min_bb_width']:.1f}% - {result['max_bb_width']:.1f}%"
            )
            print(f"  交易次數: {result['trades']}")
            print(f"  總收益率: {result['total_return']:.2%}")
            print(f"  勝率: {result['win_rate']:.1%}")
            print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {result['max_drawdown']:.2%}")
            print(f"  平均每筆收益: {result['avg_return_per_trade']:.3%}")

        print("\n" + "=" * 120)
        print("🏆 按夏普比率排序 - 前10名:")
        print("=" * 120)

        sorted_by_sharpe = sorted(
            valid_results, key=lambda x: x["sharpe_ratio"], reverse=True
        )
        for i, result in enumerate(sorted_by_sharpe[:10], 1):
            print(f"\n第 {i:2d} 名:")
            print(f"  布林通道: BB({result['bb_period']}, {result['bb_std']})")
            print(
                f"  振幅範圍: {result['min_bb_width']:.1f}% - {result['max_bb_width']:.1f}%"
            )
            print(f"  交易次數: {result['trades']}")
            print(f"  總收益率: {result['total_return']:.2%}")
            print(f"  勝率: {result['win_rate']:.1%}")
            print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {result['max_drawdown']:.2%}")
            print(f"  平均每筆收益: {result['avg_return_per_trade']:.3%}")

        print("\n" + "=" * 120)
        print("🛡️ 按最大回撤排序 - 前10名 (回撤最低):")
        print("=" * 120)

        sorted_by_drawdown = sorted(
            valid_results, key=lambda x: x["max_drawdown"]
        )
        for i, result in enumerate(sorted_by_drawdown[:10], 1):
            print(f"\n第 {i:2d} 名:")
            print(f"  布林通道: BB({result['bb_period']}, {result['bb_std']})")
            print(
                f"  振幅範圍: {result['min_bb_width']:.1f}% - {result['max_bb_width']:.1f}%"
            )
            print(f"  交易次數: {result['trades']}")
            print(f"  總收益率: {result['total_return']:.2%}")
            print(f"  勝率: {result['win_rate']:.1%}")
            print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {result['max_drawdown']:.2%}")
            print(f"  平均每筆收益: {result['avg_return_per_trade']:.3%}")

        print("\n" + "=" * 120)
        print("💰 按總收益率排序 - 前10名:")
        print("=" * 120)

        sorted_by_return = sorted(
            valid_results, key=lambda x: x["total_return"], reverse=True
        )
        for i, result in enumerate(sorted_by_return[:10], 1):
            print(f"\n第 {i:2d} 名:")
            print(f"  布林通道: BB({result['bb_period']}, {result['bb_std']})")
            print(
                f"  振幅範圍: {result['min_bb_width']:.1f}% - {result['max_bb_width']:.1f}%"
            )
            print(f"  交易次數: {result['trades']}")
            print(f"  總收益率: {result['total_return']:.2%}")
            print(f"  勝率: {result['win_rate']:.1%}")
            print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {result['max_drawdown']:.2%}")
            print(f"  平均每筆收益: {result['avg_return_per_trade']:.3%}")

        # 統計摘要
        print("\n" + "=" * 120)
        print("📊 統計摘要:")
        print("=" * 120)

        returns = [r["total_return"] for r in valid_results]
        sharpes = [r["sharpe_ratio"] for r in valid_results]
        drawdowns = [r["max_drawdown"] for r in valid_results]
        trade_counts = [r["trades"] for r in valid_results]

        print(
            f"總收益率 - 最高: {max(returns):.2%}, 最低: {min(returns):.2%}, 平均: {np.mean(returns):.2%}"
        )
        print(
            f"夏普比率 - 最高: {max(sharpes):.2f}, 最低: {min(sharpes):.2f}, 平均: {np.mean(sharpes):.2f}"
        )
        print(
            f"最大回撤 - 最高: {max(drawdowns):.2%}, 最低: {min(drawdowns):.2%}, 平均: {np.mean(drawdowns):.2%}"
        )
        print(
            f"交易次數 - 最多: {max(trade_counts)}, 最少: {min(trade_counts)}, 平均: {np.mean(trade_counts):.0f}"
        )

        # 篩選優質組合 (收益>20%, 回撤<10%, 夏普>1.5, 交易>50)
        quality_results = [
            r
            for r in valid_results
            if r["total_return"] > 0.20
            and r["max_drawdown"] < 0.10
            and r["sharpe_ratio"] > 1.5
            and r["trades"] > 50
        ]

        print(
            f"\n🎯 優質組合 (收益>20%, 回撤<10%, 夏普>1.5, 交易>50): {len(quality_results)} 個"
        )

        if quality_results:
            best_quality = max(
                quality_results, key=lambda x: x["composite_score"]
            )
            print(f"\n最佳優質組合:")
            print(
                f"  BB({best_quality['bb_period']}, {best_quality['bb_std']}) | "
                f"振幅({best_quality['min_bb_width']:.1f}-{best_quality['max_bb_width']:.1f}%) | "
                f"交易:{best_quality['trades']} | 收益:{best_quality['total_return']:.2%} | "
                f"勝率:{best_quality['win_rate']:.1%} | 夏普:{best_quality['sharpe_ratio']:.2f} | "
                f"回撤:{best_quality['max_drawdown']:.2%} | 評分:{best_quality['composite_score']:.3f}"
            )
    else:
        print("❌ 沒有找到符合條件的組合")


if __name__ == "__main__":
    main()
