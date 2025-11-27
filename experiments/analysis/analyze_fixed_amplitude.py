#!/usr/bin/env python3
"""
分析固定振幅0.8-2.0%的低回撤組合
"""

import pandas as pd


def main():
    # 讀取最新的優化結果
    df = pd.read_csv(
        "data/sideways_optimization_with_drawdown_20251109_083332.csv"
    )

    # 篩選振幅0.8-2.0%的組合
    filtered = df[(df["min_bb_width"] == 0.8) & (df["max_bb_width"] == 2.0)]

    print("🔍 振幅固定在0.8-2.0%的組合分析")
    print("=" * 80)
    print(f"符合條件的組合數量: {len(filtered)}")

    # 按回撤排序
    sorted_by_drawdown = filtered.sort_values("max_drawdown")

    print("\n🛡️ 按最大回撤排序 - 所有組合:")
    print("=" * 80)

    for i, (_, row) in enumerate(sorted_by_drawdown.iterrows(), 1):
        print(f"\n第 {i:2d} 名:")
        print(
            f"  BB({row['bb_period']}, {row['bb_std']}) | 振幅({row['min_bb_width']:.1f}-{row['max_bb_width']:.1f}%)"
        )
        print(
            f"  交易:{int(row['trades'])} | 收益:{row['total_return']:6.2%} | 勝率:{row['win_rate']:5.1%}"
        )
        print(
            f"  夏普:{row['sharpe_ratio']:5.2f} | 回撤:{row['max_drawdown']:5.2%} | 平均:{row['avg_return_per_trade']:5.3%}"
        )
        print(f"  綜合評分: {row['composite_score']:.3f}")

    # 統計摘要
    print(f"\n📊 統計摘要 (振幅0.8-2.0%):")
    print("=" * 80)
    print(
        f"收益率 - 最高: {filtered['total_return'].max():.2%}, 最低: {filtered['total_return'].min():.2%}, 平均: {filtered['total_return'].mean():.2%}"
    )
    print(
        f"回撤 - 最高: {filtered['max_drawdown'].max():.2%}, 最低: {filtered['max_drawdown'].min():.2%}, 平均: {filtered['max_drawdown'].mean():.2%}"
    )
    print(
        f"夏普 - 最高: {filtered['sharpe_ratio'].max():.2f}, 最低: {filtered['sharpe_ratio'].min():.2f}, 平均: {filtered['sharpe_ratio'].mean():.2f}"
    )
    print(
        f"交易 - 最多: {int(filtered['trades'].max())}, 最少: {int(filtered['trades'].min())}, 平均: {filtered['trades'].mean():.0f}"
    )

    # 找出最佳平衡組合 (回撤<12% + 收益>400%)
    best_balance = filtered[
        (filtered["max_drawdown"] < 0.12) & (filtered["total_return"] > 4.0)
    ]

    if len(best_balance) > 0:
        print(
            f"\n🎯 最佳平衡組合 (回撤<12% & 收益>400%): {len(best_balance)} 個"
        )
        print("=" * 80)

        best = best_balance.sort_values(
            "composite_score", ascending=False
        ).iloc[0]
        print(f"推薦配置:")
        print(
            f"  BB({int(best['bb_period'])}, {best['bb_std']}) | 振幅({best['min_bb_width']:.1f}-{best['max_bb_width']:.1f}%)"
        )
        print(
            f"  交易:{int(best['trades'])} | 收益:{best['total_return']:6.2%} | 勝率:{best['win_rate']:5.1%}"
        )
        print(
            f"  夏普:{best['sharpe_ratio']:5.2f} | 回撤:{best['max_drawdown']:5.2%} | 平均:{best['avg_return_per_trade']:5.3%}"
        )
        print(f"  綜合評分: {best['composite_score']:.3f}")


if __name__ == "__main__":
    main()
