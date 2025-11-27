#!/usr/bin/env python3
"""
根據180天回測結果，選出勝率最高的10個策略並計算權重分配
"""

import pandas as pd
import sys
import os

def main():
    print("=" * 80)
    print("選擇勝率最高的10個策略並分配權重")
    print("=" * 80)
    
    # 讀取180天回測結果
    csv_file = "output/backtest_results_debug_20251127_202335.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ 找不到文件: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    print(f"\n✅ 讀取到 {len(df)} 個策略的回測結果")
    
    # 按勝率排序，選出前10名
    df_sorted = df.sort_values('win_rate', ascending=False)
    top10 = df_sorted.head(10).copy()
    
    print("\n" + "=" * 80)
    print("📊 勝率最高的10個策略")
    print("=" * 80)
    
    for idx, row in top10.iterrows():
        print(f"{row.name + 1:2d}. {row['strategy']:40s} "
              f"勝率={row['win_rate']:.2%} "
              f"交易={row['trades']:4d} "
              f"報酬={row['total_return']:+.2%}")
    
    # 計算權重分配
    # 方法：綜合考慮勝率(60%)和報酬率(40%)
    print("\n" + "=" * 80)
    print("💡 權重分配策略")
    print("=" * 80)
    print("計算公式: 綜合分數 = 勝率 × 60% + 報酬率 × 40%")
    
    # 標準化報酬率（避免負值影響）
    top10['return_normalized'] = top10['total_return'].clip(lower=0)
    
    # 計算綜合分數
    top10['combined_score'] = (
        top10['win_rate'] * 0.6 + 
        top10['return_normalized'] * 0.4
    )
    
    # 歸一化權重
    total_score = top10['combined_score'].sum()
    if total_score > 0:
        top10['weight'] = top10['combined_score'] / total_score
    else:
        top10['weight'] = 1.0 / len(top10)
    
    print("\n" + "=" * 80)
    print("🎯 最終權重分配")
    print("=" * 80)
    
    for idx, row in top10.iterrows():
        print(f"{row['strategy']:40s} "
              f"勝率={row['win_rate']:.2%} "
              f"報酬={row['total_return']:+6.2%} "
              f"→ 權重={row['weight']:.4f} ({row['weight']*100:.2f}%)")
    
    # 驗證權重總和
    weight_sum = top10['weight'].sum()
    print(f"\n✅ 權重總和驗證: {weight_sum:.6f} (應該 = 1.0)")
    
    # 保存結果
    output_file = "output/top10_strategies_weights.csv"
    top10[['strategy', 'trades', 'total_return', 'win_rate', 'weight']].to_csv(
        output_file, index=False
    )
    print(f"\n💾 結果已保存: {output_file}")
    
    # 生成配置代碼
    print("\n" + "=" * 80)
    print("📝 配置代碼（可直接用於 coin_configs.py）")
    print("=" * 80)
    print("\n# Top 10 策略配置（按勝率優化）")
    
    for idx, row in top10.iterrows():
        strategy_name = row['strategy']
        weight = row['weight']
        
        # 根據策略名稱生成對應的類名和配置
        if strategy_name == "Triple Confirmation Reversal":
            print(f"TripleConfirmationStrategy(weight={weight:.4f}),  # 勝率={row['win_rate']:.1%}")
        elif strategy_name == "Reversal A/V Turn":
            print(f"ReversalAVTurnStrategy(weight={weight:.4f}),  # 勝率={row['win_rate']:.1%}")
        elif strategy_name == "ML Reversal A/V Turn":
            print(f"MLReversalAVTurnStrategy(weight={weight:.4f}),  # 勝率={row['win_rate']:.1%}")
        elif strategy_name == "MACD Volume Detection":
            print(f"MACDVolumeDetectionStrategy(weight={weight:.4f}),  # 勝率={row['win_rate']:.1%}")
        elif strategy_name == "MACD Reversal Detection":
            print(f"MACDReversalDetectionStrategy(weight={weight:.4f}),  # 勝率={row['win_rate']:.1%}")
        elif strategy_name == "LSTM Detection":
            print(f"LSTMDetectionStrategy(weight={weight:.4f}),  # 勝率={row['win_rate']:.1%}")
        elif strategy_name == "Stochastic RSI":
            print(f"StochasticRSIStrategy(weight={weight:.4f}),  # 勝率={row['win_rate']:.1%}")
        elif strategy_name == "8H Crash Prediction":
            print(f"CrashPredictionStrategy(weight={weight:.4f}),  # 勝率={row['win_rate']:.1%}")
        elif strategy_name == "ATR Breakout":
            print(f"ATRBreakoutStrategy(weight={weight:.4f}),  # 勝率={row['win_rate']:.1%}")
        elif strategy_name == "Trend ML":
            print(f"TrendMLStrategy(weight={weight:.4f}),  # 勝率={row['win_rate']:.1%}")
        else:
            # 通用格式
            class_name = strategy_name.replace(" ", "").replace("-", "") + "Strategy"
            print(f"{class_name}(weight={weight:.4f}),  # 勝率={row['win_rate']:.1%}")
    
    print("\n" + "=" * 80)
    print("✅ 完成！")
    print("=" * 80)
    
    # 統計信息
    print("\n📈 統計摘要:")
    print(f"  平均勝率: {top10['win_rate'].mean():.2%}")
    print(f"  平均報酬: {top10['total_return'].mean():.2%}")
    print(f"  總交易數: {top10['trades'].sum()}")
    print(f"  最高勝率: {top10['win_rate'].max():.2%} ({top10.loc[top10['win_rate'].idxmax(), 'strategy']})")
    print(f"  最高報酬: {top10['total_return'].max():.2%} ({top10.loc[top10['total_return'].idxmax(), 'strategy']})")

if __name__ == "__main__":
    main()
