#!/usr/bin/env python3
"""
分析信號強度與盈虧關係
"""

import pandas as pd
import sys
import os
import logging

logger = logging.getLogger(__name__)


def analyze_signal_strength(csv_file: str) -> None:
    """
    分析交易記錄中的信號強度與盈虧關係
    
    Args:
        csv_file: 交易記錄 CSV 文件路徑
    """
    print("=" * 80)
    print("信號強度與盈虧關係分析")
    print("=" * 80)
    
    # 讀取交易記錄
    if not os.path.exists(csv_file):
        print(f"❌ 找不到文件: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    print(f"\n✅ 讀取到 {len(df)} 筆交易記錄")
    
    # 檢查是否有 signal_strength 欄位
    if 'signal_strength' not in df.columns:
        print(f"\n⚠️  找不到 signal_strength 欄位")
        print(f"可用欄位: {list(df.columns)}")
        return
    
    # 基本統計
    print("\n" + "=" * 80)
    print("📊 基本統計")
    print("=" * 80)
    
    df['is_profit'] = df['pnl'] > 0
    total_trades = len(df)
    profit_trades = df['is_profit'].sum()
    loss_trades = total_trades - profit_trades
    
    print(f"總交易數: {total_trades}")
    print(f"獲利交易: {profit_trades} ({profit_trades/total_trades:.1%})")
    print(f"虧損交易: {loss_trades} ({loss_trades/total_trades:.1%})")
    print(f"平均盈虧: {df['pnl'].mean():.2f}")
    print(f"總盈虧: {df['pnl'].sum():.2f}")
    
    # 信號強度統計
    print("\n" + "=" * 80)
    print("🔍 信號強度統計")
    print("=" * 80)
    
    print(f"平均信號強度: {df['signal_strength'].mean():.4f}")
    print(f"信號強度範圍: {df['signal_strength'].min():.4f} - {df['signal_strength'].max():.4f}")
    print(f"信號強度中位數: {df['signal_strength'].median():.4f}")
    print(f"信號強度標準差: {df['signal_strength'].std():.4f}")
    
    # 獲利 vs 虧損的信號強度對比
    print("\n" + "=" * 80)
    print("💰 獲利 vs 虧損交易對比")
    print("=" * 80)
    
    if profit_trades > 0:
        profit_df = df[df['is_profit']]
        print(f"\n✅ 獲利交易 ({profit_trades}筆):")
        print(f"  平均信號強度: {profit_df['signal_strength'].mean():.4f}")
        print(f"  中位數: {profit_df['signal_strength'].median():.4f}")
        print(f"  標準差: {profit_df['signal_strength'].std():.4f}")
        print(f"  平均盈利: {profit_df['pnl'].mean():.2f}")
    
    if loss_trades > 0:
        loss_df = df[~df['is_profit']]
        print(f"\n❌ 虧損交易 ({loss_trades}筆):")
        print(f"  平均信號強度: {loss_df['signal_strength'].mean():.4f}")
        print(f"  中位數: {loss_df['signal_strength'].median():.4f}")
        print(f"  標準差: {loss_df['signal_strength'].std():.4f}")
        print(f"  平均虧損: {loss_df['pnl'].mean():.2f}")
    
    # 信號強度差異
    if profit_trades > 0 and loss_trades > 0:
        strength_diff = profit_df['signal_strength'].mean() - loss_df['signal_strength'].mean()
        print(f"\n📈 信號強度差異: {strength_diff:+.4f}")
        if strength_diff > 0:
            print("   → 獲利交易的信號強度較高 ✅")
        else:
            print("   → 虧損交易的信號強度較高 ⚠️")
    
    # 按信號強度分組分析
    print("\n" + "=" * 80)
    print("📊 信號強度分組分析")
    print("=" * 80)
    
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float('inf')]
    labels = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', 
              '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0', '>1.0']
    
    df['strength_bin'] = pd.cut(df['signal_strength'], bins=bins, labels=labels)
    
    analysis = df.groupby('strength_bin', observed=True).agg({
        'pnl': ['count', 'mean', 'sum'],
        'is_profit': 'mean'
    }).round(4)
    
    analysis.columns = ['交易數', '平均盈虧', '總盈虧', '勝率']
    
    print("\n信號強度區間分析:")
    print(analysis.to_string())
    
    # 找出最佳信號強度區間
    print("\n" + "=" * 80)
    print("🎯 最佳信號強度區間")
    print("=" * 80)
    
    best_winrate = analysis['勝率'].idxmax()
    best_profit = analysis['平均盈虧'].idxmax()
    most_trades = analysis['交易數'].idxmax()
    
    print(f"最高勝率區間: {best_winrate} (勝率={analysis.loc[best_winrate, '勝率']:.1%})")
    print(f"最高平均盈虧區間: {best_profit} (平均={analysis.loc[best_profit, '平均盈虧']:.2f})")
    print(f"交易最多區間: {most_trades} (數量={int(analysis.loc[most_trades, '交易數'])})")
    
    # 建議閾值
    print("\n" + "=" * 80)
    print("💡 信號強度閾值建議")
    print("=" * 80)
    
    # 計算不同閾值的效果
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print("\n不同閾值的過濾效果:")
    print(f"{'閾值':<8} {'保留交易':<12} {'勝率':<10} {'平均盈虧':<12} {'總盈虧':<12}")
    print("-" * 60)
    
    for threshold in thresholds:
        filtered = df[df['signal_strength'] >= threshold]
        if len(filtered) > 0:
            win_rate = (filtered['pnl'] > 0).mean()
            avg_pnl = filtered['pnl'].mean()
            total_pnl = filtered['pnl'].sum()
            kept_pct = len(filtered) / len(df)
            
            print(f"{threshold:<8.1f} {len(filtered):>5} ({kept_pct:>5.1%}) "
                  f"{win_rate:>8.1%}  {avg_pnl:>10.2f}  {total_pnl:>10.2f}")
    
    # 推薦閾值
    print("\n" + "=" * 80)
    print("🎯 推薦策略")
    print("=" * 80)
    
    # 找出能提升勝率的最低閾值
    original_winrate = (df['pnl'] > 0).mean()
    
    best_threshold = None
    best_improvement = 0
    
    for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        filtered = df[df['signal_strength'] >= threshold]
        if len(filtered) >= 10:  # 至少保留10筆交易
            filtered_winrate = (filtered['pnl'] > 0).mean()
            improvement = filtered_winrate - original_winrate
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_threshold = threshold
    
    if best_threshold:
        filtered = df[df['signal_strength'] >= best_threshold]
        print(f"\n推薦信號強度閾值: >= {best_threshold:.2f}")
        print(f"  原始勝率: {original_winrate:.1%}")
        print(f"  過濾後勝率: {(filtered['pnl'] > 0).mean():.1%}")
        print(f"  勝率提升: {best_improvement:+.1%}")
        print(f"  保留交易: {len(filtered)}/{len(df)} ({len(filtered)/len(df):.1%})")
        print(f"  平均盈虧: {filtered['pnl'].mean():.2f}")
    else:
        print("\n⚠️  未找到能顯著提升勝率的閾值")
        print("   建議: 當前策略組合已經較優，無需額外過濾")
    
    # 保存分析結果
    output_file = "output/signal_strength_analysis.csv"
    analysis.to_csv(output_file)
    print(f"\n💾 分析結果已保存: {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)


def main():
    """命令行入口"""
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # 默認使用最新的 Combined 文件
        import glob
        files = glob.glob("output/trades_*_Combined_*.csv")
        if files:
            csv_file = max(files, key=os.path.getctime)
            print(f"📁 使用最新的交易記錄: {csv_file}\n")
        else:
            print("❌ 找不到交易記錄文件")
            print("用法: python utils/signal_strength_analyzer.py [csv_file]")
            return
    
    analyze_signal_strength(csv_file)


if __name__ == "__main__":
    main()
