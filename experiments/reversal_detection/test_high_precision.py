#!/usr/bin/env python3
"""
測試高精確度檢測器

展示如何使用新的高精確度反轉檢測器
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from detection.high_precision_reversal import (
    HighPrecisionReversalDetection,
    BalancedReversalDetection,
    UltraConservativeReversalDetection
)
from detection.optimal_reversal import OptimalReversalDetection


def load_test_data():
    """載入測試數據"""
    data_path = "training_data/reversal_training_data_2022-01-01_to_2025-09-01_15m.csv"
    
    print("📥 載入測試數據...")
    df = pd.read_csv(data_path)
    
    # 轉換時間戳
    df['datetime'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ 載入完成：{len(df)} 個數據點")
    print(f"   時間範圍：{df['datetime'].min()} 到 {df['datetime'].max()}")
    
    return df


def compare_detectors(df):
    """比較不同檢測器"""
    print("\n" + "=" * 80)
    print("📊 比較檢測器性能")
    print("=" * 80)
    
    # 創建檢測器
    detectors = {
        'Baseline': OptimalReversalDetection(),
        'High Precision': HighPrecisionReversalDetection(),
        'Balanced': BalancedReversalDetection(),
        'Ultra Conservative': UltraConservativeReversalDetection()
    }
    
    results = {}
    
    print(f"\n{'檢測器':<25} {'信號數':>10} {'反轉點':>10} {'精確度':>10}")
    print("-" * 80)
    
    for name, detector in detectors.items():
        # 檢測
        signals = detector.detect(df)
        
        # 統計
        signal_count = signals.sum()
        reversal_count = df['reverse'].sum()
        true_positives = (signals & df['reverse']).sum()
        precision = true_positives / signal_count if signal_count > 0 else 0
        
        results[name] = {
            'signals': signals,
            'signal_count': signal_count,
            'true_positives': true_positives,
            'precision': precision
        }
        
        print(f"{name:<25} {signal_count:>10} {true_positives:>10} {precision:>9.2%}")
    
    return results


def visualize_recent_signals(df, results, days=30):
    """視覺化最近的信號"""
    print(f"\n" + "=" * 80)
    print(f"📈 視覺化最近 {days} 天的信號")
    print("=" * 80)
    
    # 取最近 N 天的數據
    recent_df = df[df['datetime'] >= df['datetime'].max() - pd.Timedelta(days=days)].copy()
    
    # 創建圖表
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    
    colors = {
        'Baseline': 'blue',
        'High Precision': 'green',
        'Balanced': 'orange',
        'Ultra Conservative': 'red'
    }
    
    markers = {
        'Baseline': 'o',
        'High Precision': '^',
        'Balanced': 'D',
        'Ultra Conservative': 'v'
    }
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        # 繪製價格曲線
        ax.plot(recent_df['datetime'], recent_df['close'], 
                'k-', linewidth=1, alpha=0.5, label='Price')
        
        # 繪製真實反轉點
        true_reversals = recent_df[recent_df['reverse'] == True]
        ax.scatter(true_reversals['datetime'], true_reversals['close'],
                  c='gray', s=100, marker='x', alpha=0.3, 
                  label='True Reversals', zorder=3)
        
        # 繪製檢測到的信號
        recent_signals = result['signals'][-len(recent_df):]
        signal_points = recent_df[recent_signals]
        
        if len(signal_points) > 0:
            ax.scatter(signal_points['datetime'], signal_points['close'],
                      c=colors[name], s=200, marker=markers[name], 
                      label=f'{name} Signals', zorder=5)
            
            # 標註信號數量
            for _, row in signal_points.iterrows():
                ax.annotate(f'${row["close"]:.0f}',
                           xy=(row['datetime'], row['close']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, alpha=0.7,
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=colors[name], alpha=0.3))
        
        # 設置標題和標籤
        precision = result['precision']
        signal_count = len(signal_points)
        ax.set_title(f'{name} - {signal_count} signals, Precision: {precision:.2%}',
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Price (USDT)', fontsize=10)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    axes[-1].set_xlabel('Date', fontsize=10)
    plt.tight_layout()
    
    # 保存圖表
    os.makedirs('output', exist_ok=True)
    output_path = f'output/high_precision_comparison_{days}days.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"\n✅ 圖表已保存：{output_path}")
    
    return output_path


def analyze_signal_quality(df, results):
    """分析信號質量"""
    print("\n" + "=" * 80)
    print("🔍 信號質量分析")
    print("=" * 80)
    
    for name, result in results.items():
        signals = result['signals']
        signal_indices = np.where(signals)[0]
        
        if len(signal_indices) == 0:
            print(f"\n{name}: 沒有信號")
            continue
        
        print(f"\n{name}:")
        print(f"  總信號數: {len(signal_indices)}")
        print(f"  真陽性: {result['true_positives']}")
        print(f"  精確度: {result['precision']:.2%}")
        
        # 分析信號間隔
        if len(signal_indices) > 1:
            intervals = np.diff(signal_indices)
            avg_interval = intervals.mean()
            min_interval = intervals.min()
            max_interval = intervals.max()
            
            # 轉換為時間（假設 15 分鐘間隔）
            avg_hours = avg_interval * 15 / 60
            min_hours = min_interval * 15 / 60
            max_hours = max_interval * 15 / 60
            
            print(f"  平均信號間隔: {avg_hours:.1f} 小時")
            print(f"  最小間隔: {min_hours:.1f} 小時")
            print(f"  最大間隔: {max_hours:.1f} 小時")
        
        # 分析信號分佈
        signal_df = df.iloc[signal_indices]
        
        print(f"  信號價格範圍: ${signal_df['close'].min():.0f} - ${signal_df['close'].max():.0f}")
        print(f"  平均價格: ${signal_df['close'].mean():.0f}")
        
        if 'rsi' in signal_df.columns:
            print(f"  平均 RSI: {signal_df['rsi'].mean():.1f}")
        
        if 'volume_trend' in signal_df.columns:
            print(f"  平均成交量趨勢: {signal_df['volume_trend'].mean():.2f}x")


def main():
    """主函數"""
    print("=" * 80)
    print("🧪 測試高精確度反轉檢測器")
    print("=" * 80)
    
    # 1. 載入數據
    df = load_test_data()
    
    # 2. 比較檢測器
    results = compare_detectors(df)
    
    # 3. 視覺化最近 30 天
    visualize_recent_signals(df, results, days=30)
    
    # 4. 分析信號質量
    analyze_signal_quality(df, results)
    
    print("\n" + "=" * 80)
    print("✅ 測試完成！")
    print("=" * 80)
    
    print("\n💡 使用建議：")
    print("  1. High Precision: 適合保守策略，精確度高")
    print("  2. Balanced: 平衡精確度和召回率")
    print("  3. Ultra Conservative: 最高精確度，但信號很少")
    print("  4. 可以組合使用：高精確度信號大倉位，基線信號小倉位")


if __name__ == '__main__':
    main()
