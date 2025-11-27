#!/usr/bin/env python3
"""
測試快速入場策略

目標：在反轉點出現後 1-2 根 K 線內入場
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from detection.fast_reversal_entry import (
    FastReversalEntry,
    UltraFastEntry,
    SmartFastEntry,
    AdaptiveFastEntry
)


def load_data():
    """載入訓練數據"""
    paths = [
        "training_data/reversal_training_data_2022-01-01_to_2025-09-01_15m.csv",
        "training_data/btc_15m_2022_2025.csv"
    ]
    
    for path in paths:
        if os.path.exists(path):
            print(f"✅ 載入數據: {path}")
            df = pd.read_csv(path)
            return df
    
    return None


def analyze_entry_quality(df, signals, reversals):
    """
    分析入場質量
    
    計算：
    1. 入場距離反轉點多遠（時間和價格）
    2. 入場後的收益
    3. 最大回撤
    """
    print("\n" + "=" * 80)
    print("📊 入場質量分析")
    print("=" * 80)
    
    signal_indices = np.where(signals)[0]
    reversal_indices = np.where(reversals)[0]
    
    if len(signal_indices) == 0:
        print("\n沒有入場信號")
        return
    
    print(f"\n入場信號數: {len(signal_indices)}")
    print(f"反轉點數: {len(reversal_indices)}")
    
    # 分析每個信號
    distances_time = []
    distances_price = []
    profits_1h = []
    profits_2h = []
    profits_4h = []
    max_drawdowns = []
    
    for sig_idx in signal_indices:
        # 找到最近的反轉點
        nearest_rev = min(reversal_indices, key=lambda x: abs(x - sig_idx))
        
        # 時間距離（K 線數）
        time_dist = sig_idx - nearest_rev
        distances_time.append(time_dist)
        
        # 價格距離
        entry_price = df['close'].iloc[sig_idx]
        reversal_price = df['close'].iloc[nearest_rev]
        price_dist = abs(entry_price - reversal_price) / reversal_price * 100
        distances_price.append(price_dist)
        
        # 計算後續收益
        for period, profit_list in [(4, profits_1h), (8, profits_2h), (16, profits_4h)]:
            if sig_idx + period < len(df):
                exit_price = df['close'].iloc[sig_idx + period]
                profit = (exit_price - entry_price) / entry_price * 100
                profit_list.append(profit)
        
        # 計算最大回撤（後續 16 根 K 線內）
        if sig_idx + 16 < len(df):
            future_prices = df['close'].iloc[sig_idx:sig_idx+17].values
            max_drawdown = (np.min(future_prices) - entry_price) / entry_price * 100
            max_drawdowns.append(max_drawdown)
    
    # 打印統計
    print(f"\n⏰ 入場時機:")
    print(f"  平均距離: {np.mean(distances_time):.1f} 根 K 線 ({np.mean(distances_time)*15:.0f} 分鐘)")
    print(f"  中位數距離: {np.median(distances_time):.1f} 根 K 線")
    print(f"  範圍: {np.min(distances_time)} - {np.max(distances_time)} 根 K 線")
    
    print(f"\n💰 價格距離:")
    print(f"  平均: {np.mean(distances_price):.3f}%")
    print(f"  中位數: {np.median(distances_price):.3f}%")
    print(f"  範圍: {np.min(distances_price):.3f}% - {np.max(distances_price):.3f}%")
    
    print(f"\n📈 後續收益:")
    
    for period, profits, label in [(4, profits_1h, "1小時"), (8, profits_2h, "2小時"), (16, profits_4h, "4小時")]:
        if profits:
            win_rate = sum(1 for p in profits if p > 0) / len(profits) * 100
            print(f"\n  {label}後:")
            print(f"    平均收益: {np.mean(profits):+.2f}%")
            print(f"    中位數收益: {np.median(profits):+.2f}%")
            print(f"    勝率: {win_rate:.1f}%")
            print(f"    最大收益: {np.max(profits):+.2f}%")
            print(f"    最大虧損: {np.min(profits):+.2f}%")
    
    if max_drawdowns:
        print(f"\n📉 最大回撤（4小時內）:")
        print(f"  平均: {np.mean(max_drawdowns):+.2f}%")
        print(f"  中位數: {np.median(max_drawdowns):+.2f}%")
        print(f"  最大: {np.min(max_drawdowns):+.2f}%")


def compare_fast_entry_strategies():
    """比較不同的快速入場策略"""
    print("=" * 80)
    print("⚡ 快速入場策略對比")
    print("=" * 80)
    
    # 載入數據
    df = load_data()
    if df is None:
        print("❌ 找不到訓練數據")
        return
    
    if 'reverse' not in df.columns:
        print("❌ 數據中沒有 'reverse' 標籤")
        return
    
    print(f"\n數據概況:")
    print(f"  總數據點: {len(df)}")
    print(f"  反轉點: {df['reverse'].sum()} ({df['reverse'].sum()/len(df)*100:.2f}%)")
    
    reversals = df['reverse'].values
    
    # 測試不同策略
    strategies = [
        UltraFastEntry(),
        FastReversalEntry(
            confirmation_candles=1,
            min_move_pct=0.0005,  # 0.05%
            max_move_pct=0.002,   # 0.2%
            name="快速入場（0.05-0.2%）"
        ),
        FastReversalEntry(
            confirmation_candles=2,
            min_move_pct=0.001,   # 0.1%
            max_move_pct=0.003,   # 0.3%
            name="快速入場（0.1-0.3%）"
        ),
        SmartFastEntry(
            confirmation_candles=2,
            min_move_pct=0.001,   # 0.1%
            require_candle_close=True,
            name="智能快速入場"
        ),
        AdaptiveFastEntry(
            confirmation_candles=2,
            base_move_pct=0.001,
            name="自適應快速入場"
        )
    ]
    
    print("\n" + "=" * 80)
    print("📊 策略對比")
    print("=" * 80)
    
    results = []
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"測試策略: {strategy.name}")
        print(f"{'='*80}")
        
        signals = strategy.detect(df)
        
        # 分析入場質量
        analyze_entry_quality(df, signals, reversals)
        
        results.append({
            'name': strategy.name,
            'signals': signals,
            'count': np.sum(signals)
        })
    
    # 總結對比
    print("\n" + "=" * 80)
    print("🏆 策略總結")
    print("=" * 80)
    
    print(f"\n{'策略':<30} {'信號數':>10} {'入場率':>10}")
    print("-" * 80)
    
    total_reversals = df['reverse'].sum()
    
    for result in results:
        entry_rate = result['count'] / total_reversals * 100 if total_reversals > 0 else 0
        print(f"{result['name']:<30} {result['count']:>10} {entry_rate:>9.1f}%")
    
    return results


def main():
    """主函數"""
    results = compare_fast_entry_strategies()
    
    print("\n" + "=" * 80)
    print("💡 建議")
    print("=" * 80)
    
    print("""
快速入場策略選擇：

1. 超快速入場（Ultra Fast）:
   - 反轉點後立即入場（下一根 K 線）
   - 優點：最快，距離反轉點最近
   - 缺點：沒有確認，可能是假突破
   - 適合：高頻交易，願意承擔更多假信號

2. 快速入場（0.05-0.2%）:
   - 1 根 K 線內確認
   - 價格移動 0.05-0.2%
   - 優點：快速且有基本確認
   - 缺點：可能錯過一些快速反轉
   - 適合：平衡速度和準確度

3. 快速入場（0.1-0.3%）:
   - 1-2 根 K 線內確認
   - 價格移動 0.1-0.3%
   - 優點：更可靠的確認
   - 缺點：入場略晚
   - 適合：追求穩定的交易者

4. 智能快速入場:
   - 結合 K 線形態確認
   - 更高的準確度
   - 適合：技術分析交易者

5. 自適應快速入場:
   - 根據市場波動自動調整
   - 高波動時更保守，低波動時更激進
   - 適合：全天候交易

建議：
- 從「快速入場（0.1-0.3%）」開始
- 觀察入場率和收益
- 根據實際表現調整參數
    """)
    
    print("=" * 80)


if __name__ == '__main__':
    main()
