#!/usr/bin/env python3
"""
對比：實時檢測 vs 事後分析

目標：揭示兩者的差異
- 事後分析：使用已知的反轉點標籤（作弊）
- 實時檢測：不使用標籤，純粹基於當前數據判斷
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from detection.fast_reversal_entry import FastReversalEntry
from detection.realtime_reversal_detection import RealtimeReversalDetection


def load_data():
    """載入訓練數據"""
    path = "training_data/reversal_training_data_2022-01-01_to_2025-09-01_15m.csv"
    
    if os.path.exists(path):
        print(f"✅ 載入數據: {path}")
        return pd.read_csv(path)
    
    return None


def analyze_performance(df, signals, name):
    """分析性能"""
    signal_indices = np.where(signals)[0]
    
    if len(signal_indices) == 0:
        print(f"\n{name}: 沒有信號")
        return None
    
    print(f"\n{'='*80}")
    print(f"📊 {name}")
    print(f"{'='*80}")
    
    print(f"\n信號數: {len(signal_indices)}")
    
    # 計算後續收益
    periods = [4, 8, 16]  # 1h, 2h, 4h
    
    for period in periods:
        hours = period * 15 / 60
        profits = []
        
        for sig_idx in signal_indices:
            if sig_idx + period < len(df):
                entry_price = df['close'].iloc[sig_idx]
                exit_price = df['close'].iloc[sig_idx + period]
                profit = (exit_price - entry_price) / entry_price * 100
                profits.append(profit)
        
        if profits:
            win_rate = sum(1 for p in profits if p > 0) / len(profits) * 100
            print(f"\n{hours:.1f} 小時後:")
            print(f"  平均收益: {np.mean(profits):+.2f}%")
            print(f"  中位數收益: {np.median(profits):+.2f}%")
            print(f"  勝率: {win_rate:.1f}%")
            print(f"  最大收益: {np.max(profits):+.2f}%")
            print(f"  最大虧損: {np.min(profits):+.2f}%")
    
    return {
        'name': name,
        'signal_count': len(signal_indices),
        'avg_profit_1h': np.mean([
            (df['close'].iloc[i+4] - df['close'].iloc[i]) / df['close'].iloc[i] * 100
            for i in signal_indices if i+4 < len(df)
        ])
    }


def compare_hindsight_vs_realtime():
    """對比事後分析 vs 實時檢測"""
    print("=" * 80)
    print("🔬 事後分析 vs 實時檢測對比")
    print("=" * 80)
    
    # 載入數據
    df = load_data()
    if df is None:
        print("❌ 找不到數據")
        return
    
    print(f"\n數據概況:")
    print(f"  總數據點: {len(df)}")
    
    if 'reverse' in df.columns:
        print(f"  已知反轉點: {df['reverse'].sum()} ({df['reverse'].sum()/len(df)*100:.2f}%)")
    
    # 1. 事後分析（使用已知標籤）
    print(f"\n" + "="*80)
    print("1️⃣  事後分析（使用已知反轉點標籤）")
    print("="*80)
    print("⚠️  這是作弊！實盤時我們沒有這些標籤")
    
    hindsight_detector = FastReversalEntry(
        confirmation_candles=2,
        min_move_pct=0.001,
        max_move_pct=0.003,
        name="事後分析（作弊）"
    )
    
    hindsight_signals = hindsight_detector.detect(df)
    hindsight_result = analyze_performance(df, hindsight_signals, "事後分析（作弊）")
    
    # 2. 實時檢測（不使用標籤）
    print(f"\n" + "="*80)
    print("2️⃣  實時檢測（不使用標籤）")
    print("="*80)
    print("✅ 這才是實盤情況")
    
    realtime_detector = RealtimeReversalDetection(
        lookback_extreme=20,
        min_reversal_pct=0.001,
        max_reversal_pct=0.003,
        require_volume=True,
        volume_threshold=1.0,
        require_rsi=False
    )
    
    realtime_signals = realtime_detector.detect(df)
    realtime_result = analyze_performance(df, realtime_signals, "實時檢測（真實）")
    
    # 3. 對比
    print(f"\n" + "="*80)
    print("📊 對比總結")
    print("="*80)
    
    if hindsight_result and realtime_result:
        print(f"\n{'方法':<30} {'信號數':>10} {'1h平均收益':>15}")
        print("-" * 80)
        print(f"{hindsight_result['name']:<30} {hindsight_result['signal_count']:>10} {hindsight_result['avg_profit_1h']:>14.2f}%")
        print(f"{realtime_result['name']:<30} {realtime_result['signal_count']:>10} {realtime_result['avg_profit_1h']:>14.2f}%")
    
    # 4. 計算重疊率
    if 'reverse' in df.columns:
        print(f"\n" + "="*80)
        print("🎯 準確度分析")
        print("="*80)
        
        true_reversals = df['reverse'].values
        
        # 事後分析的準確度
        hindsight_tp = np.sum(hindsight_signals & true_reversals)
        hindsight_precision = hindsight_tp / np.sum(hindsight_signals) * 100 if np.sum(hindsight_signals) > 0 else 0
        
        # 實時檢測的準確度
        realtime_tp = np.sum(realtime_signals & true_reversals)
        realtime_precision = realtime_tp / np.sum(realtime_signals) * 100 if np.sum(realtime_signals) > 0 else 0
        
        print(f"\n事後分析:")
        print(f"  命中真實反轉點: {hindsight_tp}/{np.sum(hindsight_signals)}")
        print(f"  精確度: {hindsight_precision:.1f}%")
        
        print(f"\n實時檢測:")
        print(f"  命中真實反轉點: {realtime_tp}/{np.sum(realtime_signals)}")
        print(f"  精確度: {realtime_precision:.1f}%")
    
    print(f"\n" + "="*80)
    print("💡 結論")
    print("="*80)
    
    print("""
事後分析（使用已知標籤）:
  - 這是「作弊」，因為我們提前知道哪裡是反轉點
  - 只能用於回測和分析
  - 無法用於實盤交易
  - 結果：勝率 ~50%，收益接近 0
  
實時檢測（不使用標籤）:
  - 這才是真實的實盤情況
  - 需要實時判斷哪裡可能是反轉點
  - 會有更多假信號
  - 但這是唯一可行的方法
  
關鍵洞察:
  1. 即使「完美」知道反轉點，1-2 根 K 線後入場的收益也不高
  2. 這說明反轉後的短期走勢是隨機的
  3. 需要更長的持有時間才能獲利
  4. 或者需要更好的入場時機選擇
    """)


def main():
    """主函數"""
    compare_hindsight_vs_realtime()
    
    print("\n" + "="*80)
    print("🎯 建議")
    print("="*80)
    
    print("""
基於這個分析，我建議：

1. 不要依賴「完美」的反轉點檢測
   - 即使知道反轉點，短期收益也不高
   
2. 使用實時檢測方法
   - RealtimeReversalDetection
   - 接受會有假信號
   
3. 改進策略：
   - 延長持有時間（4-8 小時）
   - 使用更嚴格的過濾條件
   - 結合趨勢判斷
   - 設定合理的止損和止盈
   
4. 現實期望：
   - 勝率 50-55% 是合理的
   - 需要用盈虧比來賺錢（例如 2:1 或 3:1）
   - 不要期望每次都對
    """)
    
    print("="*80)


if __name__ == '__main__':
    main()
