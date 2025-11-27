#!/usr/bin/env python3
"""
分析被漏掉的反轉點

目標：理解為什麼某些反轉點沒被檢測到
然後改進階段 1 的檢測邏輯
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from two_stage_detection import load_data, stage1_find_candidates


def analyze_missed():
    """分析被漏掉的反轉點"""
    print("="*80)
    print("🔍 分析被漏掉的反轉點")
    print("="*80)
    
    # 載入數據
    df = load_data()
    if df is None:
        return
    
    true_reversals = np.where(df['reverse'].values)[0]
    
    # 運行階段 1
    candidates, _, _ = stage1_find_candidates(df, lookback=20)
    
    # 找出被漏掉的反轉點
    missed = [idx for idx in true_reversals if idx not in candidates]
    caught = [idx for idx in true_reversals if idx in candidates]
    
    print(f"\n統計:")
    print(f"  真實反轉點: {len(true_reversals)}")
    print(f"  檢測到: {len(caught)} ({len(caught)/len(true_reversals)*100:.1f}%)")
    print(f"  漏掉: {len(missed)} ({len(missed)/len(true_reversals)*100:.1f}%)")
    
    # 分析被漏掉的反轉點特徵
    print("\n" + "="*80)
    print("📊 被漏掉的反轉點特徵")
    print("="*80)
    
    missed_features = []
    for idx in missed:
        if idx < 20 or idx >= len(df) - 1:
            continue
        
        feature = {
            'index': idx,
            'price': df['close'].iloc[idx]
        }
        
        # 價格變化幅度
        lookback = 20
        window_prices = df['close'].iloc[idx-lookback:idx+1]
        price_range = window_prices.max() - window_prices.min()
        feature['price_range_pct'] = price_range / window_prices.mean() * 100
        
        # 是波峰還是波谷
        feature['is_peak'] = df['close'].iloc[idx] == window_prices.max()
        feature['is_valley'] = df['close'].iloc[idx] == window_prices.min()
        
        # 與左側價格的差異
        left_prices = df['close'].iloc[idx-5:idx]
        feature['left_diff_pct'] = abs(df['close'].iloc[idx] - left_prices.mean()) / left_prices.mean() * 100
        
        # 成交量
        if 'volume' in df.columns:
            volume_ma = df['volume'].iloc[idx-20:idx].mean()
            feature['volume_ratio'] = df['volume'].iloc[idx] / volume_ma if volume_ma > 0 else 1.0
        
        missed_features.append(feature)
    
    missed_df = pd.DataFrame(missed_features)
    
    # 對比：被檢測到的反轉點
    caught_features = []
    for idx in caught[:len(missed)]:  # 取相同數量
        if idx < 20 or idx >= len(df) - 1:
            continue
        
        feature = {
            'index': idx,
            'price': df['close'].iloc[idx]
        }
        
        lookback = 20
        window_prices = df['close'].iloc[idx-lookback:idx+1]
        price_range = window_prices.max() - window_prices.min()
        feature['price_range_pct'] = price_range / window_prices.mean() * 100
        
        feature['is_peak'] = df['close'].iloc[idx] == window_prices.max()
        feature['is_valley'] = df['close'].iloc[idx] == window_prices.min()
        
        left_prices = df['close'].iloc[idx-5:idx]
        feature['left_diff_pct'] = abs(df['close'].iloc[idx] - left_prices.mean()) / left_prices.mean() * 100
        
        if 'volume' in df.columns:
            volume_ma = df['volume'].iloc[idx-20:idx].mean()
            feature['volume_ratio'] = df['volume'].iloc[idx] / volume_ma if volume_ma > 0 else 1.0
        
        caught_features.append(feature)
    
    caught_df = pd.DataFrame(caught_features)
    
    # 對比統計
    print(f"\n{'特徵':<25} {'被漏掉':<15} {'被檢測到':<15} {'差異':<10}")
    print("-"*80)
    
    print(f"{'價格波動範圍 (%)':<25} {missed_df['price_range_pct'].mean():>14.2f} "
          f"{caught_df['price_range_pct'].mean():>14.2f} "
          f"{(missed_df['price_range_pct'].mean() - caught_df['price_range_pct'].mean()):>9.2f}")
    
    print(f"{'與左側差異 (%)':<25} {missed_df['left_diff_pct'].mean():>14.2f} "
          f"{caught_df['left_diff_pct'].mean():>14.2f} "
          f"{(missed_df['left_diff_pct'].mean() - caught_df['left_diff_pct'].mean()):>9.2f}")
    
    if 'volume_ratio' in missed_df.columns:
        print(f"{'成交量比率':<25} {missed_df['volume_ratio'].mean():>14.2f} "
              f"{caught_df['volume_ratio'].mean():>14.2f} "
              f"{(missed_df['volume_ratio'].mean() - caught_df['volume_ratio'].mean()):>9.2f}")
    
    # 關鍵洞察
    print("\n" + "="*80)
    print("💡 關鍵洞察")
    print("="*80)
    
    print(f"\n被漏掉的反轉點特徵:")
    print(f"  平均價格波動: {missed_df['price_range_pct'].mean():.2f}%")
    print(f"  平均與左側差異: {missed_df['left_diff_pct'].mean():.2f}%")
    
    if missed_df['price_range_pct'].mean() < caught_df['price_range_pct'].mean():
        print(f"\n⚠️  被漏掉的反轉點波動較小")
        print(f"  → 可能需要降低 prominence 閾值")
    
    if missed_df['left_diff_pct'].mean() < caught_df['left_diff_pct'].mean():
        print(f"\n⚠️  被漏掉的反轉點與左側差異較小")
        print(f"  → 這些是「溫和」的反轉，較難檢測")
    
    # 建議
    print("\n" + "="*80)
    print("🎯 改進建議")
    print("="*80)
    
    print("""
基於分析，建議：

1. 接受當前的召回率 (88.6%)
   - 被漏掉的 11.4% 可能是「溫和」反轉
   - 這些反轉幅度小，難以檢測
   - 即使檢測到，收益也可能不高

2. Focus 在階段 2
   - 階段 1 保持現狀（25,235 候選點）
   - 階段 2 大幅提升精確度（目標：從 15% 到 25%+）
   - 用更強的規則或 ML 過濾

3. 考慮止損 1.8%
   - 溫和反轉可能不值得交易
   - Focus 在明顯的反轉（幅度 > 2%）
    """)


def main():
    """主函數"""
    analyze_missed()


if __name__ == '__main__':
    main()
