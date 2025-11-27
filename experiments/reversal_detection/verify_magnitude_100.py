#!/usr/bin/env python3
"""
驗證：符號改變 ⟺ |反轉幅度| > 100%

如果不相等，說明計算有問題
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from improved_stage1 import load_data, find_local_extremes
from analyze_price_2nd import calculate_price_2nd, extract_price_2nd_features


def verify_magnitude_100():
    """驗證反轉幅度 > 100% 是否等於符號改變"""
    
    # 載入數據
    df = load_data()
    if df is None:
        return
    
    true_reversals = set(np.where(df['reverse'].values)[0])
    
    # 生成候選點
    prices = df['close'].values
    extremes, peaks, valleys = find_local_extremes(prices, distance=20)
    
    candidates = []
    for i in range(20, len(df)):
        for offset in [1, 2]:
            check_idx = i - offset
            if check_idx in extremes:
                if i not in candidates:
                    candidates.append(i)
                break
    
    # 提取特徵
    features = extract_price_2nd_features(df, candidates, true_reversals)
    
    print(f"{'='*80}")
    print(f"🔍 驗證：符號改變 ⟺ |反轉幅度| > 100%")
    print(f"{'='*80}")
    
    # 統計
    total = len(features)
    sign_changed = features['sign_changed'] == True
    magnitude_100 = features['reversal_magnitude'].abs() > 1.0
    
    # 集合關係
    sign_changed_count = sign_changed.sum()
    magnitude_100_count = magnitude_100.sum()
    both_count = (sign_changed & magnitude_100).sum()
    sign_only_count = (sign_changed & ~magnitude_100).sum()
    magnitude_only_count = (~sign_changed & magnitude_100).sum()
    
    print(f"\n📊 集合統計:")
    print(f"  總候選點: {total}")
    print(f"  符號改變: {sign_changed_count} ({sign_changed_count/total*100:.1f}%)")
    print(f"  |反轉幅度| > 100%: {magnitude_100_count} ({magnitude_100_count/total*100:.1f}%)")
    print(f"  兩者都滿足: {both_count} ({both_count/total*100:.1f}%)")
    print(f"  只有符號改變: {sign_only_count} ({sign_only_count/total*100:.1f}%)")
    print(f"  只有 |反轉幅度| > 100%: {magnitude_only_count} ({magnitude_only_count/total*100:.1f}%)")
    
    # 驗證等價關係
    print(f"\n🔍 等價關係驗證:")
    if sign_only_count == 0 and magnitude_only_count == 0:
        print(f"  ✅ 符號改變 ⟺ |反轉幅度| > 100%")
        print(f"  兩個條件完全等價！")
    else:
        print(f"  ❌ 符號改變 ≠ |反轉幅度| > 100%")
        print(f"  發現不一致的情況！")
        
        # 分析不一致的原因
        if sign_only_count > 0:
            print(f"\n  問題1：有 {sign_only_count} 個點符號改變但 |反轉幅度| <= 100%")
            print(f"  這在理論上不應該發生！")
            
            # 顯示例子
            sign_only = features[sign_changed & ~magnitude_100]
            print(f"\n  例子：")
            print(f"  {'反轉點 price_2nd':>20} {'當前 price_2nd':>20} {'反轉幅度':>15} {'|反轉幅度|':>15}")
            print("-"*80)
            for _, row in sign_only.head(10).iterrows():
                rev_2nd = row['rev_price_2nd']
                curr_2nd = row['curr_price_2nd']
                magnitude = row['reversal_magnitude']
                abs_magnitude = abs(magnitude)
                print(f"  {rev_2nd:>20.6f} {curr_2nd:>20.6f} {magnitude:>14.2f} {abs_magnitude:>14.2f}")
        
        if magnitude_only_count > 0:
            print(f"\n  問題2：有 {magnitude_only_count} 個點 |反轉幅度| > 100% 但符號未改變")
            print(f"  這在理論上也不應該發生！")
            
            # 顯示例子
            magnitude_only = features[~sign_changed & magnitude_100]
            print(f"\n  例子：")
            print(f"  {'反轉點 price_2nd':>20} {'當前 price_2nd':>20} {'反轉幅度':>15} {'符號':>15}")
            print("-"*80)
            for _, row in magnitude_only.head(10).iterrows():
                rev_2nd = row['rev_price_2nd']
                curr_2nd = row['curr_price_2nd']
                magnitude = row['reversal_magnitude']
                rev_sign = np.sign(rev_2nd)
                curr_sign = np.sign(curr_2nd)
                print(f"  {rev_2nd:>20.6f} {curr_2nd:>20.6f} {magnitude:>14.2f} {rev_sign:>7.0f} → {curr_sign:<7.0f}")
    
    # 檢查邊界情況
    print(f"\n{'='*80}")
    print(f"🔍 邊界情況檢查")
    print(f"{'='*80}")
    
    # 情況1：反轉點或當前值接近零
    near_zero_threshold = 0.0001
    rev_near_zero = features['rev_price_2nd'].abs() < near_zero_threshold
    curr_near_zero = features['curr_price_2nd'].abs() < near_zero_threshold
    
    print(f"\n接近零的情況（|price_2nd| < {near_zero_threshold}）:")
    print(f"  反轉點接近零: {rev_near_zero.sum()} 個")
    print(f"  當前值接近零: {curr_near_zero.sum()} 個")
    print(f"  任一接近零: {(rev_near_zero | curr_near_zero).sum()} 個")
    
    # 這些接近零的點可能導致計算問題
    near_zero_cases = features[rev_near_zero | curr_near_zero]
    if len(near_zero_cases) > 0:
        print(f"\n  這些點的符號改變和反轉幅度:")
        print(f"  {'反轉點':>15} {'當前':>15} {'反轉幅度':>15} {'符號改變':>12} {'|幅度|>100%':>15}")
        print("-"*80)
        for _, row in near_zero_cases.head(10).iterrows():
            rev_2nd = row['rev_price_2nd']
            curr_2nd = row['curr_price_2nd']
            magnitude = row['reversal_magnitude']
            sign_change = '✓' if row['sign_changed'] else '✗'
            mag_100 = '✓' if abs(magnitude) > 1.0 else '✗'
            print(f"  {rev_2nd:>15.6f} {curr_2nd:>15.6f} {magnitude:>14.2f} {sign_change:>12} {mag_100:>15}")
    
    # 對比結果
    print(f"\n{'='*80}")
    print(f"📊 過濾效果對比")
    print(f"{'='*80}")
    
    # 方法1：符號改變
    method1 = features[sign_changed]
    method1_true = method1[method1['is_true'] == True]
    method1_precision = len(method1_true) / len(method1) * 100 if len(method1) > 0 else 0
    method1_coverage = len(method1_true) / features['is_true'].sum() * 100
    
    # 方法2：|反轉幅度| > 100%
    method2 = features[magnitude_100]
    method2_true = method2[method2['is_true'] == True]
    method2_precision = len(method2_true) / len(method2) * 100 if len(method2) > 0 else 0
    method2_coverage = len(method2_true) / features['is_true'].sum() * 100
    
    # 方法3：反轉幅度 > 0%（正向反轉）
    magnitude_0 = features['reversal_magnitude'] > 0
    method3 = features[magnitude_0]
    method3_true = method3[method3['is_true'] == True]
    method3_precision = len(method3_true) / len(method3) * 100 if len(method3) > 0 else 0
    method3_coverage = len(method3_true) / features['is_true'].sum() * 100
    
    print(f"\n{'方法':<35} {'候選點':>10} {'精確度':>10} {'覆蓋率':>10}")
    print("-"*75)
    print(f"{'符號改變':<35} {len(method1):>10} {method1_precision:>9.1f}% {method1_coverage:>9.1f}%")
    print(f"{'|反轉幅度| > 100%':<35} {len(method2):>10} {method2_precision:>9.1f}% {method2_coverage:>9.1f}%")
    print(f"{'反轉幅度 > 0%':<35} {len(method3):>10} {method3_precision:>9.1f}% {method3_coverage:>9.1f}%")
    
    # 結論
    print(f"\n{'='*80}")
    print(f"💡 結論")
    print(f"{'='*80}")
    
    if sign_only_count == 0 and magnitude_only_count == 0:
        print(f"\n✅ 驗證通過！")
        print(f"  符號改變 ⟺ |反轉幅度| > 100%")
        print(f"  兩個條件在數學上完全等價")
    else:
        print(f"\n⚠️ 發現不一致！")
        print(f"  可能的原因：")
        print(f"  1. 反轉點或當前值接近零，導致除法不穩定")
        print(f"  2. 浮點數精度問題")
        print(f"  3. 計算邏輯有誤")
        
        # 建議
        print(f"\n  建議：")
        if rev_near_zero.sum() > 0:
            print(f"  - 在計算反轉幅度時，過濾掉 |反轉點| < {near_zero_threshold} 的情況")
        print(f"  - 使用符號改變作為主要條件（更穩定）")
        print(f"  - 反轉幅度作為輔助特徵（用於分層）")


if __name__ == '__main__':
    verify_magnitude_100()
