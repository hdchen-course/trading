#!/usr/bin/env python3
"""
驗證反轉幅度 vs 符號改變的關係

證明：反轉幅度 >= 150% 是符號改變的真子集
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from improved_stage1 import load_data, find_local_extremes
from analyze_price_2nd import calculate_price_2nd, extract_price_2nd_features


def verify_relationship():
    """驗證反轉幅度和符號改變的關係"""
    
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
    print(f"🔍 驗證：反轉幅度 >= 150% vs 符號改變")
    print(f"{'='*80}")
    
    # 統計
    total = len(features)
    sign_changed = features['sign_changed'] == True
    magnitude_150 = features['reversal_magnitude'] >= 1.5
    
    # 集合關係
    sign_changed_count = sign_changed.sum()
    magnitude_150_count = magnitude_150.sum()
    both_count = (sign_changed & magnitude_150).sum()
    sign_only_count = (sign_changed & ~magnitude_150).sum()
    magnitude_only_count = (~sign_changed & magnitude_150).sum()
    
    print(f"\n📊 集合統計:")
    print(f"  總候選點: {total}")
    print(f"  符號改變: {sign_changed_count} ({sign_changed_count/total*100:.1f}%)")
    print(f"  反轉幅度 >= 150%: {magnitude_150_count} ({magnitude_150_count/total*100:.1f}%)")
    print(f"  兩者都滿足: {both_count} ({both_count/total*100:.1f}%)")
    print(f"  只有符號改變: {sign_only_count} ({sign_only_count/total*100:.1f}%)")
    print(f"  只有反轉幅度 >= 150%: {magnitude_only_count} ({magnitude_only_count/total*100:.1f}%)")
    
    # 驗證子集關係
    print(f"\n🔍 子集關係驗證:")
    if magnitude_only_count == 0:
        print(f"  ✅ 反轉幅度 >= 150% ⊆ 符號改變")
        print(f"  所有滿足反轉幅度 >= 150% 的點都滿足符號改變")
    else:
        print(f"  ❌ 反轉幅度 >= 150% ⊄ 符號改變")
        print(f"  有 {magnitude_only_count} 個點滿足反轉幅度但不滿足符號改變")
    
    # 分析被過濾掉的部分（符號改變但反轉幅度 < 150%）
    sign_only = features[sign_changed & ~magnitude_150]
    sign_only_true = sign_only[sign_only['is_true'] == True]
    
    print(f"\n📉 被過濾掉的部分（符號改變但反轉幅度 < 150%）:")
    print(f"  候選點數: {len(sign_only)}")
    print(f"  真陽性數: {len(sign_only_true)}")
    print(f"  精確度: {len(sign_only_true)/len(sign_only)*100:.1f}%" if len(sign_only) > 0 else "  精確度: N/A")
    
    # 反轉幅度分布
    print(f"\n📊 被過濾部分的反轉幅度分布:")
    if len(sign_only) > 0:
        print(f"  最小值: {sign_only['reversal_magnitude'].min():.2f}")
        print(f"  25分位: {sign_only['reversal_magnitude'].quantile(0.25):.2f}")
        print(f"  中位數: {sign_only['reversal_magnitude'].median():.2f}")
        print(f"  75分位: {sign_only['reversal_magnitude'].quantile(0.75):.2f}")
        print(f"  最大值: {sign_only['reversal_magnitude'].max():.2f}")
    
    # 對比兩種過濾方法
    print(f"\n{'='*80}")
    print(f"📊 兩種過濾方法對比")
    print(f"{'='*80}")
    
    # 方法1：符號改變
    method1 = features[sign_changed]
    method1_true = method1[method1['is_true'] == True]
    method1_precision = len(method1_true) / len(method1) * 100 if len(method1) > 0 else 0
    method1_coverage = len(method1_true) / features['is_true'].sum() * 100
    
    # 方法2：反轉幅度 >= 150%
    method2 = features[magnitude_150]
    method2_true = method2[method2['is_true'] == True]
    method2_precision = len(method2_true) / len(method2) * 100 if len(method2) > 0 else 0
    method2_coverage = len(method2_true) / features['is_true'].sum() * 100
    
    print(f"\n{'方法':<30} {'候選點':>10} {'精確度':>10} {'覆蓋率':>10}")
    print("-"*70)
    print(f"{'符號改變':<30} {len(method1):>10} {method1_precision:>9.1f}% {method1_coverage:>9.1f}%")
    print(f"{'反轉幅度 >= 150%':<30} {len(method2):>10} {method2_precision:>9.1f}% {method2_coverage:>9.1f}%")
    print(f"{'差異':<30} {len(method1)-len(method2):>10} {method1_precision-method2_precision:>9.1f}% {method1_coverage-method2_coverage:>9.1f}%")
    
    # 分析為什麼符號改變更好
    print(f"\n{'='*80}")
    print(f"💡 為什麼符號改變比反轉幅度 >= 150% 更好？")
    print(f"{'='*80}")
    
    print(f"\n原因1：更高的覆蓋率")
    print(f"  符號改變捕捉了 {method1_coverage:.1f}% 的真陽性")
    print(f"  反轉幅度 >= 150% 只捕捉了 {method2_coverage:.1f}% 的真陽性")
    print(f"  損失了 {method1_coverage - method2_coverage:.1f}% 的真陽性！")
    
    print(f"\n原因2：被過濾掉的部分仍然有價值")
    if len(sign_only) > 0:
        sign_only_precision = len(sign_only_true) / len(sign_only) * 100
        baseline_precision = features['is_true'].sum() / len(features) * 100
        print(f"  被過濾部分的精確度: {sign_only_precision:.1f}%")
        print(f"  基準精確度: {baseline_precision:.1f}%")
        if sign_only_precision > baseline_precision:
            print(f"  ✅ 被過濾部分的精確度高於基準，過濾掉是損失！")
        else:
            print(f"  ❌ 被過濾部分的精確度低於基準，過濾掉是合理的")
    
    print(f"\n原因3：更簡單的條件")
    print(f"  符號改變：只需比較符號")
    print(f"  反轉幅度 >= 150%：需要計算複雜的比例")
    
    # 具體例子
    print(f"\n{'='*80}")
    print(f"📝 具體例子：為什麼符號改變 ≠ 反轉幅度 >= 150%")
    print(f"{'='*80}")
    
    # 找一些符號改變但反轉幅度 < 150% 的例子
    examples = sign_only.head(5)
    
    print(f"\n例子（符號改變但反轉幅度 < 150%）:")
    print(f"{'反轉點 price_2nd':>20} {'當前 price_2nd':>20} {'反轉幅度':>15} {'是否真陽性':>15}")
    print("-"*80)
    
    for _, row in examples.iterrows():
        rev_2nd = row['rev_price_2nd']
        curr_2nd = row['curr_price_2nd']
        magnitude = row['reversal_magnitude']
        is_true = '✓' if row['is_true'] else '✗'
        
        print(f"{rev_2nd:>20.6f} {curr_2nd:>20.6f} {magnitude:>14.1%} {is_true:>15}")
    
    print(f"\n說明：")
    print(f"  這些點都發生了符號改變（負 → 正 或 正 → 負）")
    print(f"  但反轉幅度 < 150%，因為當前值的絕對值較小")
    print(f"  例如：-2.0 → +0.5 是符號改變，但反轉幅度只有 125%")
    
    # 數學解釋
    print(f"\n{'='*80}")
    print(f"📐 數學解釋")
    print(f"{'='*80}")
    
    print(f"\n對於從負到正的符號改變：")
    print(f"  反轉點 = -A (A > 0)")
    print(f"  當前 = +B (B > 0)")
    print(f"  反轉幅度 = (B - (-A)) / A = (B + A) / A = B/A + 1")
    print(f"\n  符號改變條件：B > 0")
    print(f"  反轉幅度 >= 150% 條件：B/A + 1 >= 1.5 → B >= 0.5A")
    print(f"\n  所以：符號改變只需要 B > 0")
    print(f"        反轉幅度 >= 150% 需要 B >= 0.5A")
    print(f"        當 0 < B < 0.5A 時，符號改變但反轉幅度 < 150%")


if __name__ == '__main__':
    verify_relationship()
