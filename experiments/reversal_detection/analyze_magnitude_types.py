#!/usr/bin/env python3
"""
分析反轉幅度的不同類型

|反轉幅度| > 100% 包含：
1. 符號改變（真正的反轉）
2. 同向增強（趨勢加強）
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from improved_stage1 import load_data, find_local_extremes
from analyze_price_2nd import calculate_price_2nd, extract_price_2nd_features


def analyze_magnitude_types():
    """分析反轉幅度的不同類型"""
    
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
    print(f"🔍 分析：|反轉幅度| > 100% 的不同類型")
    print(f"{'='*80}")
    
    # 分類
    sign_changed = features['sign_changed'] == True
    magnitude_100 = features['reversal_magnitude'].abs() > 1.0
    
    # 類型1：符號改變（真正的反轉）
    type1 = features[sign_changed & magnitude_100]
    type1_true = type1[type1['is_true'] == True]
    
    # 類型2：同向增強（不是反轉）
    type2 = features[~sign_changed & magnitude_100]
    type2_true = type2[type2['is_true'] == True]
    
    # 類型3：符號未改變且幅度 <= 100%
    type3 = features[~sign_changed & ~magnitude_100]
    type3_true = type3[type3['is_true'] == True]
    
    print(f"\n📊 分類統計:")
    print(f"{'類型':<40} {'候選點':>10} {'真陽性':>10} {'精確度':>10} {'覆蓋率':>10}")
    print("-"*90)
    
    total_true = features['is_true'].sum()
    
    # 類型1
    type1_precision = len(type1_true) / len(type1) * 100 if len(type1) > 0 else 0
    type1_coverage = len(type1_true) / total_true * 100
    print(f"{'類型1：符號改變（真反轉）':<40} {len(type1):>10} {len(type1_true):>10} {type1_precision:>9.1f}% {type1_coverage:>9.1f}%")
    
    # 類型2
    type2_precision = len(type2_true) / len(type2) * 100 if len(type2) > 0 else 0
    type2_coverage = len(type2_true) / total_true * 100
    print(f"{'類型2：同向增強（假反轉）':<40} {len(type2):>10} {len(type2_true):>10} {type2_precision:>9.1f}% {type2_coverage:>9.1f}%")
    
    # 類型3
    type3_precision = len(type3_true) / len(type3) * 100 if len(type3) > 0 else 0
    type3_coverage = len(type3_true) / total_true * 100
    print(f"{'類型3：同向減弱':<40} {len(type3):>10} {len(type3_true):>10} {type3_precision:>9.1f}% {type3_coverage:>9.1f}%")
    
    print(f"\n{'總計':<40} {len(features):>10} {total_true:>10} {total_true/len(features)*100:>9.1f}% {'100.0%':>10}")
    
    # 分析類型2（同向增強）
    print(f"\n{'='*80}")
    print(f"🔍 類型2 詳細分析：同向增強（不是真正的反轉）")
    print(f"{'='*80}")
    
    # 正向增強 vs 負向增強
    type2_positive = type2[(type2['rev_price_2nd'] > 0) & (type2['curr_price_2nd'] > 0)]
    type2_negative = type2[(type2['rev_price_2nd'] < 0) & (type2['curr_price_2nd'] < 0)]
    
    print(f"\n子類型:")
    print(f"  正向增強（正 → 更正）: {len(type2_positive)} 個")
    print(f"  負向增強（負 → 更負）: {len(type2_negative)} 個")
    
    # 顯示例子
    print(f"\n正向增強例子（正 → 更正）:")
    print(f"{'反轉點 price_2nd':>20} {'當前 price_2nd':>20} {'反轉幅度':>15} {'是否真陽性':>15}")
    print("-"*80)
    for _, row in type2_positive.head(5).iterrows():
        rev_2nd = row['rev_price_2nd']
        curr_2nd = row['curr_price_2nd']
        magnitude = row['reversal_magnitude']
        is_true = '✓' if row['is_true'] else '✗'
        print(f"{rev_2nd:>20.6f} {curr_2nd:>20.6f} {magnitude:>14.2f} {is_true:>15}")
    
    print(f"\n負向增強例子（負 → 更負）:")
    print(f"{'反轉點 price_2nd':>20} {'當前 price_2nd':>20} {'反轉幅度':>15} {'是否真陽性':>15}")
    print("-"*80)
    for _, row in type2_negative.head(5).iterrows():
        rev_2nd = row['rev_price_2nd']
        curr_2nd = row['curr_price_2nd']
        magnitude = row['reversal_magnitude']
        is_true = '✓' if row['is_true'] else '✗'
        print(f"{rev_2nd:>20.6f} {curr_2nd:>20.6f} {magnitude:>14.2f} {is_true:>15}")
    
    # 對比過濾效果
    print(f"\n{'='*80}")
    print(f"📊 過濾效果對比")
    print(f"{'='*80}")
    
    methods = [
        ('無過濾（基準）', features),
        ('符號改變', features[sign_changed]),
        ('|反轉幅度| > 100%', features[magnitude_100]),
        ('反轉幅度 > 100%（正向）', features[features['reversal_magnitude'] > 1.0]),
        ('反轉幅度 < -100%（負向）', features[features['reversal_magnitude'] < -1.0]),
        ('反轉幅度 >= 150%', features[features['reversal_magnitude'] >= 1.5]),
    ]
    
    print(f"\n{'方法':<35} {'候選點':>10} {'精確度':>10} {'覆蓋率':>10}")
    print("-"*75)
    
    for name, data in methods:
        if len(data) > 0:
            data_true = data[data['is_true'] == True]
            precision = len(data_true) / len(data) * 100
            coverage = len(data_true) / total_true * 100
            print(f"{name:<35} {len(data):>10} {precision:>9.1f}% {coverage:>9.1f}%")
    
    # 結論
    print(f"\n{'='*80}")
    print(f"💡 關鍵發現")
    print(f"{'='*80}")
    
    print(f"\n1. |反轉幅度| > 100% ≠ 符號改變")
    print(f"   |反轉幅度| > 100% 包含兩種情況：")
    print(f"   - 符號改變（真反轉）：{len(type1)} 個，精確度 {type1_precision:.1f}%")
    print(f"   - 同向增強（假反轉）：{len(type2)} 個，精確度 {type2_precision:.1f}%")
    
    print(f"\n2. 同向增強會降低精確度")
    print(f"   符號改變：精確度 {type1_precision:.1f}%")
    print(f"   |反轉幅度| > 100%：精確度 {(len(type1_true)+len(type2_true))/(len(type1)+len(type2))*100:.1f}%")
    print(f"   差異：{type1_precision - (len(type1_true)+len(type2_true))/(len(type1)+len(type2))*100:.1f}%")
    
    print(f"\n3. 為什麼反轉幅度 >= 150% 比符號改變差？")
    print(f"   因為 150% 是符號改變的子集，損失了覆蓋率")
    print(f"   符號改變覆蓋率：{type1_coverage:.1f}%")
    
    # 找出反轉幅度 >= 150% 的數量
    magnitude_150 = features[features['reversal_magnitude'] >= 1.5]
    magnitude_150_true = magnitude_150[magnitude_150['is_true'] == True]
    magnitude_150_coverage = len(magnitude_150_true) / total_true * 100
    print(f"   反轉幅度 >= 150% 覆蓋率：{magnitude_150_coverage:.1f}%")
    print(f"   損失：{type1_coverage - magnitude_150_coverage:.1f}%")
    
    print(f"\n4. 最佳策略")
    print(f"   ✅ 使用「符號改變」作為主要過濾條件")
    print(f"   ✅ 反轉幅度可以作為輔助特徵（用於分層或機器學習）")
    print(f"   ❌ 不要使用 |反轉幅度| > 100%（包含同向增強）")
    print(f"   ❌ 不要使用反轉幅度 >= 150%（損失覆蓋率）")


if __name__ == '__main__':
    analyze_magnitude_types()
