#!/usr/bin/env python3
"""
測試所有二階導數特徵的組合

測試 Price_2nd, Vol_2nd, MACD_2nd 的所有可能組合：
- 單一特徵
- 兩兩 AND 組合
- 兩兩 OR 組合
- 三個 AND 組合
- 三個 OR 組合
- 其他複雜組合
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from improved_stage1 import load_data, find_local_extremes
from analyze_price_2nd import calculate_price_2nd, extract_price_2nd_features
from analyze_vol_2nd import calculate_vol_2nd
from analyze_macd_2nd import calculate_macd_2nd_zscore


def extract_all_features(df, candidates, true_reversals):
    """提取所有三個二階導數特徵"""
    print("\n計算所有二階導數特徵...")
    
    # 計算 price_2nd
    price_2nd_raw, price_2nd_smooth = calculate_price_2nd(df, window=3)
    
    # 計算 vol_2nd
    vol_2nd_raw, vol_2nd_smooth = calculate_vol_2nd(df, window=3)
    
    # 計算 MACD_2nd z-score
    macd_2nd_raw, macd_2nd_zscore, macd_hist = calculate_macd_2nd_zscore(df, window=200)
    
    # 添加到 DataFrame
    df['price_2nd_smooth'] = price_2nd_smooth
    df['vol_2nd_smooth'] = vol_2nd_smooth
    df['macd_2nd_zscore'] = macd_2nd_zscore
    
    features = []
    
    for cand_idx in candidates:
        # 找出對應的反轉點
        reversal_idx = None
        for offset in [1, 2]:
            check_idx = cand_idx - offset
            if check_idx >= 0:
                reversal_idx = check_idx
                break
        
        if reversal_idx is None or reversal_idx < 5:
            continue
        
        # Price_2nd
        rev_price_2nd = df['price_2nd_smooth'].iloc[reversal_idx]
        curr_price_2nd = df['price_2nd_smooth'].iloc[cand_idx]
        price_sign_changed = (np.sign(rev_price_2nd) != np.sign(curr_price_2nd))
        
        # Vol_2nd
        rev_vol_2nd = df['vol_2nd_smooth'].iloc[reversal_idx]
        curr_vol_2nd = df['vol_2nd_smooth'].iloc[cand_idx]
        vol_sign_changed = (np.sign(rev_vol_2nd) != np.sign(curr_vol_2nd))
        
        # MACD_2nd
        rev_macd_2nd = df['macd_2nd_zscore'].iloc[reversal_idx]
        curr_macd_2nd = df['macd_2nd_zscore'].iloc[cand_idx]
        macd_sign_changed = (np.sign(rev_macd_2nd) != np.sign(curr_macd_2nd))
        
        # 判斷是否為真陽性
        is_true = reversal_idx in true_reversals
        
        features.append({
            'reversal_idx': reversal_idx,
            'is_true': is_true,
            'price_sign_changed': price_sign_changed,
            'vol_sign_changed': vol_sign_changed,
            'macd_sign_changed': macd_sign_changed,
        })
    
    return pd.DataFrame(features)


def test_all_combinations(features):
    """測試所有可能的組合"""
    
    df_true = features[features['is_true'] == True]
    total_true = len(df_true)
    baseline_precision = total_true / len(features) * 100
    
    print(f"\n{'='*80}")
    print(f"🧪 測試所有二階導數特徵組合")
    print(f"{'='*80}")
    
    print(f"\n基準:")
    print(f"  總候選點: {len(features)}")
    print(f"  真陽性: {total_true}")
    print(f"  基準精確度: {baseline_precision:.1f}%")
    
    results = []
    
    # ========== 單一特徵 ==========
    print(f"\n{'='*80}")
    print(f"📊 單一特徵")
    print(f"{'='*80}")
    
    # 1. Price_2nd
    rule = features[features['price_sign_changed'] == True]
    rule_true = len(rule[rule['is_true'] == True])
    results.append({
        'category': '單一特徵',
        'name': 'Price_2nd',
        'candidates': len(rule),
        'precision': rule_true / len(rule) * 100 if len(rule) > 0 else 0,
        'coverage': rule_true / total_true * 100
    })
    
    # 2. Vol_2nd
    rule = features[features['vol_sign_changed'] == True]
    rule_true = len(rule[rule['is_true'] == True])
    results.append({
        'category': '單一特徵',
        'name': 'Vol_2nd',
        'candidates': len(rule),
        'precision': rule_true / len(rule) * 100 if len(rule) > 0 else 0,
        'coverage': rule_true / total_true * 100
    })
    
    # 3. MACD_2nd
    rule = features[features['macd_sign_changed'] == True]
    rule_true = len(rule[rule['is_true'] == True])
    results.append({
        'category': '單一特徵',
        'name': 'MACD_2nd',
        'candidates': len(rule),
        'precision': rule_true / len(rule) * 100 if len(rule) > 0 else 0,
        'coverage': rule_true / total_true * 100
    })
    
    # ========== 兩兩 AND 組合 ==========
    print(f"\n{'='*80}")
    print(f"📊 兩兩 AND 組合")
    print(f"{'='*80}")
    
    # 4. Price AND Vol
    rule = features[(features['price_sign_changed'] == True) & 
                    (features['vol_sign_changed'] == True)]
    rule_true = len(rule[rule['is_true'] == True])
    results.append({
        'category': '兩兩 AND',
        'name': 'Price AND Vol',
        'candidates': len(rule),
        'precision': rule_true / len(rule) * 100 if len(rule) > 0 else 0,
        'coverage': rule_true / total_true * 100
    })
    
    # 5. Price AND MACD
    rule = features[(features['price_sign_changed'] == True) & 
                    (features['macd_sign_changed'] == True)]
    rule_true = len(rule[rule['is_true'] == True])
    results.append({
        'category': '兩兩 AND',
        'name': 'Price AND MACD',
        'candidates': len(rule),
        'precision': rule_true / len(rule) * 100 if len(rule) > 0 else 0,
        'coverage': rule_true / total_true * 100
    })
    
    # 6. Vol AND MACD
    rule = features[(features['vol_sign_changed'] == True) & 
                    (features['macd_sign_changed'] == True)]
    rule_true = len(rule[rule['is_true'] == True])
    results.append({
        'category': '兩兩 AND',
        'name': 'Vol AND MACD',
        'candidates': len(rule),
        'precision': rule_true / len(rule) * 100 if len(rule) > 0 else 0,
        'coverage': rule_true / total_true * 100
    })
    
    # ========== 兩兩 OR 組合 ==========
    print(f"\n{'='*80}")
    print(f"📊 兩兩 OR 組合")
    print(f"{'='*80}")
    
    # 7. Price OR Vol
    rule = features[(features['price_sign_changed'] == True) | 
                    (features['vol_sign_changed'] == True)]
    rule_true = len(rule[rule['is_true'] == True])
    results.append({
        'category': '兩兩 OR',
        'name': 'Price OR Vol',
        'candidates': len(rule),
        'precision': rule_true / len(rule) * 100 if len(rule) > 0 else 0,
        'coverage': rule_true / total_true * 100
    })
    
    # 8. Price OR MACD
    rule = features[(features['price_sign_changed'] == True) | 
                    (features['macd_sign_changed'] == True)]
    rule_true = len(rule[rule['is_true'] == True])
    results.append({
        'category': '兩兩 OR',
        'name': 'Price OR MACD',
        'candidates': len(rule),
        'precision': rule_true / len(rule) * 100 if len(rule) > 0 else 0,
        'coverage': rule_true / total_true * 100
    })
    
    # 9. Vol OR MACD
    rule = features[(features['vol_sign_changed'] == True) | 
                    (features['macd_sign_changed'] == True)]
    rule_true = len(rule[rule['is_true'] == True])
    results.append({
        'category': '兩兩 OR',
        'name': 'Vol OR MACD',
        'candidates': len(rule),
        'precision': rule_true / len(rule) * 100 if len(rule) > 0 else 0,
        'coverage': rule_true / total_true * 100
    })
    
    # ========== 三個 AND 組合 ==========
    print(f"\n{'='*80}")
    print(f"📊 三個 AND 組合")
    print(f"{'='*80}")
    
    # 10. Price AND Vol AND MACD
    rule = features[(features['price_sign_changed'] == True) & 
                    (features['vol_sign_changed'] == True) &
                    (features['macd_sign_changed'] == True)]
    rule_true = len(rule[rule['is_true'] == True])
    results.append({
        'category': '三個 AND',
        'name': 'Price AND Vol AND MACD',
        'candidates': len(rule),
        'precision': rule_true / len(rule) * 100 if len(rule) > 0 else 0,
        'coverage': rule_true / total_true * 100
    })
    
    # ========== 三個 OR 組合 ==========
    print(f"\n{'='*80}")
    print(f"📊 三個 OR 組合")
    print(f"{'='*80}")
    
    # 11. Price OR Vol OR MACD
    rule = features[(features['price_sign_changed'] == True) | 
                    (features['vol_sign_changed'] == True) |
                    (features['macd_sign_changed'] == True)]
    rule_true = len(rule[rule['is_true'] == True])
    results.append({
        'category': '三個 OR',
        'name': 'Price OR Vol OR MACD',
        'candidates': len(rule),
        'precision': rule_true / len(rule) * 100 if len(rule) > 0 else 0,
        'coverage': rule_true / total_true * 100
    })
    
    # ========== 複雜組合 ==========
    print(f"\n{'='*80}")
    print(f"📊 複雜組合")
    print(f"{'='*80}")
    
    # 12. (Price AND MACD) OR Vol
    rule = features[((features['price_sign_changed'] == True) & 
                     (features['macd_sign_changed'] == True)) |
                    (features['vol_sign_changed'] == True)]
    rule_true = len(rule[rule['is_true'] == True])
    results.append({
        'category': '複雜組合',
        'name': '(Price AND MACD) OR Vol',
        'candidates': len(rule),
        'precision': rule_true / len(rule) * 100 if len(rule) > 0 else 0,
        'coverage': rule_true / total_true * 100
    })
    
    # 13. (Price AND Vol) OR MACD
    rule = features[((features['price_sign_changed'] == True) & 
                     (features['vol_sign_changed'] == True)) |
                    (features['macd_sign_changed'] == True)]
    rule_true = len(rule[rule['is_true'] == True])
    results.append({
        'category': '複雜組合',
        'name': '(Price AND Vol) OR MACD',
        'candidates': len(rule),
        'precision': rule_true / len(rule) * 100 if len(rule) > 0 else 0,
        'coverage': rule_true / total_true * 100
    })
    
    # 14. Price AND (Vol OR MACD)
    rule = features[(features['price_sign_changed'] == True) & 
                    ((features['vol_sign_changed'] == True) |
                     (features['macd_sign_changed'] == True))]
    rule_true = len(rule[rule['is_true'] == True])
    results.append({
        'category': '複雜組合',
        'name': 'Price AND (Vol OR MACD)',
        'candidates': len(rule),
        'precision': rule_true / len(rule) * 100 if len(rule) > 0 else 0,
        'coverage': rule_true / total_true * 100
    })
    
    # 15. 至少兩個改變
    rule = features[
        (features['price_sign_changed'].astype(int) + 
         features['vol_sign_changed'].astype(int) + 
         features['macd_sign_changed'].astype(int)) >= 2
    ]
    rule_true = len(rule[rule['is_true'] == True])
    results.append({
        'category': '複雜組合',
        'name': '至少兩個改變',
        'candidates': len(rule),
        'precision': rule_true / len(rule) * 100 if len(rule) > 0 else 0,
        'coverage': rule_true / total_true * 100
    })
    
    # 轉換為 DataFrame
    results_df = pd.DataFrame(results)
    
    # 按類別顯示結果
    print(f"\n{'='*80}")
    print(f"📊 完整結果")
    print(f"{'='*80}")
    
    for category in ['單一特徵', '兩兩 AND', '兩兩 OR', '三個 AND', '三個 OR', '複雜組合']:
        cat_results = results_df[results_df['category'] == category]
        if len(cat_results) > 0:
            print(f"\n{category}:")
            print(f"{'策略':<35} {'候選點':>10} {'精確度':>10} {'覆蓋率':>10}")
            print("-"*75)
            for _, row in cat_results.iterrows():
                print(f"{row['name']:<35} {row['candidates']:>10} {row['precision']:>9.1f}% {row['coverage']:>9.1f}%")
    
    # 找出最佳策略
    print(f"\n{'='*80}")
    print(f"🏆 最佳策略")
    print(f"{'='*80}")
    
    # 最高精確度
    best_precision = results_df.loc[results_df['precision'].idxmax()]
    print(f"\n最高精確度:")
    print(f"  策略: {best_precision['name']}")
    print(f"  精確度: {best_precision['precision']:.1f}%")
    print(f"  覆蓋率: {best_precision['coverage']:.1f}%")
    print(f"  候選點: {best_precision['candidates']:.0f}")
    
    # 最高覆蓋率（精確度 > 60%）
    high_precision = results_df[results_df['precision'] > 60]
    if len(high_precision) > 0:
        best_coverage = high_precision.loc[high_precision['coverage'].idxmax()]
        print(f"\n最高覆蓋率（精確度 > 60%）:")
        print(f"  策略: {best_coverage['name']}")
        print(f"  精確度: {best_coverage['precision']:.1f}%")
        print(f"  覆蓋率: {best_coverage['coverage']:.1f}%")
        print(f"  候選點: {best_coverage['candidates']:.0f}")
    
    # 最佳平衡（F1分數）
    results_df['f1'] = 2 * (results_df['precision'] * results_df['coverage']) / (results_df['precision'] + results_df['coverage'])
    best_f1 = results_df.loc[results_df['f1'].idxmax()]
    print(f"\n最佳平衡（F1分數）:")
    print(f"  策略: {best_f1['name']}")
    print(f"  精確度: {best_f1['precision']:.1f}%")
    print(f"  覆蓋率: {best_f1['coverage']:.1f}%")
    print(f"  F1分數: {best_f1['f1']:.1f}")
    print(f"  候選點: {best_f1['candidates']:.0f}")
    
    return results_df


def main():
    """主函數"""
    # 載入數據
    df = load_data()
    if df is None:
        return
    
    true_reversals = set(np.where(df['reverse'].values)[0])
    
    print(f"✅ 載入數據")
    print(f"  總數據點: {len(df)}")
    print(f"  真實反轉點: {len(true_reversals)}")
    
    # 生成候選點
    print("\n生成候選點...")
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
    
    print(f"  候選點數量: {len(candidates)}")
    
    # 提取所有特徵
    features = extract_all_features(df, candidates, true_reversals)
    print(f"  提取了 {len(features)} 個候選點的特徵")
    
    # 測試所有組合
    results_df = test_all_combinations(features)
    
    # 保存結果
    output_file = 'experiments/reversal_detection/all_combinations_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ 結果已保存到: {output_file}")


if __name__ == '__main__':
    main()
