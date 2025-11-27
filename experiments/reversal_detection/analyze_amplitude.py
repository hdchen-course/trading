#!/usr/bin/env python3
"""
振幅分析

分析真實反轉點 vs 假反轉點的振幅特徵：
1. 反轉點K線的振幅 (high - low)
2. 當前K線的振幅
3. 反轉點前後的價格變化幅度
4. 振幅的相對大小（與ATR比較）
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from improved_stage1 import load_data, find_local_extremes


def calculate_atr(df, period=14):
    """計算ATR（平均真實波幅）"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR = TR的移動平均
    atr = tr.rolling(window=period, min_periods=1).mean()
    
    return atr


def extract_amplitude_features(df, candidates, true_reversals):
    """提取振幅特徵"""
    print("\n計算振幅特徵...")
    
    # 計算ATR
    atr = calculate_atr(df, period=14)
    
    features = []
    
    for cand_idx in candidates:
        # 找出對應的反轉點
        reversal_idx = None
        for offset in [1, 2]:
            check_idx = cand_idx - offset
            if check_idx >= 0:
                reversal_idx = check_idx
                break
        
        if reversal_idx is None or reversal_idx < 20:
            continue
        
        # 反轉點K線的振幅
        rev_amplitude = df['high'].iloc[reversal_idx] - df['low'].iloc[reversal_idx]
        rev_close = df['close'].iloc[reversal_idx]
        rev_amplitude_pct = (rev_amplitude / rev_close) * 100  # 百分比
        
        # 當前K線的振幅
        curr_amplitude = df['high'].iloc[cand_idx] - df['low'].iloc[cand_idx]
        curr_close = df['close'].iloc[cand_idx]
        curr_amplitude_pct = (curr_amplitude / curr_close) * 100
        
        # ATR相對振幅
        rev_atr = atr.iloc[reversal_idx]
        curr_atr = atr.iloc[cand_idx]
        rev_amplitude_vs_atr = rev_amplitude / rev_atr if rev_atr > 0 else 0
        curr_amplitude_vs_atr = curr_amplitude / curr_atr if curr_atr > 0 else 0
        
        # 反轉點前後的價格變化
        # 反轉前的價格變化（前5根K線）
        if reversal_idx >= 5:
            price_before = df['close'].iloc[reversal_idx-5:reversal_idx].mean()
            price_change_before = abs(rev_close - price_before) / price_before * 100
        else:
            price_change_before = 0
        
        # 反轉後的價格變化（反轉點到當前）
        price_change_after = abs(curr_close - rev_close) / rev_close * 100
        
        # 判斷反轉點類型
        lookback = 20
        if reversal_idx >= lookback:
            window_prices = df['close'].iloc[reversal_idx-lookback:reversal_idx+1].values
            rev_price_val = df['close'].iloc[reversal_idx]
            is_peak = rev_price_val == np.max(window_prices)
        else:
            is_peak = None
        
        # 判斷是否為真陽性
        is_true = reversal_idx in true_reversals
        
        features.append({
            'reversal_idx': reversal_idx,
            'is_true': is_true,
            'is_peak': is_peak,
            # 絕對振幅
            'rev_amplitude': rev_amplitude,
            'curr_amplitude': curr_amplitude,
            # 百分比振幅
            'rev_amplitude_pct': rev_amplitude_pct,
            'curr_amplitude_pct': curr_amplitude_pct,
            # 相對ATR的振幅
            'rev_amplitude_vs_atr': rev_amplitude_vs_atr,
            'curr_amplitude_vs_atr': curr_amplitude_vs_atr,
            # 價格變化
            'price_change_before': price_change_before,
            'price_change_after': price_change_after,
            # 振幅比率
            'amplitude_ratio': curr_amplitude / rev_amplitude if rev_amplitude > 0 else 0,
        })
    
    return pd.DataFrame(features)


def analyze_amplitude_comparison(features):
    """對比真假反轉點的振幅特徵"""
    
    df_true = features[features['is_true'] == True]
    df_false = features[features['is_true'] == False]
    
    print(f"\n{'='*80}")
    print(f"📊 振幅特徵對比：真實 vs 假反轉點")
    print(f"{'='*80}")
    
    print(f"\n樣本數量:")
    print(f"  真陽性: {len(df_true)}")
    print(f"  假陽性: {len(df_false)}")
    print(f"  基準精確度: {len(df_true) / len(features) * 100:.1f}%")
    
    # 1. 反轉點K線振幅
    print(f"\n{'='*80}")
    print(f"1. 反轉點K線振幅")
    print(f"{'='*80}")
    
    print(f"\n{'特徵':<40} {'真陽性':<15} {'假陽性':<15} {'差異':<10}")
    print("-"*80)
    
    true_rev_amp = df_true['rev_amplitude_pct'].mean()
    false_rev_amp = df_false['rev_amplitude_pct'].mean()
    print(f"{'反轉點振幅 (%)':<40} {true_rev_amp:>14.3f} {false_rev_amp:>14.3f} {true_rev_amp-false_rev_amp:>9.3f}")
    
    true_rev_atr = df_true['rev_amplitude_vs_atr'].mean()
    false_rev_atr = df_false['rev_amplitude_vs_atr'].mean()
    print(f"{'反轉點振幅/ATR':<40} {true_rev_atr:>14.3f} {false_rev_atr:>14.3f} {true_rev_atr-false_rev_atr:>9.3f}")
    
    # 2. 當前K線振幅
    print(f"\n{'='*80}")
    print(f"2. 當前K線振幅")
    print(f"{'='*80}")
    
    print(f"\n{'特徵':<40} {'真陽性':<15} {'假陽性':<15} {'差異':<10}")
    print("-"*80)
    
    true_curr_amp = df_true['curr_amplitude_pct'].mean()
    false_curr_amp = df_false['curr_amplitude_pct'].mean()
    print(f"{'當前振幅 (%)':<40} {true_curr_amp:>14.3f} {false_curr_amp:>14.3f} {true_curr_amp-false_curr_amp:>9.3f}")
    
    true_curr_atr = df_true['curr_amplitude_vs_atr'].mean()
    false_curr_atr = df_false['curr_amplitude_vs_atr'].mean()
    print(f"{'當前振幅/ATR':<40} {true_curr_atr:>14.3f} {false_curr_atr:>14.3f} {true_curr_atr-false_curr_atr:>9.3f}")
    
    # 3. 價格變化
    print(f"\n{'='*80}")
    print(f"3. 價格變化幅度")
    print(f"{'='*80}")
    
    print(f"\n{'特徵':<40} {'真陽性':<15} {'假陽性':<15} {'差異':<10}")
    print("-"*80)
    
    true_before = df_true['price_change_before'].mean()
    false_before = df_false['price_change_before'].mean()
    print(f"{'反轉前價格變化 (%)':<40} {true_before:>14.3f} {false_before:>14.3f} {true_before-false_before:>9.3f}")
    
    true_after = df_true['price_change_after'].mean()
    false_after = df_false['price_change_after'].mean()
    print(f"{'反轉後價格變化 (%)':<40} {true_after:>14.3f} {false_after:>14.3f} {true_after-false_after:>9.3f}")
    
    # 4. 振幅比率
    print(f"\n{'='*80}")
    print(f"4. 振幅比率")
    print(f"{'='*80}")
    
    print(f"\n{'特徵':<40} {'真陽性':<15} {'假陽性':<15} {'差異':<10}")
    print("-"*80)
    
    true_ratio = df_true['amplitude_ratio'].mean()
    false_ratio = df_false['amplitude_ratio'].mean()
    print(f"{'當前振幅/反轉點振幅':<40} {true_ratio:>14.3f} {false_ratio:>14.3f} {true_ratio-false_ratio:>9.3f}")
    
    # 5. 分布分析
    print(f"\n{'='*80}")
    print(f"5. 振幅分布分析")
    print(f"{'='*80}")
    
    # 反轉點振幅分布
    print(f"\n反轉點振幅/ATR 分布:")
    print(f"{'區間':<20} {'真陽性':<15} {'假陽性':<15} {'真陽性比例':<15}")
    print("-"*65)
    
    bins = [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 100]
    labels = ['<0.5', '0.5-0.8', '0.8-1.0', '1.0-1.2', '1.2-1.5', '1.5-2.0', '>2.0']
    
    for i, label in enumerate(labels):
        true_count = len(df_true[(df_true['rev_amplitude_vs_atr'] >= bins[i]) & 
                                  (df_true['rev_amplitude_vs_atr'] < bins[i+1])])
        false_count = len(df_false[(df_false['rev_amplitude_vs_atr'] >= bins[i]) & 
                                    (df_false['rev_amplitude_vs_atr'] < bins[i+1])])
        total = true_count + false_count
        true_pct = (true_count / total * 100) if total > 0 else 0
        
        print(f"{label:<20} {true_count:<15} {false_count:<15} {true_pct:<14.1f}%")
    
    # 當前振幅分布
    print(f"\n當前振幅/ATR 分布:")
    print(f"{'區間':<20} {'真陽性':<15} {'假陽性':<15} {'真陽性比例':<15}")
    print("-"*65)
    
    for i, label in enumerate(labels):
        true_count = len(df_true[(df_true['curr_amplitude_vs_atr'] >= bins[i]) & 
                                  (df_true['curr_amplitude_vs_atr'] < bins[i+1])])
        false_count = len(df_false[(df_false['curr_amplitude_vs_atr'] >= bins[i]) & 
                                    (df_false['curr_amplitude_vs_atr'] < bins[i+1])])
        total = true_count + false_count
        true_pct = (true_count / total * 100) if total > 0 else 0
        
        print(f"{label:<20} {true_count:<15} {false_count:<15} {true_pct:<14.1f}%")


def test_amplitude_filters(features):
    """測試基於振幅的過濾規則"""
    
    df_true = features[features['is_true'] == True]
    
    print(f"\n{'='*80}")
    print(f"💡 基於振幅的過濾規則")
    print(f"{'='*80}")
    
    rules = []
    
    # 規則1：反轉點振幅 > 1.0 ATR
    rule1 = features[features['rev_amplitude_vs_atr'] > 1.0]
    rule1_true = len(rule1[rule1['is_true'] == True])
    rule1_precision = rule1_true / len(rule1) * 100 if len(rule1) > 0 else 0
    rule1_coverage = rule1_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '反轉點振幅 > 1.0 ATR',
        'candidates': len(rule1),
        'precision': rule1_precision,
        'coverage': rule1_coverage
    })
    
    # 規則2：反轉點振幅 > 1.2 ATR
    rule2 = features[features['rev_amplitude_vs_atr'] > 1.2]
    rule2_true = len(rule2[rule2['is_true'] == True])
    rule2_precision = rule2_true / len(rule2) * 100 if len(rule2) > 0 else 0
    rule2_coverage = rule2_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '反轉點振幅 > 1.2 ATR',
        'candidates': len(rule2),
        'precision': rule2_precision,
        'coverage': rule2_coverage
    })
    
    # 規則3：反轉點振幅 > 1.5 ATR
    rule3 = features[features['rev_amplitude_vs_atr'] > 1.5]
    rule3_true = len(rule3[rule3['is_true'] == True])
    rule3_precision = rule3_true / len(rule3) * 100 if len(rule3) > 0 else 0
    rule3_coverage = rule3_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '反轉點振幅 > 1.5 ATR',
        'candidates': len(rule3),
        'precision': rule3_precision,
        'coverage': rule3_coverage
    })
    
    # 規則4：當前振幅 > 1.0 ATR
    rule4 = features[features['curr_amplitude_vs_atr'] > 1.0]
    rule4_true = len(rule4[rule4['is_true'] == True])
    rule4_precision = rule4_true / len(rule4) * 100 if len(rule4) > 0 else 0
    rule4_coverage = rule4_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '當前振幅 > 1.0 ATR',
        'candidates': len(rule4),
        'precision': rule4_precision,
        'coverage': rule4_coverage
    })
    
    # 規則5：反轉點和當前都 > 1.0 ATR
    rule5 = features[
        (features['rev_amplitude_vs_atr'] > 1.0) & 
        (features['curr_amplitude_vs_atr'] > 1.0)
    ]
    rule5_true = len(rule5[rule5['is_true'] == True])
    rule5_precision = rule5_true / len(rule5) * 100 if len(rule5) > 0 else 0
    rule5_coverage = rule5_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '反轉點和當前都 > 1.0 ATR',
        'candidates': len(rule5),
        'precision': rule5_precision,
        'coverage': rule5_coverage
    })
    
    # 規則6：反轉後價格變化 > 2%
    rule6 = features[features['price_change_after'] > 2.0]
    rule6_true = len(rule6[rule6['is_true'] == True])
    rule6_precision = rule6_true / len(rule6) * 100 if len(rule6) > 0 else 0
    rule6_coverage = rule6_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '反轉後價格變化 > 2%',
        'candidates': len(rule6),
        'precision': rule6_precision,
        'coverage': rule6_coverage
    })
    
    # 規則7：組合規則
    rule7 = features[
        (features['rev_amplitude_vs_atr'] > 1.0) & 
        (features['price_change_after'] > 1.5)
    ]
    rule7_true = len(rule7[rule7['is_true'] == True])
    rule7_precision = rule7_true / len(rule7) * 100 if len(rule7) > 0 else 0
    rule7_coverage = rule7_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '反轉點振幅>1.0 ATR + 價格變化>1.5%',
        'candidates': len(rule7),
        'precision': rule7_precision,
        'coverage': rule7_coverage
    })
    
    # 🆕 規則8：過濾小振幅（反轉點振幅 >= 0.5 ATR）
    rule8 = features[features['rev_amplitude_vs_atr'] >= 0.5]
    rule8_true = len(rule8[rule8['is_true'] == True])
    rule8_precision = rule8_true / len(rule8) * 100 if len(rule8) > 0 else 0
    rule8_coverage = rule8_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '🆕 過濾小振幅（反轉點 >= 0.5 ATR）',
        'candidates': len(rule8),
        'precision': rule8_precision,
        'coverage': rule8_coverage
    })
    
    # 🆕 規則9：過濾小振幅（當前振幅 >= 0.5 ATR）
    rule9 = features[features['curr_amplitude_vs_atr'] >= 0.5]
    rule9_true = len(rule9[rule9['is_true'] == True])
    rule9_precision = rule9_true / len(rule9) * 100 if len(rule9) > 0 else 0
    rule9_coverage = rule9_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '🆕 過濾小振幅（當前 >= 0.5 ATR）',
        'candidates': len(rule9),
        'precision': rule9_precision,
        'coverage': rule9_coverage
    })
    
    # 🆕 規則10：兩者都過濾小振幅
    rule10 = features[
        (features['rev_amplitude_vs_atr'] >= 0.5) & 
        (features['curr_amplitude_vs_atr'] >= 0.5)
    ]
    rule10_true = len(rule10[rule10['is_true'] == True])
    rule10_precision = rule10_true / len(rule10) * 100 if len(rule10) > 0 else 0
    rule10_coverage = rule10_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '🆕 兩者都 >= 0.5 ATR',
        'candidates': len(rule10),
        'precision': rule10_precision,
        'coverage': rule10_coverage
    })
    
    # 顯示結果
    print(f"\n{'規則':<45} {'候選點':>10} {'精確度':>10} {'覆蓋率':>10}")
    print("-"*85)
    
    for rule in rules:
        print(f"{rule['name']:<45} {rule['candidates']:>10} {rule['precision']:>9.1f}% {rule['coverage']:>9.1f}%")
    
    print(f"\n基準 (無過濾):")
    baseline_precision = len(df_true) / len(features) * 100
    print(f"  候選點: {len(features)}")
    print(f"  精確度: {baseline_precision:.1f}%")
    print(f"  覆蓋率: 100.0%")


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
    
    # 提取振幅特徵
    features = extract_amplitude_features(df, candidates, true_reversals)
    print(f"  提取了 {len(features)} 個候選點的特徵")
    
    # 分析振幅對比
    analyze_amplitude_comparison(features)
    
    # 測試振幅過濾規則
    test_amplitude_filters(features)


if __name__ == '__main__':
    main()
