#!/usr/bin/env python3
"""
價格二階導數分析

分析真實反轉點 vs 假反轉點的 price_2nd 特徵：
1. 反轉點的 price_2nd 值
2. 當前K線的 price_2nd 值
3. price_2nd 的變化趨勢
4. price_2nd 的符號變化（加速度方向改變）
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from improved_stage1 import load_data, find_local_extremes


def calculate_price_2nd(df, window=3):
    """
    計算價格二階導數（加速度）
    
    price_2nd = (price[i] - price[i-1]) - (price[i-1] - price[i-2])
              = price[i] - 2*price[i-1] + price[i-2]
    """
    close = df['close'].values
    
    # 一階導數（速度）
    price_1st = np.diff(close, prepend=close[0])
    
    # 二階導數（加速度）
    price_2nd = np.diff(price_1st, prepend=price_1st[0])
    
    # 平滑處理（使用移動平均）
    price_2nd_smooth = pd.Series(price_2nd).rolling(window=window, min_periods=1).mean().values
    
    return price_2nd, price_2nd_smooth


def extract_price_2nd_features(df, candidates, true_reversals):
    """提取 price_2nd 特徵"""
    print("\n計算 price_2nd 特徵...")
    
    # 計算 price_2nd
    price_2nd_raw, price_2nd_smooth = calculate_price_2nd(df, window=3)
    
    # 添加到 DataFrame
    df['price_2nd_raw'] = price_2nd_raw
    df['price_2nd_smooth'] = price_2nd_smooth
    
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
        
        # 反轉點的 price_2nd
        rev_price_2nd = df['price_2nd_smooth'].iloc[reversal_idx]
        rev_price_2nd_raw = df['price_2nd_raw'].iloc[reversal_idx]
        
        # 當前K線的 price_2nd
        curr_price_2nd = df['price_2nd_smooth'].iloc[cand_idx]
        curr_price_2nd_raw = df['price_2nd_raw'].iloc[cand_idx]
        
        # price_2nd 的變化
        price_2nd_change = curr_price_2nd - rev_price_2nd
        
        # 反轉點前後的 price_2nd 變化
        if reversal_idx >= 3:
            price_2nd_before = df['price_2nd_smooth'].iloc[reversal_idx-3:reversal_idx].mean()
        else:
            price_2nd_before = 0
        
        price_2nd_after = df['price_2nd_smooth'].iloc[reversal_idx+1:cand_idx+1].mean() if cand_idx > reversal_idx else 0
        
        # 符號變化（加速度方向改變）
        rev_sign = np.sign(rev_price_2nd)
        curr_sign = np.sign(curr_price_2nd)
        sign_changed = (rev_sign != curr_sign)
        
        # 反轉點前後的符號變化
        before_sign = np.sign(price_2nd_before)
        after_sign = np.sign(price_2nd_after)
        sign_changed_at_reversal = (before_sign != after_sign)
        
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
        
        # 歸一化 price_2nd（相對於價格）
        rev_close = df['close'].iloc[reversal_idx]
        curr_close = df['close'].iloc[cand_idx]
        rev_price_2nd_pct = (rev_price_2nd / rev_close) * 10000  # 萬分比
        curr_price_2nd_pct = (curr_price_2nd / curr_close) * 10000
        
        # 計算反轉幅度（反轉完成度）
        # 反轉幅度 = (當前 - 反轉點) / |反轉點|
        # 0% = 沒有變化
        # 100% = 完全反轉到零點
        # >100% = 超過零點，反向
        if abs(rev_price_2nd) > 0.0001:  # 避免除以零
            reversal_magnitude = (curr_price_2nd - rev_price_2nd) / abs(rev_price_2nd)
        else:
            reversal_magnitude = 0
        
        features.append({
            'reversal_idx': reversal_idx,
            'is_true': is_true,
            'is_peak': is_peak,
            # 原始 price_2nd
            'rev_price_2nd': rev_price_2nd,
            'curr_price_2nd': curr_price_2nd,
            'rev_price_2nd_raw': rev_price_2nd_raw,
            'curr_price_2nd_raw': curr_price_2nd_raw,
            # 歸一化 price_2nd
            'rev_price_2nd_pct': rev_price_2nd_pct,
            'curr_price_2nd_pct': curr_price_2nd_pct,
            # price_2nd 變化
            'price_2nd_change': price_2nd_change,
            'price_2nd_before': price_2nd_before,
            'price_2nd_after': price_2nd_after,
            # 符號變化
            'rev_sign': rev_sign,
            'curr_sign': curr_sign,
            'sign_changed': sign_changed,
            'sign_changed_at_reversal': sign_changed_at_reversal,
            # 絕對值
            'rev_price_2nd_abs': abs(rev_price_2nd),
            'curr_price_2nd_abs': abs(curr_price_2nd),
            # 🆕 反轉幅度
            'reversal_magnitude': reversal_magnitude,
        })
    
    return pd.DataFrame(features)


def analyze_price_2nd_comparison(features):
    """對比真假反轉點的 price_2nd 特徵"""
    
    df_true = features[features['is_true'] == True]
    df_false = features[features['is_true'] == False]
    
    print(f"\n{'='*80}")
    print(f"📊 price_2nd 特徵對比：真實 vs 假反轉點")
    print(f"{'='*80}")
    
    print(f"\n樣本數量:")
    print(f"  真陽性: {len(df_true)}")
    print(f"  假陽性: {len(df_false)}")
    print(f"  基準精確度: {len(df_true) / len(features) * 100:.1f}%")
    
    # 1. price_2nd 值對比
    print(f"\n{'='*80}")
    print(f"1. price_2nd 值對比")
    print(f"{'='*80}")
    
    print(f"\n{'特徵':<40} {'真陽性':<15} {'假陽性':<15} {'差異':<10}")
    print("-"*80)
    
    true_rev = df_true['rev_price_2nd_pct'].mean()
    false_rev = df_false['rev_price_2nd_pct'].mean()
    print(f"{'反轉點 price_2nd (萬分比)':<40} {true_rev:>14.3f} {false_rev:>14.3f} {true_rev-false_rev:>9.3f}")
    
    true_curr = df_true['curr_price_2nd_pct'].mean()
    false_curr = df_false['curr_price_2nd_pct'].mean()
    print(f"{'當前 price_2nd (萬分比)':<40} {true_curr:>14.3f} {false_curr:>14.3f} {true_curr-false_curr:>9.3f}")
    
    true_rev_abs = df_true['rev_price_2nd_abs'].mean()
    false_rev_abs = df_false['rev_price_2nd_abs'].mean()
    print(f"{'反轉點 |price_2nd|':<40} {true_rev_abs:>14.6f} {false_rev_abs:>14.6f} {true_rev_abs-false_rev_abs:>9.6f}")
    
    true_curr_abs = df_true['curr_price_2nd_abs'].mean()
    false_curr_abs = df_false['curr_price_2nd_abs'].mean()
    print(f"{'當前 |price_2nd|':<40} {true_curr_abs:>14.6f} {false_curr_abs:>14.6f} {true_curr_abs-false_curr_abs:>9.6f}")
    
    # 2. price_2nd 變化
    print(f"\n{'='*80}")
    print(f"2. price_2nd 變化")
    print(f"{'='*80}")
    
    print(f"\n{'特徵':<40} {'真陽性':<15} {'假陽性':<15} {'差異':<10}")
    print("-"*80)
    
    true_change = df_true['price_2nd_change'].mean()
    false_change = df_false['price_2nd_change'].mean()
    print(f"{'price_2nd 變化':<40} {true_change:>14.6f} {false_change:>14.6f} {true_change-false_change:>9.6f}")
    
    true_before = df_true['price_2nd_before'].mean()
    false_before = df_false['price_2nd_before'].mean()
    print(f"{'反轉前 price_2nd':<40} {true_before:>14.6f} {false_before:>14.6f} {true_before-false_before:>9.6f}")
    
    true_after = df_true['price_2nd_after'].mean()
    false_after = df_false['price_2nd_after'].mean()
    print(f"{'反轉後 price_2nd':<40} {true_after:>14.6f} {false_after:>14.6f} {true_after-false_after:>9.6f}")
    
    # 3. 符號變化
    print(f"\n{'='*80}")
    print(f"3. 符號變化（加速度方向改變）")
    print(f"{'='*80}")
    
    print(f"\n{'特徵':<40} {'真陽性':<15} {'假陽性':<15} {'差異':<10}")
    print("-"*80)
    
    true_sign_changed = (df_true['sign_changed'].sum() / len(df_true) * 100) if len(df_true) > 0 else 0
    false_sign_changed = (df_false['sign_changed'].sum() / len(df_false) * 100) if len(df_false) > 0 else 0
    print(f"{'反轉點到當前符號改變比例':<40} {true_sign_changed:>13.1f}% {false_sign_changed:>13.1f}% {true_sign_changed-false_sign_changed:>8.1f}%")
    
    true_sign_at_rev = (df_true['sign_changed_at_reversal'].sum() / len(df_true) * 100) if len(df_true) > 0 else 0
    false_sign_at_rev = (df_false['sign_changed_at_reversal'].sum() / len(df_false) * 100) if len(df_false) > 0 else 0
    print(f"{'反轉點處符號改變比例':<40} {true_sign_at_rev:>13.1f}% {false_sign_at_rev:>13.1f}% {true_sign_at_rev-false_sign_at_rev:>8.1f}%")
    
    # 4. 波峰 vs 波谷分析
    print(f"\n{'='*80}")
    print(f"4. 波峰 vs 波谷的 price_2nd")
    print(f"{'='*80}")
    
    # 波峰
    df_true_peak = df_true[df_true['is_peak'] == True]
    df_false_peak = df_false[df_false['is_peak'] == True]
    
    if len(df_true_peak) > 0 and len(df_false_peak) > 0:
        print(f"\n波峰 (price_2nd 應該為負):")
        print(f"  真陽性樣本: {len(df_true_peak)}")
        print(f"  假陽性樣本: {len(df_false_peak)}")
        
        true_peak_2nd = df_true_peak['rev_price_2nd'].mean()
        false_peak_2nd = df_false_peak['rev_price_2nd'].mean()
        print(f"  反轉點 price_2nd: 真 {true_peak_2nd:.6f} vs 假 {false_peak_2nd:.6f}")
        
        true_peak_neg = (df_true_peak['rev_price_2nd'] < 0).sum() / len(df_true_peak) * 100
        false_peak_neg = (df_false_peak['rev_price_2nd'] < 0).sum() / len(df_false_peak) * 100
        print(f"  price_2nd < 0 比例: 真 {true_peak_neg:.1f}% vs 假 {false_peak_neg:.1f}%")
    
    # 波谷
    df_true_valley = df_true[df_true['is_peak'] == False]
    df_false_valley = df_false[df_false['is_peak'] == False]
    
    if len(df_true_valley) > 0 and len(df_false_valley) > 0:
        print(f"\n波谷 (price_2nd 應該為正):")
        print(f"  真陽性樣本: {len(df_true_valley)}")
        print(f"  假陽性樣本: {len(df_false_valley)}")
        
        true_valley_2nd = df_true_valley['rev_price_2nd'].mean()
        false_valley_2nd = df_false_valley['rev_price_2nd'].mean()
        print(f"  反轉點 price_2nd: 真 {true_valley_2nd:.6f} vs 假 {false_valley_2nd:.6f}")
        
        true_valley_pos = (df_true_valley['rev_price_2nd'] > 0).sum() / len(df_true_valley) * 100
        false_valley_pos = (df_false_valley['rev_price_2nd'] > 0).sum() / len(df_false_valley) * 100
        print(f"  price_2nd > 0 比例: 真 {true_valley_pos:.1f}% vs 假 {false_valley_pos:.1f}%")
    
    # 5. 反轉幅度分析
    print(f"\n{'='*80}")
    print(f"5. 反轉幅度分析（反轉完成度）")
    print(f"{'='*80}")
    
    print(f"\n反轉幅度統計:")
    print(f"  真陽性平均: {df_true['reversal_magnitude'].mean():.2f}")
    print(f"  假陽性平均: {df_false['reversal_magnitude'].mean():.2f}")
    print(f"  差異: {df_true['reversal_magnitude'].mean() - df_false['reversal_magnitude'].mean():.2f}")
    
    # 反轉幅度分布
    print(f"\n反轉幅度分布:")
    print(f"{'區間':<30} {'真陽性':<15} {'假陽性':<15} {'真陽性比例':<15}")
    print("-"*75)
    
    bins = [
        (-float('inf'), 0),      # 反向（負值）
        (0, 0.5),                # 0-50%
        (0.5, 1.0),              # 50-100%
        (1.0, 1.5),              # 100-150%（完全反轉）
        (1.5, 2.0),              # 150-200%
        (2.0, float('inf'))      # >200%
    ]
    labels = ['反向（<0%）', '0-50%', '50-100%', '100-150%', '150-200%', '>200%']
    
    for (low, high), label in zip(bins, labels):
        true_count = len(df_true[(df_true['reversal_magnitude'] >= low) & 
                                  (df_true['reversal_magnitude'] < high)])
        false_count = len(df_false[(df_false['reversal_magnitude'] >= low) & 
                                    (df_false['reversal_magnitude'] < high)])
        total = true_count + false_count
        true_pct = (true_count / total * 100) if total > 0 else 0
        
        print(f"{label:<30} {true_count:<15} {false_count:<15} {true_pct:<14.1f}%")
    
    # 6. 幅度變化分析
    print(f"\n{'='*80}")
    print(f"6. price_2nd 幅度變化分析")
    print(f"{'='*80}")
    
    # 幅度減弱 vs 增強
    df_true_weaken = df_true[df_true['curr_price_2nd_abs'] < df_true['rev_price_2nd_abs']]
    df_false_weaken = df_false[df_false['curr_price_2nd_abs'] < df_false['rev_price_2nd_abs']]
    
    true_weaken_pct = len(df_true_weaken) / len(df_true) * 100 if len(df_true) > 0 else 0
    false_weaken_pct = len(df_false_weaken) / len(df_false) * 100 if len(df_false) > 0 else 0
    
    print(f"\n幅度減弱（當前 |price_2nd| < 反轉點 |price_2nd|）:")
    print(f"  真陽性: {true_weaken_pct:.1f}%")
    print(f"  假陽性: {false_weaken_pct:.1f}%")
    print(f"  差異: {true_weaken_pct - false_weaken_pct:.1f}%")
    
    # 符號改變 + 幅度變化的組合
    print(f"\n組合情況:")
    print(f"{'情況':<40} {'真陽性':<15} {'假陽性':<15} {'真陽性比例':<15}")
    print("-"*85)
    
    # 1. 符號改變 + 幅度減弱
    combo1_true = len(df_true[(df_true['sign_changed'] == True) & 
                               (df_true['curr_price_2nd_abs'] < df_true['rev_price_2nd_abs'])])
    combo1_false = len(df_false[(df_false['sign_changed'] == True) & 
                                 (df_false['curr_price_2nd_abs'] < df_false['rev_price_2nd_abs'])])
    combo1_total = combo1_true + combo1_false
    combo1_pct = combo1_true / combo1_total * 100 if combo1_total > 0 else 0
    print(f"{'符號改變 + 幅度減弱':<40} {combo1_true:<15} {combo1_false:<15} {combo1_pct:<14.1f}%")
    
    # 2. 符號改變 + 幅度增強
    combo2_true = len(df_true[(df_true['sign_changed'] == True) & 
                               (df_true['curr_price_2nd_abs'] >= df_true['rev_price_2nd_abs'])])
    combo2_false = len(df_false[(df_false['sign_changed'] == True) & 
                                 (df_false['curr_price_2nd_abs'] >= df_false['rev_price_2nd_abs'])])
    combo2_total = combo2_true + combo2_false
    combo2_pct = combo2_true / combo2_total * 100 if combo2_total > 0 else 0
    print(f"{'符號改變 + 幅度增強':<40} {combo2_true:<15} {combo2_false:<15} {combo2_pct:<14.1f}%")
    
    # 3. 符號不變 + 幅度減弱
    combo3_true = len(df_true[(df_true['sign_changed'] == False) & 
                               (df_true['curr_price_2nd_abs'] < df_true['rev_price_2nd_abs'])])
    combo3_false = len(df_false[(df_false['sign_changed'] == False) & 
                                 (df_false['curr_price_2nd_abs'] < df_false['rev_price_2nd_abs'])])
    combo3_total = combo3_true + combo3_false
    combo3_pct = combo3_true / combo3_total * 100 if combo3_total > 0 else 0
    print(f"{'符號不變 + 幅度減弱':<40} {combo3_true:<15} {combo3_false:<15} {combo3_pct:<14.1f}%")
    
    # 4. 符號不變 + 幅度增強
    combo4_true = len(df_true[(df_true['sign_changed'] == False) & 
                               (df_true['curr_price_2nd_abs'] >= df_true['rev_price_2nd_abs'])])
    combo4_false = len(df_false[(df_false['sign_changed'] == False) & 
                                 (df_false['curr_price_2nd_abs'] >= df_false['rev_price_2nd_abs'])])
    combo4_total = combo4_true + combo4_false
    combo4_pct = combo4_true / combo4_total * 100 if combo4_total > 0 else 0
    print(f"{'符號不變 + 幅度增強':<40} {combo4_true:<15} {combo4_false:<15} {combo4_pct:<14.1f}%")
    
    # 6. 分布分析
    print(f"\n{'='*80}")
    print(f"6. |price_2nd| 分布分析")
    print(f"{'='*80}")
    
    # 計算分位數
    quantiles = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
    rev_quantiles = features['rev_price_2nd_abs'].quantile(quantiles).values
    
    print(f"\n反轉點 |price_2nd| 分布:")
    print(f"{'區間':<30} {'真陽性':<15} {'假陽性':<15} {'真陽性比例':<15}")
    print("-"*75)
    
    labels = ['<25%', '25%-50%', '50%-75%', '75%-90%', '90%-95%', '>95%']
    for i, label in enumerate(labels):
        true_count = len(df_true[(df_true['rev_price_2nd_abs'] >= rev_quantiles[i]) & 
                                  (df_true['rev_price_2nd_abs'] < rev_quantiles[i+1])])
        false_count = len(df_false[(df_false['rev_price_2nd_abs'] >= rev_quantiles[i]) & 
                                    (df_false['rev_price_2nd_abs'] < rev_quantiles[i+1])])
        total = true_count + false_count
        true_pct = (true_count / total * 100) if total > 0 else 0
        
        print(f"{label:<30} {true_count:<15} {false_count:<15} {true_pct:<14.1f}%")


def test_price_2nd_filters(features):
    """測試基於 price_2nd 的過濾規則"""
    
    df_true = features[features['is_true'] == True]
    
    print(f"\n{'='*80}")
    print(f"💡 基於 price_2nd 的過濾規則")
    print(f"{'='*80}")
    
    rules = []
    
    # 計算閾值
    rev_2nd_median = features['rev_price_2nd_abs'].median()
    rev_2nd_75 = features['rev_price_2nd_abs'].quantile(0.75)
    rev_2nd_90 = features['rev_price_2nd_abs'].quantile(0.90)
    
    print(f"\n|price_2nd| 統計:")
    print(f"  中位數: {rev_2nd_median:.6f}")
    print(f"  75分位: {rev_2nd_75:.6f}")
    print(f"  90分位: {rev_2nd_90:.6f}")
    
    # 規則1：|price_2nd| > 中位數
    rule1 = features[features['rev_price_2nd_abs'] > rev_2nd_median]
    rule1_true = len(rule1[rule1['is_true'] == True])
    rule1_precision = rule1_true / len(rule1) * 100 if len(rule1) > 0 else 0
    rule1_coverage = rule1_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': f'反轉點 |price_2nd| > 中位數',
        'candidates': len(rule1),
        'precision': rule1_precision,
        'coverage': rule1_coverage
    })
    
    # 規則2：|price_2nd| > 75分位
    rule2 = features[features['rev_price_2nd_abs'] > rev_2nd_75]
    rule2_true = len(rule2[rule2['is_true'] == True])
    rule2_precision = rule2_true / len(rule2) * 100 if len(rule2) > 0 else 0
    rule2_coverage = rule2_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': f'反轉點 |price_2nd| > 75分位',
        'candidates': len(rule2),
        'precision': rule2_precision,
        'coverage': rule2_coverage
    })
    
    # 規則3：|price_2nd| > 90分位
    rule3 = features[features['rev_price_2nd_abs'] > rev_2nd_90]
    rule3_true = len(rule3[rule3['is_true'] == True])
    rule3_precision = rule3_true / len(rule3) * 100 if len(rule3) > 0 else 0
    rule3_coverage = rule3_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': f'反轉點 |price_2nd| > 90分位',
        'candidates': len(rule3),
        'precision': rule3_precision,
        'coverage': rule3_coverage
    })
    
    # 規則4：符號在反轉點處改變
    rule4 = features[features['sign_changed_at_reversal'] == True]
    rule4_true = len(rule4[rule4['is_true'] == True])
    rule4_precision = rule4_true / len(rule4) * 100 if len(rule4) > 0 else 0
    rule4_coverage = rule4_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '符號在反轉點處改變',
        'candidates': len(rule4),
        'precision': rule4_precision,
        'coverage': rule4_coverage
    })
    
    # 規則5：波峰 price_2nd < 0 OR 波谷 price_2nd > 0
    rule5 = features[
        ((features['is_peak'] == True) & (features['rev_price_2nd'] < 0)) |
        ((features['is_peak'] == False) & (features['rev_price_2nd'] > 0))
    ]
    rule5_true = len(rule5[rule5['is_true'] == True])
    rule5_precision = rule5_true / len(rule5) * 100 if len(rule5) > 0 else 0
    rule5_coverage = rule5_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '波峰負 OR 波谷正',
        'candidates': len(rule5),
        'precision': rule5_precision,
        'coverage': rule5_coverage
    })
    
    # 🆕 規則6：反轉點到當前符號改變（右側交易可用）
    rule6 = features[features['sign_changed'] == True]
    rule6_true = len(rule6[rule6['is_true'] == True])
    rule6_precision = rule6_true / len(rule6) * 100 if len(rule6) > 0 else 0
    rule6_coverage = rule6_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '🆕 反轉點到當前符號改變',
        'candidates': len(rule6),
        'precision': rule6_precision,
        'coverage': rule6_coverage
    })
    
    # 🆕 規則7：price_2nd 變化為正（加速度增加）
    rule7 = features[features['price_2nd_change'] > 0]
    rule7_true = len(rule7[rule7['is_true'] == True])
    rule7_precision = rule7_true / len(rule7) * 100 if len(rule7) > 0 else 0
    rule7_coverage = rule7_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '🆕 price_2nd 變化為正',
        'candidates': len(rule7),
        'precision': rule7_precision,
        'coverage': rule7_coverage
    })
    
    # 🆕 規則8：|price_2nd| 減弱（幅度減小）
    rule8 = features[features['curr_price_2nd_abs'] < features['rev_price_2nd_abs']]
    rule8_true = len(rule8[rule8['is_true'] == True])
    rule8_precision = rule8_true / len(rule8) * 100 if len(rule8) > 0 else 0
    rule8_coverage = rule8_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '🆕 |price_2nd| 減弱',
        'candidates': len(rule8),
        'precision': rule8_precision,
        'coverage': rule8_coverage
    })
    
    # 🆕 規則9：符號改變 OR |price_2nd| 減弱
    rule9 = features[
        (features['sign_changed'] == True) | 
        (features['curr_price_2nd_abs'] < features['rev_price_2nd_abs'])
    ]
    rule9_true = len(rule9[rule9['is_true'] == True])
    rule9_precision = rule9_true / len(rule9) * 100 if len(rule9) > 0 else 0
    rule9_coverage = rule9_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '🆕 符號改變 OR 幅度減弱',
        'candidates': len(rule9),
        'precision': rule9_precision,
        'coverage': rule9_coverage
    })
    
    # 🆕 規則10：符號改變 AND |price_2nd| 減弱
    rule10 = features[
        (features['sign_changed'] == True) & 
        (features['curr_price_2nd_abs'] < features['rev_price_2nd_abs'])
    ]
    rule10_true = len(rule10[rule10['is_true'] == True])
    rule10_precision = rule10_true / len(rule10) * 100 if len(rule10) > 0 else 0
    rule10_coverage = rule10_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '🆕 符號改變 AND 幅度減弱',
        'candidates': len(rule10),
        'precision': rule10_precision,
        'coverage': rule10_coverage
    })
    
    # 🆕 規則11-15：基於反轉幅度的過濾
    thresholds = [0.5, 0.8, 1.0, 1.5, 2.0]
    for threshold in thresholds:
        rule_data = features[features['reversal_magnitude'] >= threshold]
        rule_true = len(rule_data[rule_data['is_true'] == True])
        rule_precision = rule_true / len(rule_data) * 100 if len(rule_data) > 0 else 0
        rule_coverage = rule_true / len(df_true) * 100 if len(df_true) > 0 else 0
        
        rules.append({
            'name': f'🆕 反轉幅度 >= {threshold:.0%}',
            'candidates': len(rule_data),
            'precision': rule_precision,
            'coverage': rule_coverage
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
    
    # 提取 price_2nd 特徵
    features = extract_price_2nd_features(df, candidates, true_reversals)
    print(f"  提取了 {len(features)} 個候選點的特徵")
    
    # 分析 price_2nd 對比
    analyze_price_2nd_comparison(features)
    
    # 測試 price_2nd 過濾規則
    test_price_2nd_filters(features)


if __name__ == '__main__':
    main()
