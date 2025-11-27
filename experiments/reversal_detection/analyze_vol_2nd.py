#!/usr/bin/env python3
"""
成交量二階導數分析

分析真實反轉點 vs 假反轉點的 vol_2nd 特徵：
1. 反轉點的 vol_2nd 值
2. 當前K線的 vol_2nd 值
3. vol_2nd 的變化趨勢
4. vol_2nd 的符號變化
5. vol_2nd 與 price_2nd 的關係
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from improved_stage1 import load_data, find_local_extremes


def calculate_vol_2nd(df, window=3):
    """
    計算成交量二階導數（加速度）
    
    vol_2nd = (vol[i] - vol[i-1]) - (vol[i-1] - vol[i-2])
            = vol[i] - 2*vol[i-1] + vol[i-2]
    """
    volume = df['volume'].values
    
    # 一階導數（速度）
    vol_1st = np.diff(volume, prepend=volume[0])
    
    # 二階導數（加速度）
    vol_2nd = np.diff(vol_1st, prepend=vol_1st[0])
    
    # 平滑處理（使用移動平均）
    vol_2nd_smooth = pd.Series(vol_2nd).rolling(window=window, min_periods=1).mean().values
    
    return vol_2nd, vol_2nd_smooth


def calculate_price_2nd(df, window=3):
    """計算價格二階導數"""
    close = df['close'].values
    price_1st = np.diff(close, prepend=close[0])
    price_2nd = np.diff(price_1st, prepend=price_1st[0])
    price_2nd_smooth = pd.Series(price_2nd).rolling(window=window, min_periods=1).mean().values
    return price_2nd, price_2nd_smooth


def extract_vol_2nd_features(df, candidates, true_reversals):
    """提取 vol_2nd 特徵"""
    print("\n計算 vol_2nd 特徵...")
    
    # 計算 vol_2nd 和 price_2nd
    vol_2nd_raw, vol_2nd_smooth = calculate_vol_2nd(df, window=3)
    price_2nd_raw, price_2nd_smooth = calculate_price_2nd(df, window=3)
    
    # 添加到 DataFrame
    df['vol_2nd_raw'] = vol_2nd_raw
    df['vol_2nd_smooth'] = vol_2nd_smooth
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
        
        # 反轉點的 vol_2nd
        rev_vol_2nd = df['vol_2nd_smooth'].iloc[reversal_idx]
        rev_vol_2nd_raw = df['vol_2nd_raw'].iloc[reversal_idx]
        
        # 當前K線的 vol_2nd
        curr_vol_2nd = df['vol_2nd_smooth'].iloc[cand_idx]
        curr_vol_2nd_raw = df['vol_2nd_raw'].iloc[cand_idx]
        
        # vol_2nd 的變化
        vol_2nd_change = curr_vol_2nd - rev_vol_2nd
        
        # 符號變化
        rev_vol_sign = np.sign(rev_vol_2nd)
        curr_vol_sign = np.sign(curr_vol_2nd)
        vol_sign_changed = (rev_vol_sign != curr_vol_sign)
        
        # price_2nd 的符號變化
        rev_price_2nd = df['price_2nd_smooth'].iloc[reversal_idx]
        curr_price_2nd = df['price_2nd_smooth'].iloc[cand_idx]
        rev_price_sign = np.sign(rev_price_2nd)
        curr_price_sign = np.sign(curr_price_2nd)
        price_sign_changed = (rev_price_sign != curr_price_sign)
        
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
        
        # 歸一化 vol_2nd（相對於平均成交量）
        avg_vol = df['volume'].iloc[max(0, reversal_idx-20):reversal_idx+1].mean()
        if avg_vol > 0:
            rev_vol_2nd_norm = rev_vol_2nd / avg_vol
            curr_vol_2nd_norm = curr_vol_2nd / avg_vol
        else:
            rev_vol_2nd_norm = 0
            curr_vol_2nd_norm = 0
        
        # vol_2nd 與 price_2nd 的關係
        # 同向：兩者符號相同
        vol_price_same_direction = (rev_vol_sign == rev_price_sign)
        
        # 背離：vol_2nd 和 price_2nd 符號變化方向相反
        vol_price_divergence = (vol_sign_changed != price_sign_changed)
        
        features.append({
            'reversal_idx': reversal_idx,
            'is_true': is_true,
            'is_peak': is_peak,
            # 原始 vol_2nd
            'rev_vol_2nd': rev_vol_2nd,
            'curr_vol_2nd': curr_vol_2nd,
            'rev_vol_2nd_raw': rev_vol_2nd_raw,
            'curr_vol_2nd_raw': curr_vol_2nd_raw,
            # 歸一化 vol_2nd
            'rev_vol_2nd_norm': rev_vol_2nd_norm,
            'curr_vol_2nd_norm': curr_vol_2nd_norm,
            # vol_2nd 變化
            'vol_2nd_change': vol_2nd_change,
            # 符號變化
            'rev_vol_sign': rev_vol_sign,
            'curr_vol_sign': curr_vol_sign,
            'vol_sign_changed': vol_sign_changed,
            # 絕對值
            'rev_vol_2nd_abs': abs(rev_vol_2nd),
            'curr_vol_2nd_abs': abs(curr_vol_2nd),
            # price_2nd 相關
            'rev_price_2nd': rev_price_2nd,
            'curr_price_2nd': curr_price_2nd,
            'price_sign_changed': price_sign_changed,
            # vol_2nd 與 price_2nd 的關係
            'vol_price_same_direction': vol_price_same_direction,
            'vol_price_divergence': vol_price_divergence,
        })
    
    return pd.DataFrame(features)


def analyze_vol_2nd_comparison(features):
    """對比真假反轉點的 vol_2nd 特徵"""
    
    df_true = features[features['is_true'] == True]
    df_false = features[features['is_true'] == False]
    
    print(f"\n{'='*80}")
    print(f"📊 vol_2nd 特徵對比：真實 vs 假反轉點")
    print(f"{'='*80}")
    
    print(f"\n樣本數量:")
    print(f"  真陽性: {len(df_true)}")
    print(f"  假陽性: {len(df_false)}")
    print(f"  基準精確度: {len(df_true) / len(features) * 100:.1f}%")
    
    # 1. vol_2nd 值對比
    print(f"\n{'='*80}")
    print(f"1. vol_2nd 值對比")
    print(f"{'='*80}")
    
    print(f"\n{'特徵':<40} {'真陽性':<15} {'假陽性':<15} {'差異':<10}")
    print("-"*80)
    
    true_rev = df_true['rev_vol_2nd_norm'].mean()
    false_rev = df_false['rev_vol_2nd_norm'].mean()
    print(f"{'反轉點 vol_2nd (歸一化)':<40} {true_rev:>14.6f} {false_rev:>14.6f} {true_rev-false_rev:>9.6f}")
    
    true_curr = df_true['curr_vol_2nd_norm'].mean()
    false_curr = df_false['curr_vol_2nd_norm'].mean()
    print(f"{'當前 vol_2nd (歸一化)':<40} {true_curr:>14.6f} {false_curr:>14.6f} {true_curr-false_curr:>9.6f}")
    
    true_rev_abs = df_true['rev_vol_2nd_abs'].mean()
    false_rev_abs = df_false['rev_vol_2nd_abs'].mean()
    print(f"{'反轉點 |vol_2nd|':<40} {true_rev_abs:>14.2f} {false_rev_abs:>14.2f} {true_rev_abs-false_rev_abs:>9.2f}")
    
    true_curr_abs = df_true['curr_vol_2nd_abs'].mean()
    false_curr_abs = df_false['curr_vol_2nd_abs'].mean()
    print(f"{'當前 |vol_2nd|':<40} {true_curr_abs:>14.2f} {false_curr_abs:>14.2f} {true_curr_abs-false_curr_abs:>9.2f}")
    
    # 2. vol_2nd 變化
    print(f"\n{'='*80}")
    print(f"2. vol_2nd 變化")
    print(f"{'='*80}")
    
    print(f"\n{'特徵':<40} {'真陽性':<15} {'假陽性':<15} {'差異':<10}")
    print("-"*80)
    
    true_change = df_true['vol_2nd_change'].mean()
    false_change = df_false['vol_2nd_change'].mean()
    print(f"{'vol_2nd 變化':<40} {true_change:>14.2f} {false_change:>14.2f} {true_change-false_change:>9.2f}")
    
    # 3. 符號變化
    print(f"\n{'='*80}")
    print(f"3. 符號變化")
    print(f"{'='*80}")
    
    print(f"\n{'特徵':<40} {'真陽性':<15} {'假陽性':<15} {'差異':<10}")
    print("-"*80)
    
    true_vol_sign = (df_true['vol_sign_changed'].sum() / len(df_true) * 100) if len(df_true) > 0 else 0
    false_vol_sign = (df_false['vol_sign_changed'].sum() / len(df_false) * 100) if len(df_false) > 0 else 0
    print(f"{'vol_2nd 符號改變比例':<40} {true_vol_sign:>13.1f}% {false_vol_sign:>13.1f}% {true_vol_sign-false_vol_sign:>8.1f}%")
    
    true_price_sign = (df_true['price_sign_changed'].sum() / len(df_true) * 100) if len(df_true) > 0 else 0
    false_price_sign = (df_false['price_sign_changed'].sum() / len(df_false) * 100) if len(df_false) > 0 else 0
    print(f"{'price_2nd 符號改變比例':<40} {true_price_sign:>13.1f}% {false_price_sign:>13.1f}% {true_price_sign-false_price_sign:>8.1f}%")
    
    # 4. vol_2nd 與 price_2nd 的關係
    print(f"\n{'='*80}")
    print(f"4. vol_2nd 與 price_2nd 的關係")
    print(f"{'='*80}")
    
    print(f"\n{'特徵':<40} {'真陽性':<15} {'假陽性':<15} {'差異':<10}")
    print("-"*80)
    
    true_same = (df_true['vol_price_same_direction'].sum() / len(df_true) * 100) if len(df_true) > 0 else 0
    false_same = (df_false['vol_price_same_direction'].sum() / len(df_false) * 100) if len(df_false) > 0 else 0
    print(f"{'vol_2nd 與 price_2nd 同向比例':<40} {true_same:>13.1f}% {false_same:>13.1f}% {true_same-false_same:>8.1f}%")
    
    true_div = (df_true['vol_price_divergence'].sum() / len(df_true) * 100) if len(df_true) > 0 else 0
    false_div = (df_false['vol_price_divergence'].sum() / len(df_false) * 100) if len(df_false) > 0 else 0
    print(f"{'vol_2nd 與 price_2nd 背離比例':<40} {true_div:>13.1f}% {false_div:>13.1f}% {true_div-false_div:>8.1f}%")
    
    # 5. 組合情況分析
    print(f"\n{'='*80}")
    print(f"5. vol_2nd 與 price_2nd 組合情況")
    print(f"{'='*80}")
    
    print(f"\n{'情況':<50} {'真陽性':<15} {'假陽性':<15} {'真陽性比例':<15}")
    print("-"*95)
    
    # 1. 兩者都符號改變
    combo1_true = len(df_true[(df_true['vol_sign_changed'] == True) & 
                               (df_true['price_sign_changed'] == True)])
    combo1_false = len(df_false[(df_false['vol_sign_changed'] == True) & 
                                 (df_false['price_sign_changed'] == True)])
    combo1_total = combo1_true + combo1_false
    combo1_pct = combo1_true / combo1_total * 100 if combo1_total > 0 else 0
    print(f"{'vol_2nd 符號改變 AND price_2nd 符號改變':<50} {combo1_true:<15} {combo1_false:<15} {combo1_pct:<14.1f}%")
    
    # 2. 只有 price_2nd 符號改變
    combo2_true = len(df_true[(df_true['vol_sign_changed'] == False) & 
                               (df_true['price_sign_changed'] == True)])
    combo2_false = len(df_false[(df_false['vol_sign_changed'] == False) & 
                                 (df_false['price_sign_changed'] == True)])
    combo2_total = combo2_true + combo2_false
    combo2_pct = combo2_true / combo2_total * 100 if combo2_total > 0 else 0
    print(f"{'只有 price_2nd 符號改變':<50} {combo2_true:<15} {combo2_false:<15} {combo2_pct:<14.1f}%")
    
    # 3. 只有 vol_2nd 符號改變
    combo3_true = len(df_true[(df_true['vol_sign_changed'] == True) & 
                               (df_true['price_sign_changed'] == False)])
    combo3_false = len(df_false[(df_false['vol_sign_changed'] == True) & 
                                 (df_false['price_sign_changed'] == False)])
    combo3_total = combo3_true + combo3_false
    combo3_pct = combo3_true / combo3_total * 100 if combo3_total > 0 else 0
    print(f"{'只有 vol_2nd 符號改變':<50} {combo3_true:<15} {combo3_false:<15} {combo3_pct:<14.1f}%")
    
    # 4. 兩者都不改變
    combo4_true = len(df_true[(df_true['vol_sign_changed'] == False) & 
                               (df_true['price_sign_changed'] == False)])
    combo4_false = len(df_false[(df_false['vol_sign_changed'] == False) & 
                                 (df_false['price_sign_changed'] == False)])
    combo4_total = combo4_true + combo4_false
    combo4_pct = combo4_true / combo4_total * 100 if combo4_total > 0 else 0
    print(f"{'兩者都不改變':<50} {combo4_true:<15} {combo4_false:<15} {combo4_pct:<14.1f}%")


def test_vol_2nd_filters(features):
    """測試基於 vol_2nd 的過濾規則"""
    
    df_true = features[features['is_true'] == True]
    
    print(f"\n{'='*80}")
    print(f"💡 基於 vol_2nd 的過濾規則")
    print(f"{'='*80}")
    
    rules = []
    
    # 規則1：vol_2nd 符號改變
    rule1 = features[features['vol_sign_changed'] == True]
    rule1_true = len(rule1[rule1['is_true'] == True])
    rule1_precision = rule1_true / len(rule1) * 100 if len(rule1) > 0 else 0
    rule1_coverage = rule1_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': 'vol_2nd 符號改變',
        'candidates': len(rule1),
        'precision': rule1_precision,
        'coverage': rule1_coverage
    })
    
    # 規則2：price_2nd 符號改變
    rule2 = features[features['price_sign_changed'] == True]
    rule2_true = len(rule2[rule2['is_true'] == True])
    rule2_precision = rule2_true / len(rule2) * 100 if len(rule2) > 0 else 0
    rule2_coverage = rule2_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': 'price_2nd 符號改變',
        'candidates': len(rule2),
        'precision': rule2_precision,
        'coverage': rule2_coverage
    })
    
    # 規則3：vol_2nd AND price_2nd 符號改變
    rule3 = features[(features['vol_sign_changed'] == True) & 
                     (features['price_sign_changed'] == True)]
    rule3_true = len(rule3[rule3['is_true'] == True])
    rule3_precision = rule3_true / len(rule3) * 100 if len(rule3) > 0 else 0
    rule3_coverage = rule3_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': 'vol_2nd AND price_2nd 符號改變',
        'candidates': len(rule3),
        'precision': rule3_precision,
        'coverage': rule3_coverage
    })
    
    # 規則4：vol_2nd OR price_2nd 符號改變
    rule4 = features[(features['vol_sign_changed'] == True) | 
                     (features['price_sign_changed'] == True)]
    rule4_true = len(rule4[rule4['is_true'] == True])
    rule4_precision = rule4_true / len(rule4) * 100 if len(rule4) > 0 else 0
    rule4_coverage = rule4_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': 'vol_2nd OR price_2nd 符號改變',
        'candidates': len(rule4),
        'precision': rule4_precision,
        'coverage': rule4_coverage
    })
    
    # 規則5：vol_2nd 與 price_2nd 背離
    rule5 = features[features['vol_price_divergence'] == True]
    rule5_true = len(rule5[rule5['is_true'] == True])
    rule5_precision = rule5_true / len(rule5) * 100 if len(rule5) > 0 else 0
    rule5_coverage = rule5_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': 'vol_2nd 與 price_2nd 背離',
        'candidates': len(rule5),
        'precision': rule5_precision,
        'coverage': rule5_coverage
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
    
    # 提取 vol_2nd 特徵
    features = extract_vol_2nd_features(df, candidates, true_reversals)
    print(f"  提取了 {len(features)} 個候選點的特徵")
    
    # 分析 vol_2nd 對比
    analyze_vol_2nd_comparison(features)
    
    # 測試 vol_2nd 過濾規則
    test_vol_2nd_filters(features)


if __name__ == '__main__':
    main()
