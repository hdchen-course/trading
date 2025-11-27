#!/usr/bin/env python3
"""
布林通道位置組合分析

新增分析：
1. 過濾中軌寬範圍 (0.0-0.55)
2. 反轉點 + 當前位置的組合分析
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from improved_stage1 import load_data, find_local_extremes


def calculate_bollinger_bands(df, period=48, std_dev=2.4):
    """計算布林通道"""
    close = df['close']
    middle = close.rolling(window=period, min_periods=1).mean()
    std = close.rolling(window=period, min_periods=1).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    return middle, upper, lower


def calculate_bb_position(price, middle, upper, lower):
    """計算價格在布林通道中的位置 (0=下軌, 0.5=中軌, 1=上軌)"""
    band_width = upper - lower
    if band_width == 0 or np.isnan(band_width):
        return 0.5
    position = (price - lower) / band_width
    return position


def extract_features(df, candidates, true_reversals):
    """提取布林通道特徵"""
    print("\n計算布林通道 (period=48, std=2.4)...")
    middle, upper, lower = calculate_bollinger_bands(df, period=48, std_dev=2.4)
    
    print("提取特徵...")
    features = []
    
    for cand_idx in candidates:
        # 找出對應的反轉點
        reversal_idx = None
        for offset in [1, 2]:
            check_idx = cand_idx - offset
            if check_idx >= 0:
                reversal_idx = check_idx
                break
        
        if reversal_idx is None or reversal_idx < 48:
            continue
        
        # 反轉點的布林通道位置
        rev_price = df['close'].iloc[reversal_idx]
        rev_bb_pos = calculate_bb_position(
            rev_price,
            middle.iloc[reversal_idx],
            upper.iloc[reversal_idx],
            lower.iloc[reversal_idx]
        )
        
        # 當前K線的布林通道位置
        curr_price = df['close'].iloc[cand_idx]
        curr_bb_pos = calculate_bb_position(
            curr_price,
            middle.iloc[cand_idx],
            upper.iloc[cand_idx],
            lower.iloc[cand_idx]
        )
        
        # 判斷是否為真陽性
        is_true = reversal_idx in true_reversals
        
        # 判斷反轉點類型
        lookback = 20
        if reversal_idx >= lookback:
            window_prices = df['close'].iloc[reversal_idx-lookback:reversal_idx+1].values
            rev_price_val = df['close'].iloc[reversal_idx]
            is_peak = rev_price_val == np.max(window_prices)
        else:
            is_peak = None
        
        features.append({
            'reversal_idx': reversal_idx,
            'is_true': is_true,
            'is_peak': is_peak,
            'reversal_bb_position': rev_bb_pos,
            'current_bb_position': curr_bb_pos,
        })
    
    return pd.DataFrame(features)


def analyze_combined_rules(features):
    """分析組合規則"""
    df_true = features[features['is_true'] == True]
    df_false = features[features['is_true'] == False]
    
    print(f"\n{'='*80}")
    print(f"🆕 布林通道組合分析")
    print(f"{'='*80}")
    
    print(f"\n樣本數量:")
    print(f"  真陽性: {len(df_true)}")
    print(f"  假陽性: {len(df_false)}")
    print(f"  基準精確度: {len(df_true) / len(features) * 100:.1f}%")
    
    rules = []
    
    # 原有規則：過濾中軌附近 (0.45-0.55)
    rule1 = features[
        ~((features['reversal_bb_position'] >= 0.45) & (features['reversal_bb_position'] <= 0.55))
    ]
    rule1_true = len(rule1[rule1['is_true'] == True])
    rule1_precision = rule1_true / len(rule1) * 100 if len(rule1) > 0 else 0
    rule1_coverage = rule1_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '過濾中軌附近 (0.45-0.55)',
        'candidates': len(rule1),
        'precision': rule1_precision,
        'coverage': rule1_coverage
    })
    
    # 🆕 規則：過濾中軌寬範圍 (0.0-0.55)
    rule2 = features[
        ~((features['reversal_bb_position'] >= 0.0) & (features['reversal_bb_position'] <= 0.55))
    ]
    rule2_true = len(rule2[rule2['is_true'] == True])
    rule2_precision = rule2_true / len(rule2) * 100 if len(rule2) > 0 else 0
    rule2_coverage = rule2_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '🆕 過濾中軌寬範圍 (0.0-0.55)',
        'candidates': len(rule2),
        'precision': rule2_precision,
        'coverage': rule2_coverage
    })
    
    # 反轉點在極端位置
    rule3 = features[
        (features['reversal_bb_position'] > 0.8) | (features['reversal_bb_position'] < 0.2)
    ]
    rule3_true = len(rule3[rule3['is_true'] == True])
    rule3_precision = rule3_true / len(rule3) * 100 if len(rule3) > 0 else 0
    rule3_coverage = rule3_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '反轉點在極端位置 (>0.8 或 <0.2)',
        'candidates': len(rule3),
        'precision': rule3_precision,
        'coverage': rule3_coverage
    })
    
    # 反轉點極端 + 當前K線不在中軌
    rule4 = features[
        ((features['reversal_bb_position'] > 0.8) | (features['reversal_bb_position'] < 0.2)) &
        ~((features['current_bb_position'] >= 0.45) & (features['current_bb_position'] <= 0.55))
    ]
    rule4_true = len(rule4[rule4['is_true'] == True])
    rule4_precision = rule4_true / len(rule4) * 100 if len(rule4) > 0 else 0
    rule4_coverage = rule4_true / len(df_true) * 100 if len(df_true) > 0 else 0
    
    rules.append({
        'name': '反轉點極端 + 當前K線不在中軌',
        'candidates': len(rule4),
        'precision': rule4_precision,
        'coverage': rule4_coverage
    })
    
    print(f"\n{'='*80}")
    print(f"📊 組合規則對比")
    print(f"{'='*80}")
    
    print(f"\n{'規則':<50} {'候選點':>10} {'精確度':>10} {'覆蓋率':>10}")
    print("-"*90)
    
    for rule in rules:
        print(f"{rule['name']:<50} {rule['candidates']:>10} {rule['precision']:>9.1f}% {rule['coverage']:>9.1f}%")
    
    # 🆕 詳細的組合分析
    print(f"\n{'='*80}")
    print(f"🆕 反轉點 + 當前位置詳細組合分析")
    print(f"{'='*80}")
    
    # 定義位置區間
    def get_position_zone(pos):
        if pos < 0.2:
            return '下軌區'
        elif pos > 0.8:
            return '上軌區'
        else:
            return '中軌區'
    
    # 創建組合特徵
    features['rev_zone'] = features['reversal_bb_position'].apply(get_position_zone)
    features['curr_zone'] = features['current_bb_position'].apply(get_position_zone)
    features['combo'] = features['rev_zone'] + ' + ' + features['curr_zone']
    
    # 統計每種組合
    combo_stats = []
    for combo in features['combo'].unique():
        combo_data = features[features['combo'] == combo]
        combo_true = len(combo_data[combo_data['is_true'] == True])
        combo_precision = combo_true / len(combo_data) * 100 if len(combo_data) > 0 else 0
        combo_coverage = combo_true / len(df_true) * 100 if len(df_true) > 0 else 0
        
        combo_stats.append({
            'combo': combo,
            'candidates': len(combo_data),
            'precision': combo_precision,
            'coverage': combo_coverage
        })
    
    # 按精確度排序
    combo_stats = sorted(combo_stats, key=lambda x: x['precision'], reverse=True)
    
    print(f"\n{'組合情況':<40} {'候選點':>10} {'精確度':>10} {'覆蓋率':>10}")
    print("-"*80)
    
    for stat in combo_stats:
        print(f"{stat['combo']:<40} {stat['candidates']:>10} {stat['precision']:>9.1f}% {stat['coverage']:>9.1f}%")
    
    # 關鍵洞察
    print(f"\n{'='*80}")
    print(f"💡 關鍵洞察")
    print(f"{'='*80}")
    
    # 找出最佳組合
    best_combo = max(combo_stats, key=lambda x: x['precision'] if x['candidates'] > 100 else 0)
    print(f"\n最佳組合 (候選點>100):")
    print(f"  {best_combo['combo']}")
    print(f"  候選點: {best_combo['candidates']}")
    print(f"  精確度: {best_combo['precision']:.1f}%")
    print(f"  覆蓋率: {best_combo['coverage']:.1f}%")
    
    # 找出最差組合
    worst_combo = min(combo_stats, key=lambda x: x['precision'] if x['candidates'] > 100 else 100)
    print(f"\n最差組合 (候選點>100):")
    print(f"  {worst_combo['combo']}")
    print(f"  候選點: {worst_combo['candidates']}")
    print(f"  精確度: {worst_combo['precision']:.1f}%")
    print(f"  覆蓋率: {worst_combo['coverage']:.1f}%")


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
    
    # 提取特徵
    features = extract_features(df, candidates, true_reversals)
    print(f"  提取了 {len(features)} 個候選點的特徵")
    
    # 分析組合規則
    analyze_combined_rules(features)


if __name__ == '__main__':
    main()
