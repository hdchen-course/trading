#!/usr/bin/env python3
"""
測試組合策略：符號改變 + 趨勢減弱

策略1：符號改變（已經反轉）
策略2：趨勢減弱（即將反轉）
  - 同方向但幅度減小
  - abs((current - reverse) / reverse) < threshold
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from improved_stage1 import load_data, find_local_extremes
from analyze_price_2nd import calculate_price_2nd, extract_price_2nd_features


def test_combined_strategies():
    """測試組合策略"""
    
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
    print(f"🧪 測試組合策略：符號改變 + 趨勢減弱")
    print(f"{'='*80}")
    
    # 基準統計
    total_true = features['is_true'].sum()
    baseline_precision = total_true / len(features) * 100
    
    print(f"\n基準:")
    print(f"  總候選點: {len(features)}")
    print(f"  真陽性: {total_true}")
    print(f"  精確度: {baseline_precision:.1f}%")
    
    # 策略1：符號改變
    sign_changed = features['sign_changed'] == True
    
    # 策略2：趨勢減弱（同方向但幅度減小）
    # 計算相對變化率
    features['relative_change'] = np.where(
        features['rev_price_2nd'].abs() > 0.0001,
        (features['curr_price_2nd'] - features['rev_price_2nd']).abs() / features['rev_price_2nd'].abs(),
        0
    )
    
    # 同方向：符號相同
    same_direction = features['sign_changed'] == False
    
    # 測試不同的減弱閾值
    print(f"\n{'='*80}")
    print(f"📊 測試不同的趨勢減弱閾值")
    print(f"{'='*80}")
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = []
    
    for threshold in thresholds:
        # 趨勢減弱：同方向 + 相對變化 < threshold
        trend_weakening = same_direction & (features['relative_change'] < threshold)
        
        # 組合策略：符號改變 OR 趨勢減弱
        combined = sign_changed | trend_weakening
        
        # 統計
        combined_data = features[combined]
        combined_true = combined_data[combined_data['is_true'] == True]
        
        precision = len(combined_true) / len(combined_data) * 100 if len(combined_data) > 0 else 0
        coverage = len(combined_true) / total_true * 100
        
        # 分別統計兩個策略的貢獻
        sign_only = features[sign_changed & ~trend_weakening]
        sign_only_true = sign_only[sign_only['is_true'] == True]
        
        trend_only = features[~sign_changed & trend_weakening]
        trend_only_true = trend_only[trend_only['is_true'] == True]
        
        both = features[sign_changed & trend_weakening]
        both_true = both[both['is_true'] == True]
        
        results.append({
            'threshold': threshold,
            'total': len(combined_data),
            'precision': precision,
            'coverage': coverage,
            'sign_only': len(sign_only),
            'sign_only_true': len(sign_only_true),
            'trend_only': len(trend_only),
            'trend_only_true': len(trend_only_true),
            'both': len(both),
            'both_true': len(both_true),
        })
    
    # 顯示結果
    print(f"\n{'閾值':>8} {'總候選':>10} {'精確度':>10} {'覆蓋率':>10} {'符號獨有':>12} {'趨勢獨有':>12} {'兩者重疊':>12}")
    print("-"*90)
    
    for r in results:
        print(f"{r['threshold']:>8.1f} {r['total']:>10} {r['precision']:>9.1f}% {r['coverage']:>9.1f}% "
              f"{r['sign_only']:>5}/{r['sign_only_true']:>5} "
              f"{r['trend_only']:>5}/{r['trend_only_true']:>5} "
              f"{r['both']:>5}/{r['both_true']:>5}")
    
    # 找出最佳閾值
    print(f"\n{'='*80}")
    print(f"🎯 最佳閾值分析")
    print(f"{'='*80}")
    
    # 按精確度排序
    results_df = pd.DataFrame(results)
    best_precision = results_df.loc[results_df['precision'].idxmax()]
    
    print(f"\n最高精確度:")
    print(f"  閾值: {best_precision['threshold']:.1f}")
    print(f"  精確度: {best_precision['precision']:.1f}%")
    print(f"  覆蓋率: {best_precision['coverage']:.1f}%")
    print(f"  候選點: {best_precision['total']:.0f}")
    
    # 按覆蓋率排序（精確度 > 60%）
    high_precision = results_df[results_df['precision'] > 60]
    if len(high_precision) > 0:
        best_coverage = high_precision.loc[high_precision['coverage'].idxmax()]
        
        print(f"\n最高覆蓋率（精確度 > 60%）:")
        print(f"  閾值: {best_coverage['threshold']:.1f}")
        print(f"  精確度: {best_coverage['precision']:.1f}%")
        print(f"  覆蓋率: {best_coverage['coverage']:.1f}%")
        print(f"  候選點: {best_coverage['total']:.0f}")
    
    # 平衡點（精確度和覆蓋率的調和平均）
    results_df['f1'] = 2 * (results_df['precision'] * results_df['coverage']) / (results_df['precision'] + results_df['coverage'])
    best_f1 = results_df.loc[results_df['f1'].idxmax()]
    
    print(f"\n最佳平衡點（F1分數）:")
    print(f"  閾值: {best_f1['threshold']:.1f}")
    print(f"  精確度: {best_f1['precision']:.1f}%")
    print(f"  覆蓋率: {best_f1['coverage']:.1f}%")
    print(f"  F1分數: {best_f1['f1']:.1f}")
    print(f"  候選點: {best_f1['total']:.0f}")
    
    # 詳細分析最佳閾值
    print(f"\n{'='*80}")
    print(f"🔍 最佳閾值詳細分析（閾值 = {best_f1['threshold']:.1f}）")
    print(f"{'='*80}")
    
    threshold = best_f1['threshold']
    trend_weakening = same_direction & (features['relative_change'] < threshold)
    combined = sign_changed | trend_weakening
    
    # 分類統計
    sign_only = features[sign_changed & ~trend_weakening]
    trend_only = features[~sign_changed & trend_weakening]
    both = features[sign_changed & trend_weakening]
    neither = features[~sign_changed & ~trend_weakening]
    
    print(f"\n分類統計:")
    print(f"{'類別':<30} {'候選點':>10} {'真陽性':>10} {'精確度':>10}")
    print("-"*70)
    
    for name, data in [
        ('只有符號改變', sign_only),
        ('只有趨勢減弱', trend_only),
        ('兩者都滿足', both),
        ('兩者都不滿足', neither),
    ]:
        data_true = data[data['is_true'] == True]
        precision = len(data_true) / len(data) * 100 if len(data) > 0 else 0
        print(f"{name:<30} {len(data):>10} {len(data_true):>10} {precision:>9.1f}%")
    
    # 趨勢減弱的詳細分析
    print(f"\n{'='*80}")
    print(f"📊 趨勢減弱的詳細分析")
    print(f"{'='*80}")
    
    if len(trend_only) > 0:
        trend_only_true = trend_only[trend_only['is_true'] == True]
        trend_only_false = trend_only[trend_only['is_true'] == False]
        
        print(f"\n趨勢減弱（不包括符號改變）:")
        print(f"  候選點: {len(trend_only)}")
        print(f"  真陽性: {len(trend_only_true)}")
        print(f"  假陽性: {len(trend_only_false)}")
        print(f"  精確度: {len(trend_only_true)/len(trend_only)*100:.1f}%")
        
        # 相對變化分布
        print(f"\n  相對變化分布:")
        print(f"    最小值: {trend_only['relative_change'].min():.3f}")
        print(f"    25分位: {trend_only['relative_change'].quantile(0.25):.3f}")
        print(f"    中位數: {trend_only['relative_change'].median():.3f}")
        print(f"    75分位: {trend_only['relative_change'].quantile(0.75):.3f}")
        print(f"    最大值: {trend_only['relative_change'].max():.3f}")
        
        # 顯示例子
        print(f"\n  真陽性例子:")
        print(f"  {'反轉點':>15} {'當前':>15} {'相對變化':>12} {'符號':>10}")
        print("-"*60)
        for _, row in trend_only_true.head(5).iterrows():
            rev_2nd = row['rev_price_2nd']
            curr_2nd = row['curr_price_2nd']
            rel_change = row['relative_change']
            sign = '正→正' if rev_2nd > 0 else '負→負'
            print(f"  {rev_2nd:>15.6f} {curr_2nd:>15.6f} {rel_change:>11.3f} {sign:>10}")
        
        print(f"\n  假陽性例子:")
        print(f"  {'反轉點':>15} {'當前':>15} {'相對變化':>12} {'符號':>10}")
        print("-"*60)
        for _, row in trend_only_false.head(5).iterrows():
            rev_2nd = row['rev_price_2nd']
            curr_2nd = row['curr_price_2nd']
            rel_change = row['relative_change']
            sign = '正→正' if rev_2nd > 0 else '負→負'
            print(f"  {rev_2nd:>15.6f} {curr_2nd:>15.6f} {rel_change:>11.3f} {sign:>10}")
    
    # 對比不同策略
    print(f"\n{'='*80}")
    print(f"📊 策略對比")
    print(f"{'='*80}")
    
    strategies = [
        ('基準（無過濾）', features),
        ('只用符號改變', features[sign_changed]),
        (f'只用趨勢減弱（< {threshold:.1f}）', features[trend_weakening]),
        (f'組合策略（閾值 {threshold:.1f}）', features[combined]),
    ]
    
    print(f"\n{'策略':<35} {'候選點':>10} {'精確度':>10} {'覆蓋率':>10} {'F1分數':>10}")
    print("-"*85)
    
    for name, data in strategies:
        if len(data) > 0:
            data_true = data[data['is_true'] == True]
            precision = len(data_true) / len(data) * 100
            coverage = len(data_true) / total_true * 100
            f1 = 2 * (precision * coverage) / (precision + coverage) if (precision + coverage) > 0 else 0
            print(f"{name:<35} {len(data):>10} {precision:>9.1f}% {coverage:>9.1f}% {f1:>9.1f}")
    
    # 結論
    print(f"\n{'='*80}")
    print(f"💡 結論")
    print(f"{'='*80}")
    
    sign_data = features[sign_changed]
    sign_true = sign_data[sign_data['is_true'] == True]
    sign_precision = len(sign_true) / len(sign_data) * 100
    sign_coverage = len(sign_true) / total_true * 100
    
    combined_data = features[combined]
    combined_true = combined_data[combined_data['is_true'] == True]
    combined_precision = len(combined_true) / len(combined_data) * 100
    combined_coverage = len(combined_true) / total_true * 100
    
    print(f"\n1. 組合策略 vs 只用符號改變:")
    print(f"   精確度變化: {combined_precision - sign_precision:+.1f}%")
    print(f"   覆蓋率變化: {combined_coverage - sign_coverage:+.1f}%")
    
    if combined_precision >= sign_precision and combined_coverage > sign_coverage:
        print(f"   ✅ 組合策略更好！提升了覆蓋率且保持精確度")
    elif combined_precision > sign_precision:
        print(f"   ✅ 組合策略提升了精確度")
    elif combined_coverage > sign_coverage:
        print(f"   ⚠️ 組合策略提升了覆蓋率，但降低了精確度")
    else:
        print(f"   ❌ 組合策略沒有改善")
    
    print(f"\n2. 趨勢減弱的價值:")
    if len(trend_only) > 0:
        trend_only_precision = len(trend_only_true) / len(trend_only) * 100
        if trend_only_precision > baseline_precision:
            print(f"   ✅ 趨勢減弱有效！精確度 {trend_only_precision:.1f}% > 基準 {baseline_precision:.1f}%")
        else:
            print(f"   ❌ 趨勢減弱效果不佳，精確度 {trend_only_precision:.1f}% < 基準 {baseline_precision:.1f}%")
    
    print(f"\n3. 最佳策略建議:")
    if combined_precision >= 65 and combined_coverage >= 75:
        print(f"   ✅ 推薦使用組合策略（閾值 {threshold:.1f}）")
        print(f"   精確度 {combined_precision:.1f}%，覆蓋率 {combined_coverage:.1f}%")
    elif sign_precision >= 70:
        print(f"   ✅ 推薦只使用符號改變")
        print(f"   精確度 {sign_precision:.1f}%，覆蓋率 {sign_coverage:.1f}%")
    else:
        print(f"   ⚠️ 需要進一步優化")


if __name__ == '__main__':
    test_combined_strategies()
