#!/usr/bin/env python3
"""
階段2：基於特徵差異的規則過濾器

基於 analyze_true_vs_false_reversals.py 的發現：
1. ATR 加速：真陽性 4.84 vs 假陽性 2.38
2. 成交量比率：真陽性 1.92 vs 假陽性 1.29
3. 成交量激增：真陽性 47.5% vs 假陽性 24.2%
4. 時間間隔：真陽性 37.9 vs 假陽性 21.4
5. 突出度：真陽性 0.61% vs 假陽性 0.16%
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from improved_stage1 import load_data, find_local_extremes
from analyze_true_vs_false_reversals import calculate_features


def filter_by_atr_acceleration(df_features, threshold=3.0):
    """
    過濾器1：ATR 加速
    
    真陽性均值：4.84
    假陽性均值：2.38
    建議閾值：3.0（介於兩者之間）
    """
    if 'atr_2nd' not in df_features.columns:
        return df_features
    
    return df_features[df_features['atr_2nd'] > threshold].copy()


def filter_by_volume_ratio(df_features, threshold=1.5):
    """
    過濾器2：成交量比率
    
    真陽性均值：1.92
    假陽性均值：1.29
    建議閾值：1.5（介於兩者之間）
    """
    if 'volume_ratio' not in df_features.columns:
        return df_features
    
    return df_features[df_features['volume_ratio'] > threshold].copy()


def filter_by_time_interval(df_features, min_interval=25):
    """
    過濾器3：時間間隔
    
    真陽性均值：37.9
    假陽性均值：21.4
    建議閾值：25（介於兩者之間）
    
    過濾掉間隔太短的反轉點（可能是噪音）
    """
    if 'time_since_last_extreme' not in df_features.columns:
        return df_features
    
    return df_features[df_features['time_since_last_extreme'] > min_interval].copy()


def filter_by_prominence(df_features, min_prominence=0.3):
    """
    過濾器4：突出度
    
    真陽性均值：0.61%
    假陽性均值：0.16%
    建議閾值：0.3%（介於兩者之間）
    
    過濾掉不夠突出的反轉點
    """
    if 'prominence' not in df_features.columns:
        return df_features
    
    return df_features[df_features['prominence'] > min_prominence].copy()


def filter_by_rsi_extreme(df_features):
    """
    過濾器5：RSI 極值
    
    真陽性：15-16% 在極值
    假陽性：4-5% 在極值
    
    保留 RSI 在極值的候選點
    """
    if 'rsi_overbought' not in df_features.columns or 'rsi_oversold' not in df_features.columns:
        return df_features
    
    # 保留 RSI 超買或超賣的候選點
    return df_features[
        df_features['rsi_overbought'] | df_features['rsi_oversold']
    ].copy()


def filter_combined(df_features, 
                   atr_threshold=3.0,
                   volume_threshold=1.5,
                   time_threshold=25,
                   prominence_threshold=0.3,
                   use_rsi=False):
    """
    組合過濾器
    
    策略：使用多個過濾器的組合
    """
    result = df_features.copy()
    
    initial_count = len(result)
    print(f"\n初始候選點: {initial_count}")
    
    # 過濾器1：ATR 加速
    if 'atr_2nd' in result.columns:
        result = result[result['atr_2nd'] > atr_threshold]
        print(f"ATR 加速過濾後: {len(result)} (過濾掉 {initial_count - len(result)})")
    
    # 過濾器2：成交量比率
    if 'volume_ratio' in result.columns:
        before = len(result)
        result = result[result['volume_ratio'] > volume_threshold]
        print(f"成交量過濾後: {len(result)} (過濾掉 {before - len(result)})")
    
    # 過濾器3：時間間隔
    if 'time_since_last_extreme' in result.columns:
        before = len(result)
        result = result[result['time_since_last_extreme'] > time_threshold]
        print(f"時間間隔過濾後: {len(result)} (過濾掉 {before - len(result)})")
    
    # 過濾器4：突出度
    if 'prominence' in result.columns:
        before = len(result)
        result = result[result['prominence'] > prominence_threshold]
        print(f"突出度過濾後: {len(result)} (過濾掉 {before - len(result)})")
    
    # 過濾器5：RSI 極值（可選）
    if use_rsi and 'rsi_overbought' in result.columns:
        before = len(result)
        result = result[result['rsi_overbought'] | result['rsi_oversold']]
        print(f"RSI 極值過濾後: {len(result)} (過濾掉 {before - len(result)})")
    
    print(f"\n最終候選點: {len(result)}")
    print(f"總共過濾掉: {initial_count - len(result)} ({(initial_count - len(result))/initial_count*100:.1f}%)")
    
    return result


def evaluate_filters():
    """評估不同過濾器的效果"""
    print("="*80)
    print("🔬 評估階段2過濾器")
    print("="*80)
    
    # 載入特徵數據
    print("\n載入特徵數據...")
    df_true = pd.read_csv('experiments/reversal_detection/true_positive_features.csv')
    df_false = pd.read_csv('experiments/reversal_detection/false_positive_features.csv')
    
    df_all = pd.concat([df_true, df_false], ignore_index=True)
    
    print(f"  真陽性: {len(df_true)}")
    print(f"  假陽性: {len(df_false)}")
    print(f"  總計: {len(df_all)}")
    
    # 測試不同的過濾器組合
    test_configs = [
        {
            'name': '基準（無過濾）',
            'atr_threshold': 0,
            'volume_threshold': 0,
            'time_threshold': 0,
            'prominence_threshold': 0,
            'use_rsi': False
        },
        {
            'name': '單一：ATR 加速',
            'atr_threshold': 3.0,
            'volume_threshold': 0,
            'time_threshold': 0,
            'prominence_threshold': 0,
            'use_rsi': False
        },
        {
            'name': '單一：成交量',
            'atr_threshold': 0,
            'volume_threshold': 1.5,
            'time_threshold': 0,
            'prominence_threshold': 0,
            'use_rsi': False
        },
        {
            'name': '組合：ATR + 成交量',
            'atr_threshold': 3.0,
            'volume_threshold': 1.5,
            'time_threshold': 0,
            'prominence_threshold': 0,
            'use_rsi': False
        },
        {
            'name': '組合：ATR + 成交量 + 時間',
            'atr_threshold': 3.0,
            'volume_threshold': 1.5,
            'time_threshold': 25,
            'prominence_threshold': 0,
            'use_rsi': False
        },
        {
            'name': '組合：全部特徵',
            'atr_threshold': 3.0,
            'volume_threshold': 1.5,
            'time_threshold': 25,
            'prominence_threshold': 0.3,
            'use_rsi': False
        },
        {
            'name': '組合：全部 + RSI',
            'atr_threshold': 3.0,
            'volume_threshold': 1.5,
            'time_threshold': 25,
            'prominence_threshold': 0.3,
            'use_rsi': True
        }
    ]
    
    print("\n" + "="*80)
    print("📊 過濾器效果對比")
    print("="*80)
    
    print(f"\n{'配置':<30} {'候選點':>10} {'真陽性':>10} {'假陽性':>10} {'精確度':>10} {'覆蓋率':>10}")
    print("-"*90)
    
    results = []
    
    for config in test_configs:
        # 應用過濾器
        filtered = filter_combined(
            df_all,
            atr_threshold=config['atr_threshold'],
            volume_threshold=config['volume_threshold'],
            time_threshold=config['time_threshold'],
            prominence_threshold=config['prominence_threshold'],
            use_rsi=config['use_rsi']
        )
        
        # 計算指標
        true_positives = len(filtered[filtered['label'] == 'true_positive'])
        false_positives = len(filtered[filtered['label'] == 'false_positive'])
        total_candidates = len(filtered)
        
        precision = true_positives / total_candidates * 100 if total_candidates > 0 else 0
        coverage = true_positives / len(df_true) * 100
        
        results.append({
            'name': config['name'],
            'candidates': total_candidates,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'precision': precision,
            'coverage': coverage
        })
        
        print(f"{config['name']:<30} {total_candidates:>10} {true_positives:>10} {false_positives:>10} "
              f"{precision:>9.1f}% {coverage:>9.1f}%")
    
    # 推薦
    print("\n" + "="*80)
    print("🎯 推薦")
    print("="*80)
    
    # 找出最佳配置
    best_precision = max(results, key=lambda x: x['precision'])
    best_balance = max([r for r in results if r['coverage'] >= 90], 
                       key=lambda x: x['precision'],
                       default=None)
    
    print(f"\n最高精確度: {best_precision['name']}")
    print(f"  精確度: {best_precision['precision']:.1f}%")
    print(f"  覆蓋率: {best_precision['coverage']:.1f}%")
    print(f"  候選點: {best_precision['candidates']}")
    
    if best_balance:
        print(f"\n最佳平衡 (覆蓋率 >= 90%): {best_balance['name']}")
        print(f"  精確度: {best_balance['precision']:.1f}%")
        print(f"  覆蓋率: {best_balance['coverage']:.1f}%")
        print(f"  候選點: {best_balance['candidates']}")
        print(f"  過濾掉: {len(df_all) - best_balance['candidates']} 個假陽性")


def main():
    """主函數"""
    evaluate_filters()


if __name__ == '__main__':
    main()
