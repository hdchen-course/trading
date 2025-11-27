#!/usr/bin/env python3
"""
分析反轉點的價格相對位置

檢查反轉點是否發生在：
1. 距離 24/48 小時高點/低點的相對位置
2. 價格回撤/反彈的幅度
3. 是否在極端位置（接近高點或低點）
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np


def calculate_price_position_features(df):
    """計算價格相對位置特徵"""
    print("=" * 80)
    print("📊 計算價格相對位置特徵")
    print("=" * 80)
    
    # 15分鐘一根K線，24小時 = 96根，48小時 = 192根
    window_24h = 96
    window_48h = 192
    
    print(f"\n使用窗口大小:")
    print(f"  24小時: {window_24h} 根K線")
    print(f"  48小時: {window_48h} 根K線")
    
    # 計算滾動高點和低點
    df['high_24h'] = df['high'].rolling(window=window_24h, min_periods=1).max()
    df['low_24h'] = df['low'].rolling(window=window_24h, min_periods=1).min()
    df['high_48h'] = df['high'].rolling(window=window_48h, min_periods=1).max()
    df['low_48h'] = df['low'].rolling(window=window_48h, min_periods=1).min()
    
    # 計算當前價格相對於高低點的位置 (0-1之間)
    # 0 = 在低點，1 = 在高點
    df['position_24h'] = (df['close'] - df['low_24h']) / (df['high_24h'] - df['low_24h'])
    df['position_48h'] = (df['close'] - df['low_48h']) / (df['high_48h'] - df['low_48h'])
    
    # 計算距離高點的百分比（負值表示低於高點）
    df['dist_from_high_24h_pct'] = (df['close'] - df['high_24h']) / df['high_24h'] * 100
    df['dist_from_high_48h_pct'] = (df['close'] - df['high_48h']) / df['high_48h'] * 100
    
    # 計算距離低點的百分比（正值表示高於低點）
    df['dist_from_low_24h_pct'] = (df['close'] - df['low_24h']) / df['low_24h'] * 100
    df['dist_from_low_48h_pct'] = (df['close'] - df['low_48h']) / df['low_48h'] * 100
    
    # 計算價格波動範圍
    df['range_24h_pct'] = (df['high_24h'] - df['low_24h']) / df['low_24h'] * 100
    df['range_48h_pct'] = (df['high_48h'] - df['low_48h']) / df['low_48h'] * 100
    
    print(f"\n✅ 計算完成")
    print(f"新增特徵:")
    print(f"  - position_24h/48h: 價格在區間中的相對位置 (0-1)")
    print(f"  - dist_from_high_24h/48h_pct: 距離高點的百分比")
    print(f"  - dist_from_low_24h/48h_pct: 距離低點的百分比")
    print(f"  - range_24h/48h_pct: 價格波動範圍")
    
    return df


def analyze_reversal_price_positions(df):
    """分析反轉點的價格位置特徵"""
    print("\n" + "=" * 80)
    print("📈 反轉點價格位置分析")
    print("=" * 80)
    
    reversal_df = df[df['reverse'] == True].copy()
    non_reversal_df = df[df['reverse'] == False].copy()
    
    # 分析價格相對位置
    print(f"\n{'特徵':<30} {'反轉點平均':>15} {'非反轉點平均':>15} {'差異':>10}")
    print("-" * 80)
    
    features = [
        'position_24h',
        'position_48h',
        'dist_from_high_24h_pct',
        'dist_from_high_48h_pct',
        'dist_from_low_24h_pct',
        'dist_from_low_48h_pct',
        'range_24h_pct',
        'range_48h_pct'
    ]
    
    for feature in features:
        rev_mean = reversal_df[feature].mean()
        non_rev_mean = non_reversal_df[feature].mean()
        diff = rev_mean - non_rev_mean
        
        print(f"{feature:<30} {rev_mean:>15.3f} {non_rev_mean:>15.3f} {diff:>10.3f}")
    
    # 分析反轉點的位置分佈
    print("\n" + "=" * 80)
    print("📊 反轉點位置分佈")
    print("=" * 80)
    
    print(f"\n24小時相對位置 (position_24h):")
    print(f"  最小值: {reversal_df['position_24h'].min():.3f}")
    print(f"  25%分位: {reversal_df['position_24h'].quantile(0.25):.3f}")
    print(f"  中位數: {reversal_df['position_24h'].median():.3f}")
    print(f"  75%分位: {reversal_df['position_24h'].quantile(0.75):.3f}")
    print(f"  最大值: {reversal_df['position_24h'].max():.3f}")
    
    # 統計極端位置的反轉點
    near_low_24h = (reversal_df['position_24h'] < 0.2).sum()
    near_high_24h = (reversal_df['position_24h'] > 0.8).sum()
    middle_24h = (reversal_df['position_24h'].between(0.4, 0.6)).sum()
    
    print(f"\n位置分佈:")
    print(f"  接近低點 (<0.2): {near_low_24h}/{len(reversal_df)} ({near_low_24h/len(reversal_df)*100:.1f}%)")
    print(f"  接近高點 (>0.8): {near_high_24h}/{len(reversal_df)} ({near_high_24h/len(reversal_df)*100:.1f}%)")
    print(f"  中間區域 (0.4-0.6): {middle_24h}/{len(reversal_df)} ({middle_24h/len(reversal_df)*100:.1f}%)")
    
    # 分析距離高低點的百分比
    print(f"\n距離高點百分比 (dist_from_high_24h_pct):")
    print(f"  平均: {reversal_df['dist_from_high_24h_pct'].mean():.2f}%")
    print(f"  中位數: {reversal_df['dist_from_high_24h_pct'].median():.2f}%")
    
    print(f"\n距離低點百分比 (dist_from_low_24h_pct):")
    print(f"  平均: {reversal_df['dist_from_low_24h_pct'].mean():.2f}%")
    print(f"  中位數: {reversal_df['dist_from_low_24h_pct'].median():.2f}%")


def test_position_based_rules(df):
    """測試基於價格位置的規則"""
    print("\n" + "=" * 80)
    print("🧪 測試價格位置規則")
    print("=" * 80)
    
    total_reversals = df['reverse'].sum()
    
    position_rules = []
    
    # 規則1: 接近24小時低點 + 已經反彈一點
    rule1 = (df['position_24h'] < 0.3) & (df['dist_from_low_24h_pct'] > 0.5)
    position_rules.append(("接近低點+小反彈", rule1))
    
    # 規則2: 接近24小時高點 + 已經回落一點
    rule2 = (df['position_24h'] > 0.7) & (df['dist_from_high_24h_pct'] < -0.5)
    position_rules.append(("接近高點+小回落", rule2))
    
    # 規則3: 極度接近低點
    rule3 = df['position_24h'] < 0.1
    position_rules.append(("極度接近低點", rule3))
    
    # 規則4: 極度接近高點
    rule4 = df['position_24h'] > 0.9
    position_rules.append(("極度接近高點", rule4))
    
    # 規則5: 距離低點 1-3%（小幅反彈）
    rule5 = df['dist_from_low_24h_pct'].between(1, 3)
    position_rules.append(("距低點1-3%", rule5))
    
    # 規則6: 距離高點 -1% 到 -3%（小幅回落）
    rule6 = df['dist_from_high_24h_pct'].between(-3, -1)
    position_rules.append(("距高點-1到-3%", rule6))
    
    # 規則7: 大幅回撤（距離高點 > 5%）
    rule7 = df['dist_from_high_24h_pct'] < -5
    position_rules.append(("大幅回撤>5%", rule7))
    
    # 規則8: 大幅反彈（距離低點 > 5%）
    rule8 = df['dist_from_low_24h_pct'] > 5
    position_rules.append(("大幅反彈>5%", rule8))
    
    # 規則9: 48小時極端位置
    rule9 = (df['position_48h'] < 0.15) | (df['position_48h'] > 0.85)
    position_rules.append(("48h極端位置", rule9))
    
    # 規則10: 高波動區間 + 極端位置
    rule10 = (df['range_24h_pct'] > df['range_24h_pct'].quantile(0.75)) & \
             ((df['position_24h'] < 0.2) | (df['position_24h'] > 0.8))
    position_rules.append(("高波動+極端位置", rule10))
    
    print(f"\n{'規則':<25} {'命中反轉':>10} {'總信號':>10} {'精確度':>10} {'召回率':>10} {'F1':>10}")
    print("-" * 90)
    
    results = []
    for rule_name, condition in position_rules:
        hits = (condition & df['reverse']).sum()
        total_signals = condition.sum()
        precision = hits / total_signals * 100 if total_signals > 0 else 0
        recall = hits / total_reversals * 100 if total_reversals > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{rule_name:<25} {hits:>10} {total_signals:>10} {precision:>9.1f}% {recall:>9.1f}% {f1:>9.1f}")
        results.append({
            'rule': rule_name,
            'hits': hits,
            'signals': total_signals,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return results


def combine_position_with_indicators(df):
    """結合價格位置和技術指標"""
    print("\n" + "=" * 80)
    print("🔗 價格位置 + 技術指標組合")
    print("=" * 80)
    
    total_reversals = df['reverse'].sum()
    
    combined_rules = []
    
    # 組合1: 接近低點 + RSI超賣
    rule1 = (df['position_24h'] < 0.2) & (df['rsi'] < 30)
    combined_rules.append(("低點+RSI超賣", rule1))
    
    # 組合2: 接近高點 + RSI超買
    rule2 = (df['position_24h'] > 0.8) & (df['rsi'] > 70)
    combined_rules.append(("高點+RSI超買", rule2))
    
    # 組合3: 接近低點 + BB下軌
    rule3 = (df['position_24h'] < 0.2) & (df['bb_position'] < 0.2)
    combined_rules.append(("低點+BB下軌", rule3))
    
    # 組合4: 接近高點 + BB上軌
    rule4 = (df['position_24h'] > 0.8) & (df['bb_position'] > 0.8)
    combined_rules.append(("高點+BB上軌", rule4))
    
    # 組合5: 小幅反彈 + 成交量放大
    rule5 = (df['dist_from_low_24h_pct'].between(1, 3)) & (df['volume_trend'] > 1.5)
    combined_rules.append(("小反彈+高量", rule5))
    
    # 組合6: 小幅回落 + 成交量放大
    rule6 = (df['dist_from_high_24h_pct'].between(-3, -1)) & (df['volume_trend'] > 1.5)
    combined_rules.append(("小回落+高量", rule6))
    
    # 組合7: 極端位置 + 波動率加速
    rule7 = ((df['position_24h'] < 0.15) | (df['position_24h'] > 0.85)) & (df['atr_2nd'] > 0.5)
    combined_rules.append(("極端位置+波動加速", rule7))
    
    # 組合8: 接近低點 + MACD轉正
    rule8 = (df['position_24h'] < 0.3) & (df['macd_histogram'] > 0)
    combined_rules.append(("低點+MACD轉正", rule8))
    
    # 組合9: 接近高點 + MACD轉負
    rule9 = (df['position_24h'] > 0.7) & (df['macd_histogram'] < 0)
    combined_rules.append(("高點+MACD轉負", rule9))
    
    # 組合10: 48h極端 + RSI極端 + 高量
    rule10 = ((df['position_48h'] < 0.15) | (df['position_48h'] > 0.85)) & \
             ((df['rsi'] < 30) | (df['rsi'] > 70)) & \
             (df['volume_trend'] > 1.5)
    combined_rules.append(("48h極端+RSI極端+高量", rule10))
    
    print(f"\n{'組合規則':<30} {'命中反轉':>10} {'總信號':>10} {'精確度':>10} {'召回率':>10} {'F1':>10}")
    print("-" * 100)
    
    results = []
    for rule_name, condition in combined_rules:
        hits = (condition & df['reverse']).sum()
        total_signals = condition.sum()
        precision = hits / total_signals * 100 if total_signals > 0 else 0
        recall = hits / total_reversals * 100 if total_reversals > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{rule_name:<30} {hits:>10} {total_signals:>10} {precision:>9.1f}% {recall:>9.1f}% {f1:>9.1f}")
        results.append({
            'rule': rule_name,
            'hits': hits,
            'signals': total_signals,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return results


def analyze_reversal_scenarios_by_position(df):
    """根據價格位置分類反轉點場景"""
    print("\n" + "=" * 80)
    print("🎯 反轉點場景分類（按價格位置）")
    print("=" * 80)
    
    reversal_df = df[df['reverse'] == True].copy()
    
    # 場景分類
    scenarios = []
    
    # 場景1: 底部反轉（接近低點）
    scenario1 = reversal_df[reversal_df['position_24h'] < 0.3]
    scenarios.append(("底部反轉", scenario1))
    
    # 場景2: 頂部反轉（接近高點）
    scenario2 = reversal_df[reversal_df['position_24h'] > 0.7]
    scenarios.append(("頂部反轉", scenario2))
    
    # 場景3: 中間反轉
    scenario3 = reversal_df[reversal_df['position_24h'].between(0.3, 0.7)]
    scenarios.append(("中間反轉", scenario3))
    
    print(f"\n{'場景':<20} {'數量':>8} {'佔比':>10} {'平均位置':>12} {'平均RSI':>10} {'平均Vol':>10}")
    print("-" * 85)
    
    for scenario_name, scenario_df in scenarios:
        if len(scenario_df) > 0:
            count = len(scenario_df)
            percentage = count / len(reversal_df) * 100
            avg_pos = scenario_df['position_24h'].mean()
            avg_rsi = scenario_df['rsi'].mean()
            avg_vol = scenario_df['volume_trend'].mean()
            
            print(f"{scenario_name:<20} {count:>8} {percentage:>9.1f}% {avg_pos:>12.3f} {avg_rsi:>10.2f} {avg_vol:>10.2f}")
            
            # 顯示該場景的詳細統計
            if count > 0:
                print(f"  距離低點: {scenario_df['dist_from_low_24h_pct'].mean():.2f}%")
                print(f"  距離高點: {scenario_df['dist_from_high_24h_pct'].mean():.2f}%")
                print()


def main(data_path='analysis_0911_0913 - data_with_reversal_labels.csv'):
    """主函數"""
    print("=" * 80)
    print("🚀 價格相對位置分析")
    print("=" * 80)
    
    # 載入數據
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"\n✅ 數據載入完成: {len(df)} 筆資料, {df['reverse'].sum()} 個反轉點")
    
    # 計算價格位置特徵
    df = calculate_price_position_features(df)
    
    # 分析反轉點的價格位置
    analyze_reversal_price_positions(df)
    
    # 測試基於位置的規則
    position_results = test_position_based_rules(df)
    
    # 結合位置和技術指標
    combined_results = combine_position_with_indicators(df)
    
    # 場景分類
    analyze_reversal_scenarios_by_position(df)
    
    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)
    
    # 找出最佳規則
    all_results = position_results + combined_results
    all_results.sort(key=lambda x: x['f1'], reverse=True)
    
    print("\n🏆 最佳規則 (Top 10):")
    print(f"\n{'規則':<30} {'精確度':>10} {'召回率':>10} {'F1':>10}")
    print("-" * 70)
    for result in all_results[:10]:
        if result['f1'] > 0:
            print(f"{result['rule']:<30} {result['precision']:>9.1f}% {result['recall']:>9.1f}% {result['f1']:>9.1f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='價格相對位置分析')
    parser.add_argument('--data', type=str, 
                        default='analysis_0911_0913 - data_with_reversal_labels.csv',
                        help='Path to training data CSV')
    
    args = parser.parse_args()
    
    main(args.data)
