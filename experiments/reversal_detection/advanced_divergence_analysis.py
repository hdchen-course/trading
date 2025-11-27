#!/usr/bin/env python3
"""
進階背離模式分析

使用 z-score 標準化的二階導數進行多維度背離分析
考慮背離的強度和多指標組合
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from scipy import stats


def load_and_prepare_data(data_path='analysis_0911_0913 - data_with_reversal_labels.csv'):
    """載入並準備數據"""
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 計算一階和二階導數
    df['price_1st'] = df['close'].diff()
    df['vol_1st'] = df['volume'].diff()
    df['macd_1st'] = df['macd'].diff()
    df['atr_1st'] = df['atr'].diff()
    
    df['price_2nd_calc'] = df['price_1st'].diff()
    df['vol_2nd_calc'] = df['vol_1st'].diff()
    df['macd_2nd_calc'] = df['macd_1st'].diff()
    df['atr_2nd_calc'] = df['atr_1st'].diff()
    
    # 使用計算值替換原始的全0值
    if df['price_2nd'].abs().sum() == 0:
        df['price_2nd'] = df['price_2nd_calc']
        df['vol_2nd'] = df['vol_2nd_calc']
        df['macd_2nd'] = df['macd_2nd_calc']
        df['atr_2nd'] = df['atr_2nd_calc']
    
    return df


def calculate_zscore_derivatives(df):
    """使用數據中已有的 z-score 標準化二階導數"""
    print("=" * 80)
    print("📊 使用已有的 Z-Score 標準化二階導數")
    print("=" * 80)
    
    # 數據中的 price_2nd, vol_2nd, macd_2nd, atr_2nd 已經是 z-score
    # 直接使用，不需要重新計算
    df['price_2nd_z'] = df['price_2nd']
    df['vol_2nd_z'] = df['vol_2nd']
    df['macd_2nd_z'] = df['macd_2nd']
    df['atr_2nd_z'] = df['atr_2nd']
    
    # 顯示統計信息
    print(f"\nZ-Score 統計 (原始數據):")
    print(f"{'指標':<20} {'平均值':>10} {'標準差':>10} {'最小值':>10} {'最大值':>10} {'非零數':>10}")
    print("-" * 90)
    
    for col in ['price_2nd_z', 'vol_2nd_z', 'macd_2nd_z', 'atr_2nd_z']:
        mean = df[col].mean()
        std = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        non_zero = (df[col] != 0).sum()
        print(f"{col:<20} {mean:>10.3f} {std:>10.3f} {min_val:>10.3f} {max_val:>10.3f} {non_zero:>10}")
    
    return df


def analyze_multi_dimensional_divergence(df):
    """分析多維度背離模式"""
    print("\n" + "=" * 80)
    print("🔍 多維度背離模式分析（使用原始 Z-Score）")
    print("=" * 80)
    
    total_reversals = df['reverse'].sum()
    
    # ========== 第一部分：單純方向分析（不考慮強度）==========
    print("\n" + "=" * 80)
    print("📊 Part 1: 二階導數方向分析（不考慮強度）")
    print("=" * 80)
    
    direction_patterns = []
    
    # 1. 價格加速上漲 (price_2nd > 0)
    pattern1 = df['price_2nd_z'] > 0
    direction_patterns.append(("價格加速上漲", pattern1))
    
    # 2. 價格加速下跌 (price_2nd < 0)
    pattern2 = df['price_2nd_z'] < 0
    direction_patterns.append(("價格加速下跌", pattern2))
    
    # 3. 成交量加速增加
    pattern3 = df['vol_2nd_z'] > 0
    direction_patterns.append(("成交量加速增", pattern3))
    
    # 4. 成交量加速減少
    pattern4 = df['vol_2nd_z'] < 0
    direction_patterns.append(("成交量加速減", pattern4))
    
    # 5. 價格加速 + 成交量減速
    pattern5 = (df['price_2nd_z'] > 0) & (df['vol_2nd_z'] < 0)
    direction_patterns.append(("價加速量減速", pattern5))
    
    # 6. 價格減速 + 成交量加速
    pattern6 = (df['price_2nd_z'] < 0) & (df['vol_2nd_z'] > 0)
    direction_patterns.append(("價減速量加速", pattern6))
    
    # 7. MACD加速轉正
    pattern7 = df['macd_2nd_z'] > 0
    direction_patterns.append(("MACD加速轉正", pattern7))
    
    # 8. MACD加速轉負
    pattern8 = df['macd_2nd_z'] < 0
    direction_patterns.append(("MACD加速轉負", pattern8))
    
    # 9. 波動率加速上升
    pattern9 = df['atr_2nd_z'] > 0
    direction_patterns.append(("波動率加速升", pattern9))
    
    # 10. 三重加速 (price, vol, macd 都加速)
    pattern10 = (df['price_2nd_z'] > 0) & (df['vol_2nd_z'] > 0) & (df['macd_2nd_z'] > 0)
    direction_patterns.append(("三重加速", pattern10))
    
    print(f"\n{'模式':<20} {'反轉點命中':>12} {'總信號':>10} {'精確度':>10} {'召回率':>10} {'F1':>10}")
    print("-" * 90)
    
    direction_results = []
    for pattern_name, condition in direction_patterns:
        hits = (condition & df['reverse']).sum()
        total_signals = condition.sum()
        precision = hits / total_signals * 100 if total_signals > 0 else 0
        recall = hits / total_reversals * 100 if total_reversals > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{pattern_name:<20} {hits:>12} {total_signals:>10} {precision:>9.1f}% {recall:>9.1f}% {f1:>9.1f}")
        direction_results.append({
            'pattern': pattern_name,
            'hits': hits,
            'signals': total_signals,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # ========== 第二部分：強度組合分析 ==========
    print("\n" + "=" * 80)
    print("📊 Part 2: 強度組合分析（使用閾值）")
    print("=" * 80)
    
    # 定義背離強度閾值（基於 z-score）
    weak_threshold = 0.5    # 弱信號（0.5個標準差）
    medium_threshold = 1.0  # 中等信號（1個標準差）
    strong_threshold = 1.5  # 強信號（1.5個標準差）
    
    print(f"\n使用閾值: 弱={weak_threshold}, 中={medium_threshold}, 強={strong_threshold}")
    
    divergence_patterns = []
    
    # ========== 三維背離模式 ==========
    
    # 1. 價格強加速 + 成交量弱減速 + MACD弱加速（頂部背離）
    pattern1 = (
        (df['price_2nd_z'] > strong_threshold) &      # 價格強加速
        (df['vol_2nd_z'] < -weak_threshold) &         # 成交量減速
        (df['macd_2nd_z'] < medium_threshold)         # MACD弱加速
    )
    divergence_patterns.append(("價強加速+量弱減+MACD弱", pattern1))
    
    # 2. 價格強減速 + 成交量強加速 + MACD減速（底部背離）
    pattern2 = (
        (df['price_2nd_z'] < -strong_threshold) &     # 價格強減速
        (df['vol_2nd_z'] > strong_threshold) &        # 成交量強加速
        (df['macd_2nd_z'] < -weak_threshold)          # MACD減速
    )
    divergence_patterns.append(("價強減速+量強加速+MACD減", pattern2))
    
    # 3. 價格中等加速 + 成交量強減速 + 波動率加速（能量耗盡）
    pattern3 = (
        (df['price_2nd_z'] > weak_threshold) &        # 價格加速
        (df['vol_2nd_z'] < -strong_threshold) &       # 成交量強減速
        (df['atr_2nd_z'] > medium_threshold)          # 波動率加速
    )
    divergence_patterns.append(("價加速+量強減+波動加速", pattern3))
    
    # 4. 價格減速 + 成交量中等加速 + 波動率強加速（恐慌性下跌）
    pattern4 = (
        (df['price_2nd_z'] < -medium_threshold) &     # 價格減速
        (df['vol_2nd_z'] > medium_threshold) &        # 成交量加速
        (df['atr_2nd_z'] > strong_threshold)          # 波動率強加速
    )
    divergence_patterns.append(("價減速+量加速+波動強加速", pattern4))
    
    # ========== 四維背離模式 ==========
    
    # 5. 價格強加速 + 成交量減速 + MACD弱 + 波動率高（頂部衰竭）
    pattern5 = (
        (df['price_2nd_z'] > strong_threshold) &      # 價格強加速
        (df['vol_2nd_z'] < -weak_threshold) &         # 成交量減速
        (df['macd_2nd_z'] < weak_threshold) &         # MACD弱
        (df['atr_2nd_z'] > medium_threshold)          # 波動率高
    )
    divergence_patterns.append(("四維頂部衰竭", pattern5))
    
    # 6. 價格強減速 + 成交量強加速 + MACD減速 + 波動率強加速（底部恐慌）
    pattern6 = (
        (df['price_2nd_z'] < -strong_threshold) &     # 價格強減速
        (df['vol_2nd_z'] > strong_threshold) &        # 成交量強加速
        (df['macd_2nd_z'] < -weak_threshold) &        # MACD減速
        (df['atr_2nd_z'] > strong_threshold)          # 波動率強加速
    )
    divergence_patterns.append(("四維底部恐慌", pattern6))
    
    # 7. 價格中等加速 + 成交量中等減速 + MACD強加速 + 波動率減速（假突破）
    pattern7 = (
        (df['price_2nd_z'] > weak_threshold) &        # 價格加速
        (df['vol_2nd_z'] < -weak_threshold) &         # 成交量減速
        (df['macd_2nd_z'] > strong_threshold) &       # MACD強加速
        (df['atr_2nd_z'] < -weak_threshold)           # 波動率減速
    )
    divergence_patterns.append(("四維假突破", pattern7))
    
    # 8. 價格弱變化 + 成交量強變化 + MACD強變化 + 波動率強變化（轉折點）
    pattern8 = (
        (df['price_2nd_z'].abs() < weak_threshold) &  # 價格弱變化
        (df['vol_2nd_z'].abs() > strong_threshold) &  # 成交量強變化
        (df['macd_2nd_z'].abs() > strong_threshold) & # MACD強變化
        (df['atr_2nd_z'].abs() > strong_threshold)    # 波動率強變化
    )
    divergence_patterns.append(("四維轉折點", pattern8))
    
    # ========== 強度分級背離 ==========
    
    # 9. 極強背離：價格和成交量方向相反且都是強信號
    pattern9 = (
        ((df['price_2nd_z'] > strong_threshold) & (df['vol_2nd_z'] < -strong_threshold)) |
        ((df['price_2nd_z'] < -strong_threshold) & (df['vol_2nd_z'] > strong_threshold))
    )
    divergence_patterns.append(("極強價量背離", pattern9))
    
    # 10. 中等背離：價格和成交量方向相反且至少一個是中等信號
    pattern10 = (
        ((df['price_2nd_z'] > medium_threshold) & (df['vol_2nd_z'] < -medium_threshold)) |
        ((df['price_2nd_z'] < -medium_threshold) & (df['vol_2nd_z'] > medium_threshold))
    ) & ~pattern9  # 排除極強背離
    divergence_patterns.append(("中等價量背離", pattern10))
    
    # 11. MACD-價格強背離
    pattern11 = (
        ((df['macd_2nd_z'] > strong_threshold) & (df['price_2nd_z'] < -medium_threshold)) |
        ((df['macd_2nd_z'] < -strong_threshold) & (df['price_2nd_z'] > medium_threshold))
    )
    divergence_patterns.append(("MACD-價格強背離", pattern11))
    
    # 12. 三重強背離：價格、成交量、MACD都是強信號但方向不一致
    pattern12 = (
        (df['price_2nd_z'].abs() > strong_threshold) &
        (df['vol_2nd_z'].abs() > strong_threshold) &
        (df['macd_2nd_z'].abs() > strong_threshold) &
        (
            ((df['price_2nd_z'] > 0) & (df['vol_2nd_z'] < 0)) |
            ((df['price_2nd_z'] < 0) & (df['vol_2nd_z'] > 0)) |
            ((df['price_2nd_z'] > 0) & (df['macd_2nd_z'] < 0)) |
            ((df['price_2nd_z'] < 0) & (df['macd_2nd_z'] > 0))
        )
    )
    divergence_patterns.append(("三重強背離", pattern12))
    
    # 測試所有模式
    print(f"\n{'模式':<30} {'命中反轉':>10} {'總信號':>10} {'精確度':>10} {'召回率':>10} {'F1':>10}")
    print("-" * 100)
    
    results = []
    for pattern_name, condition in divergence_patterns:
        hits = (condition & df['reverse']).sum()
        total_signals = condition.sum()
        precision = hits / total_signals * 100 if total_signals > 0 else 0
        recall = hits / total_reversals * 100 if total_reversals > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{pattern_name:<30} {hits:>10} {total_signals:>10} {precision:>9.1f}% {recall:>9.1f}% {f1:>9.1f}")
        results.append({
            'pattern': pattern_name,
            'hits': hits,
            'signals': total_signals,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # 合併兩部分結果
    all_results = {
        'direction_patterns': direction_results,
        'strength_patterns': results
    }
    
    return all_results, df


def analyze_divergence_strength_distribution(df):
    """分析背離強度分佈"""
    print("\n" + "=" * 80)
    print("📈 反轉點的背離強度分佈")
    print("=" * 80)
    
    reversal_df = df[df['reverse'] == True].copy()
    non_reversal_df = df[df['reverse'] == False].copy()
    
    print(f"\n{'指標':<20} {'反轉點平均Z':>15} {'非反轉點平均Z':>15} {'差異':>10}")
    print("-" * 80)
    
    for col in ['price_2nd_z', 'vol_2nd_z', 'macd_2nd_z', 'atr_2nd_z']:
        rev_mean = reversal_df[col].mean()
        non_rev_mean = non_reversal_df[col].mean()
        diff = rev_mean - non_rev_mean
        
        print(f"{col:<20} {rev_mean:>15.3f} {non_rev_mean:>15.3f} {diff:>10.3f}")
    
    # 分析背離強度
    print("\n" + "=" * 80)
    print("🎯 背離強度分析")
    print("=" * 80)
    
    # 計算價量背離強度
    reversal_df['price_vol_divergence'] = reversal_df['price_2nd_z'] * (-reversal_df['vol_2nd_z'])
    non_reversal_df['price_vol_divergence'] = non_reversal_df['price_2nd_z'] * (-non_reversal_df['vol_2nd_z'])
    
    print(f"\n價量背離強度:")
    print(f"  反轉點平均: {reversal_df['price_vol_divergence'].mean():.3f}")
    print(f"  非反轉點平均: {non_reversal_df['price_vol_divergence'].mean():.3f}")
    print(f"  反轉點中位數: {reversal_df['price_vol_divergence'].median():.3f}")
    print(f"  非反轉點中位數: {non_reversal_df['price_vol_divergence'].median():.3f}")
    
    # 計算 MACD-價格背離強度
    reversal_df['macd_price_divergence'] = reversal_df['macd_2nd_z'] * (-reversal_df['price_2nd_z'])
    non_reversal_df['macd_price_divergence'] = non_reversal_df['macd_2nd_z'] * (-non_reversal_df['price_2nd_z'])
    
    print(f"\nMACD-價格背離強度:")
    print(f"  反轉點平均: {reversal_df['macd_price_divergence'].mean():.3f}")
    print(f"  非反轉點平均: {non_reversal_df['macd_price_divergence'].mean():.3f}")
    
    # 分析強背離的比例
    print("\n" + "=" * 80)
    print("💪 強背離比例分析")
    print("=" * 80)
    
    strong_threshold = 1.5
    
    # 價量強背離
    rev_strong_pv = (
        ((reversal_df['price_2nd_z'] > strong_threshold) & (reversal_df['vol_2nd_z'] < -strong_threshold)) |
        ((reversal_df['price_2nd_z'] < -strong_threshold) & (reversal_df['vol_2nd_z'] > strong_threshold))
    ).sum()
    
    non_rev_strong_pv = (
        ((non_reversal_df['price_2nd_z'] > strong_threshold) & (non_reversal_df['vol_2nd_z'] < -strong_threshold)) |
        ((non_reversal_df['price_2nd_z'] < -strong_threshold) & (non_reversal_df['vol_2nd_z'] > strong_threshold))
    ).sum()
    
    print(f"\n價量強背離:")
    print(f"  反轉點: {rev_strong_pv}/{len(reversal_df)} ({rev_strong_pv/len(reversal_df)*100:.1f}%)")
    print(f"  非反轉點: {non_rev_strong_pv}/{len(non_reversal_df)} ({non_rev_strong_pv/len(non_reversal_df)*100:.1f}%)")


def find_best_threshold_combinations(df):
    """尋找最佳閾值組合"""
    print("\n" + "=" * 80)
    print("🔬 最佳閾值組合搜索")
    print("=" * 80)
    
    total_reversals = df['reverse'].sum()
    
    # 測試不同的閾值組合
    thresholds = [0.5, 1.0, 1.5, 2.0]
    
    best_results = []
    
    print("\n測試價量背離的不同閾值組合...")
    print(f"\n{'價格閾值':>10} {'成交量閾值':>12} {'命中':>8} {'信號':>8} {'精確度':>10} {'召回率':>10} {'F1':>10}")
    print("-" * 90)
    
    for price_th in thresholds:
        for vol_th in thresholds:
            # 價格加速 + 成交量減速
            condition = (
                (df['price_2nd_z'] > price_th) & 
                (df['vol_2nd_z'] < -vol_th)
            )
            
            hits = (condition & df['reverse']).sum()
            total_signals = condition.sum()
            
            if total_signals > 0:
                precision = hits / total_signals * 100
                recall = hits / total_reversals * 100
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"{price_th:>10.1f} {vol_th:>12.1f} {hits:>8} {total_signals:>8} "
                      f"{precision:>9.1f}% {recall:>9.1f}% {f1:>9.1f}")
                
                if f1 > 5:  # 只保留 F1 > 5 的組合
                    best_results.append({
                        'type': '價漲量縮',
                        'price_th': price_th,
                        'vol_th': vol_th,
                        'f1': f1,
                        'precision': precision,
                        'recall': recall
                    })
    
    # 測試價格減速 + 成交量加速
    print("\n測試價格減速 + 成交量加速的閾值組合...")
    print(f"\n{'價格閾值':>10} {'成交量閾值':>12} {'命中':>8} {'信號':>8} {'精確度':>10} {'召回率':>10} {'F1':>10}")
    print("-" * 90)
    
    for price_th in thresholds:
        for vol_th in thresholds:
            condition = (
                (df['price_2nd_z'] < -price_th) & 
                (df['vol_2nd_z'] > vol_th)
            )
            
            hits = (condition & df['reverse']).sum()
            total_signals = condition.sum()
            
            if total_signals > 0:
                precision = hits / total_signals * 100
                recall = hits / total_reversals * 100
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"{price_th:>10.1f} {vol_th:>12.1f} {hits:>8} {total_signals:>8} "
                      f"{precision:>9.1f}% {recall:>9.1f}% {f1:>9.1f}")
                
                if f1 > 5:
                    best_results.append({
                        'type': '價跌量增',
                        'price_th': price_th,
                        'vol_th': vol_th,
                        'f1': f1,
                        'precision': precision,
                        'recall': recall
                    })
    
    # 顯示最佳組合
    if best_results:
        best_results.sort(key=lambda x: x['f1'], reverse=True)
        print("\n" + "=" * 80)
        print("🏆 最佳閾值組合 (Top 10)")
        print("=" * 80)
        print(f"\n{'類型':<12} {'價格閾值':>10} {'成交量閾值':>12} {'精確度':>10} {'召回率':>10} {'F1':>10}")
        print("-" * 80)
        
        for result in best_results[:10]:
            print(f"{result['type']:<12} {result['price_th']:>10.1f} {result['vol_th']:>12.1f} "
                  f"{result['precision']:>9.1f}% {result['recall']:>9.1f}% {result['f1']:>9.1f}")
    
    return best_results


def main(data_path='analysis_0911_0913 - data_with_reversal_labels.csv'):
    """主函數"""
    print("=" * 80)
    print("🚀 進階背離模式分析（Z-Score + 多維度）")
    print("=" * 80)
    
    # 載入數據
    df = load_and_prepare_data(data_path)
    print(f"\n✅ 數據載入完成: {len(df)} 筆資料, {df['reverse'].sum()} 個反轉點")
    
    # 計算 z-score
    df = calculate_zscore_derivatives(df)
    
    # 分析多維度背離（包含方向和強度兩部分）
    all_results, df = analyze_multi_dimensional_divergence(df)
    
    # 分析背離強度分佈
    analyze_divergence_strength_distribution(df)
    
    # 尋找最佳閾值組合
    best_thresholds = find_best_threshold_combinations(df)
    
    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)
    
    # 總結建議
    print("\n💡 建議:")
    print("1. 使用 z-score 標準化可以更好地比較不同指標的強度")
    print("2. 多維度背離（3-4個指標）比單純的價量背離更有效")
    print("3. 背離強度很重要，強背離（z-score > 1.5）的信號更可靠")
    print("4. 可以根據最佳閾值組合來調整檢測規則")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='進階背離模式分析')
    parser.add_argument('--data', type=str, 
                        default='analysis_0911_0913 - data_with_reversal_labels.csv',
                        help='Path to training data CSV')
    
    args = parser.parse_args()
    
    main(args.data)
