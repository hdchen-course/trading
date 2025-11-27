#!/usr/bin/env python3
"""
分析真實反轉點 vs 假反轉點的特徵差異

目標：
1. 找出真實反轉點具備的特徵
2. 找出假反轉點具備但真實反轉點不具備的特徵
3. 用這些差異來設計過濾規則

方法：
- 從階段1的 7,043 個候選點中
- 分為兩組：真陽性（3,224）和假陽性（3,819）
- 對比兩組的特徵分佈
- 找出顯著差異
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from improved_stage1 import find_local_extremes, load_data


def calculate_features(df, idx, extremes, peaks, valleys):
    """
    計算候選點的所有特徵
    
    Args:
        df: DataFrame
        idx: 候選點索引（當前K線）
        extremes: 所有局部極值
        peaks: 波峰
        valleys: 波谷
    """
    features = {}
    
    # 基本信息
    features['index'] = idx
    features['price'] = df['close'].iloc[idx]
    
    # 找出對應的反轉點（前1-2根K線）
    reversal_idx = None
    for offset in [1, 2]:
        check_idx = idx - offset
        if check_idx in extremes:
            reversal_idx = check_idx
            features['reversal_offset'] = offset
            break
    
    if reversal_idx is None:
        return None
    
    features['reversal_idx'] = reversal_idx
    features['reversal_price'] = df['close'].iloc[reversal_idx]
    
    # 判斷是波峰還是波谷
    features['is_peak'] = reversal_idx in peaks
    features['is_valley'] = reversal_idx in valleys
    
    # === 1. 價格位置特徵 ===
    # 24小時價格位置（Task 2 最佳特徵）
    lookback_24h = 96  # 24小時 = 96個15分鐘K線
    if reversal_idx >= lookback_24h:
        high_24h = df['close'].iloc[reversal_idx-lookback_24h:reversal_idx+1].max()
        low_24h = df['close'].iloc[reversal_idx-lookback_24h:reversal_idx+1].min()
        range_24h = high_24h - low_24h
        
        if range_24h > 0:
            features['price_position_24h'] = (features['reversal_price'] - low_24h) / range_24h
        else:
            features['price_position_24h'] = 0.5
    else:
        features['price_position_24h'] = 0.5
    
    # 是否在極端位置
    features['is_extreme_high'] = features['price_position_24h'] > 0.8
    features['is_extreme_low'] = features['price_position_24h'] < 0.2
    
    # === 2. 波動特徵 ===
    # ATR 及其導數
    if 'atr' in df.columns:
        features['atr'] = df['atr'].iloc[reversal_idx]
        
        # ATR 一階導數
        if reversal_idx >= 1:
            features['atr_1st'] = df['atr'].iloc[reversal_idx] - df['atr'].iloc[reversal_idx-1]
        
        # ATR 二階導數（Task 2 關鍵特徵）
        if reversal_idx >= 2:
            atr_1st_prev = df['atr'].iloc[reversal_idx-1] - df['atr'].iloc[reversal_idx-2]
            features['atr_2nd'] = features['atr_1st'] - atr_1st_prev
    
    # 價格波動範圍
    lookback = 20
    if reversal_idx >= lookback:
        window_prices = df['close'].iloc[reversal_idx-lookback:reversal_idx+1]
        price_range = window_prices.max() - window_prices.min()
        features['price_range_pct'] = price_range / window_prices.mean() * 100
    
    # === 3. 動量特徵 ===
    # RSI
    if 'rsi' in df.columns:
        features['rsi'] = df['rsi'].iloc[reversal_idx]
        features['rsi_overbought'] = features['rsi'] > 70
        features['rsi_oversold'] = features['rsi'] < 30
    
    # MACD
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        features['macd'] = df['macd'].iloc[reversal_idx]
        features['macd_signal'] = df['macd_signal'].iloc[reversal_idx]
        features['macd_diff'] = features['macd'] - features['macd_signal']
    
    # === 4. 成交量特徵 ===
    if 'volume' in df.columns:
        features['volume'] = df['volume'].iloc[reversal_idx]
        
        # 成交量比率
        if reversal_idx >= 20:
            volume_ma = df['volume'].iloc[reversal_idx-20:reversal_idx].mean()
            if volume_ma > 0:
                features['volume_ratio'] = features['volume'] / volume_ma
            else:
                features['volume_ratio'] = 1.0
        else:
            features['volume_ratio'] = 1.0
        
        # 成交量是否放大
        features['volume_surge'] = features['volume_ratio'] > 1.5
    
    # === 5. 反轉確認特徵 ===
    # 價格變化（反轉點到當前）
    features['price_change_pct'] = (features['price'] - features['reversal_price']) / features['reversal_price'] * 100
    
    # 是否確認反轉方向
    if features['is_peak']:
        features['confirms_direction'] = features['price'] < features['reversal_price']
        features['strong_confirmation'] = features['price'] < features['reversal_price'] * 0.98
    else:
        features['confirms_direction'] = features['price'] > features['reversal_price']
        features['strong_confirmation'] = features['price'] > features['reversal_price'] * 1.02
    
    # === 6. 結構特徵 ===
    # 距離上一個極值的時間
    prev_extremes = [e for e in extremes if e < reversal_idx]
    if prev_extremes:
        features['time_since_last_extreme'] = reversal_idx - prev_extremes[-1]
    else:
        features['time_since_last_extreme'] = 0
    
    # 極值的突出度（相對於左右兩側）
    if reversal_idx >= 10 and reversal_idx < len(df) - 10:
        left_prices = df['close'].iloc[reversal_idx-10:reversal_idx]
        right_prices = df['close'].iloc[reversal_idx+1:reversal_idx+11]
        
        if features['is_peak']:
            features['prominence'] = min(
                features['reversal_price'] - left_prices.mean(),
                features['reversal_price'] - right_prices.mean()
            ) / features['reversal_price'] * 100
        else:
            features['prominence'] = min(
                left_prices.mean() - features['reversal_price'],
                right_prices.mean() - features['reversal_price']
            ) / features['reversal_price'] * 100
    
    return features


def analyze_candidates():
    """分析所有候選點的特徵"""
    print("="*80)
    print("🔬 分析真實反轉點 vs 假反轉點")
    print("="*80)
    
    # 載入數據
    df = load_data()
    if df is None:
        return
    
    true_reversals = set(np.where(df['reverse'].values)[0])
    
    print(f"\n數據概況:")
    print(f"  總數據點: {len(df)}")
    print(f"  真實反轉點: {len(true_reversals)}")
    
    # 運行階段1檢測
    print("\n運行階段1檢測...")
    prices = df['close'].values
    extremes, peaks, valleys = find_local_extremes(prices, distance=20)
    
    print(f"  找到 {len(extremes)} 個局部極值")
    
    # 生成候選點
    candidates = []
    for i in range(20, len(df)):
        for offset in [1, 2]:
            check_idx = i - offset
            if check_idx in extremes:
                if i not in candidates:
                    candidates.append(i)
                break
    
    print(f"  生成 {len(candidates)} 個候選點")
    
    # 提取特徵
    print("\n提取特徵...")
    true_positive_features = []
    false_positive_features = []
    
    for cand_idx in candidates:
        features = calculate_features(df, cand_idx, extremes, peaks, valleys)
        
        if features is None:
            continue
        
        # 判斷是真陽性還是假陽性
        reversal_idx = features['reversal_idx']
        
        if reversal_idx in true_reversals:
            features['label'] = 'true_positive'
            true_positive_features.append(features)
        else:
            features['label'] = 'false_positive'
            false_positive_features.append(features)
    
    print(f"  真陽性: {len(true_positive_features)}")
    print(f"  假陽性: {len(false_positive_features)}")
    
    # 轉換為 DataFrame
    df_true = pd.DataFrame(true_positive_features)
    df_false = pd.DataFrame(false_positive_features)
    
    return df_true, df_false


def compare_features(df_true, df_false):
    """對比特徵分佈"""
    print("\n" + "="*80)
    print("📊 特徵對比分析")
    print("="*80)
    
    # 數值特徵
    numeric_features = [
        'price_position_24h',
        'atr_2nd',
        'price_range_pct',
        'rsi',
        'volume_ratio',
        'price_change_pct',
        'time_since_last_extreme',
        'prominence'
    ]
    
    print(f"\n{'特徵':<30} {'真陽性均值':<15} {'假陽性均值':<15} {'差異':<10} {'顯著性':<10}")
    print("-"*90)
    
    significant_features = []
    
    for feature in numeric_features:
        if feature not in df_true.columns or feature not in df_false.columns:
            continue
        
        true_mean = df_true[feature].mean()
        false_mean = df_false[feature].mean()
        diff = true_mean - false_mean
        diff_pct = abs(diff) / (abs(true_mean) + 1e-10) * 100
        
        # 判斷顯著性
        if diff_pct > 20:  # 差異超過20%
            significance = "⭐⭐⭐"
            significant_features.append((feature, diff, diff_pct))
        elif diff_pct > 10:
            significance = "⭐⭐"
            significant_features.append((feature, diff, diff_pct))
        elif diff_pct > 5:
            significance = "⭐"
        else:
            significance = ""
        
        print(f"{feature:<30} {true_mean:>14.4f} {false_mean:>14.4f} {diff:>9.4f} {significance:<10}")
    
    # 布林特徵
    print("\n" + "="*80)
    print("布林特徵對比")
    print("="*80)
    
    boolean_features = [
        'is_extreme_high',
        'is_extreme_low',
        'rsi_overbought',
        'rsi_oversold',
        'volume_surge',
        'confirms_direction',
        'strong_confirmation'
    ]
    
    print(f"\n{'特徵':<30} {'真陽性比例':<15} {'假陽性比例':<15} {'差異':<10} {'顯著性':<10}")
    print("-"*90)
    
    for feature in boolean_features:
        if feature not in df_true.columns or feature not in df_false.columns:
            continue
        
        true_pct = df_true[feature].mean() * 100
        false_pct = df_false[feature].mean() * 100
        diff = true_pct - false_pct
        
        # 判斷顯著性
        if abs(diff) > 20:
            significance = "⭐⭐⭐"
            significant_features.append((feature, diff, abs(diff)))
        elif abs(diff) > 10:
            significance = "⭐⭐"
            significant_features.append((feature, diff, abs(diff)))
        elif abs(diff) > 5:
            significance = "⭐"
        else:
            significance = ""
        
        print(f"{feature:<30} {true_pct:>14.1f}% {false_pct:>14.1f}% {diff:>9.1f}% {significance:<10}")
    
    return significant_features


def suggest_filters(significant_features):
    """根據顯著特徵建議過濾規則"""
    print("\n" + "="*80)
    print("💡 建議的過濾規則")
    print("="*80)
    
    # 排序：按顯著性排序
    significant_features.sort(key=lambda x: x[2], reverse=True)
    
    print("\n最顯著的特徵（可用於過濾）：")
    print("-"*80)
    
    for i, (feature, diff, significance) in enumerate(significant_features[:10], 1):
        print(f"\n{i}. {feature}")
        print(f"   差異: {diff:.4f}")
        print(f"   顯著性: {significance:.1f}%")
        
        # 建議過濾規則
        if 'price_position' in feature:
            if diff > 0:
                print(f"   建議: 保留價格位置 > 某閾值的候選點")
            else:
                print(f"   建議: 保留價格位置 < 某閾值的候選點")
        
        elif 'atr_2nd' in feature:
            if diff > 0:
                print(f"   建議: 保留 ATR 加速 > 某閾值的候選點")
            else:
                print(f"   建議: 過濾 ATR 加速過大的候選點")
        
        elif 'volume_ratio' in feature:
            if diff > 0:
                print(f"   建議: 保留成交量放大的候選點")
            else:
                print(f"   建議: 過濾成交量過大的候選點")
        
        elif 'rsi' in feature:
            print(f"   建議: 使用 RSI 極值過濾")
        
        elif 'prominence' in feature:
            if diff > 0:
                print(f"   建議: 保留突出度較高的候選點")
            else:
                print(f"   建議: 過濾突出度過高的候選點")


def main():
    """主函數"""
    # 分析候選點
    df_true, df_false = analyze_candidates()
    
    # 對比特徵
    significant_features = compare_features(df_true, df_false)
    
    # 建議過濾規則
    suggest_filters(significant_features)
    
    # 保存結果
    print("\n" + "="*80)
    print("💾 保存分析結果")
    print("="*80)
    
    df_true.to_csv('experiments/reversal_detection/true_positive_features.csv', index=False)
    df_false.to_csv('experiments/reversal_detection/false_positive_features.csv', index=False)
    
    print("\n已保存:")
    print("  - true_positive_features.csv")
    print("  - false_positive_features.csv")
    
    print("\n這些文件可用於:")
    print("  1. 進一步的特徵分析")
    print("  2. 機器學習模型訓練")
    print("  3. 規則過濾器設計")


if __name__ == '__main__':
    main()
