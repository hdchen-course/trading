#!/usr/bin/env python3
"""
階段1：高召回率 + 降低訊號數量

目標：
1. 召回率 > 95%（找到盡可能多的真實反轉點）
2. 候選點 < 10,000（從 25,235 大幅降低）

策略：
- 使用 peak_valley 的 scipy.find_peaks 找局部極值
- 分析反轉點的左側特徵來過濾假陽性
- 結合 task2 的發現（極端位置、ATR加速等）
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from scipy.signal import find_peaks


def load_data():
    """載入訓練數據"""
    path = "training_data/reversal_training_data_2022-01-01_to_2025-09-01_15m.csv"
    
    if os.path.exists(path):
        print(f"✅ 載入數據: {path}")
        return pd.read_csv(path)
    
    return None


def calculate_left_side_features(df, idx, lookback=20):
    """
    計算反轉點左側的特徵
    
    這些特徵可以幫助我們識別真實的反轉點
    """
    if idx < lookback:
        return None
    
    features = {}
    
    # 1. 價格波動幅度
    window_prices = df['close'].iloc[idx-lookback:idx+1]
    price_range = window_prices.max() - window_prices.min()
    features['price_range_pct'] = price_range / window_prices.mean() * 100
    
    # 2. 是否為局部極值
    curr_price = df['close'].iloc[idx]
    features['is_peak'] = curr_price == window_prices.max()
    features['is_valley'] = curr_price == window_prices.min()
    
    # 3. 與左側的差異
    left_prices = df['close'].iloc[idx-5:idx]
    if len(left_prices) > 0:
        features['left_diff_pct'] = abs(curr_price - left_prices.mean()) / left_prices.mean() * 100
    else:
        features['left_diff_pct'] = 0
    
    # 4. 價格位置（相對於24小時範圍）
    lookback_24h = min(96, idx)  # 96 = 24小時的15分鐘K線
    high_24h = df['close'].iloc[idx-lookback_24h:idx+1].max()
    low_24h = df['close'].iloc[idx-lookback_24h:idx+1].min()
    range_24h = high_24h - low_24h
    if range_24h > 0:
        features['price_position'] = (curr_price - low_24h) / range_24h
    else:
        features['price_position'] = 0.5
    
    # 5. 成交量
    if 'volume' in df.columns and idx >= 20:
        vol_ma = df['volume'].iloc[idx-20:idx].mean()
        features['volume_ratio'] = df['volume'].iloc[idx] / vol_ma if vol_ma > 0 else 1.0
    else:
        features['volume_ratio'] = 1.0
    
    # 6. ATR（如果有的話）
    if 'atr' in df.columns and idx >= 1:
        atr_curr = df['atr'].iloc[idx]
        atr_prev = df['atr'].iloc[idx-1]
        features['atr_change'] = (atr_curr - atr_prev) / atr_prev if atr_prev > 0 else 0
    else:
        features['atr_change'] = 0
    
    return features


def stage1_baseline(df, lookback=20):
    """
    基準版本：基礎 peak_valley
    
    只使用 scipy.find_peaks，不做任何過濾
    """
    print("\n" + "="*80)
    print("基準版本：基礎 Peak-Valley")
    print("="*80)
    
    prices = df['close'].values
    
    # 動態計算 prominence
    window_size = min(500, len(prices) // 4)
    if window_size < 100:
        window_size = len(prices)
    
    rolling_std = pd.Series(prices).rolling(window=window_size, min_periods=1).std()
    prominence = rolling_std.median() * 0.5
    
    print(f"使用 prominence: {prominence:.4f}")
    
    # 找波谷
    valleys, _ = find_peaks(-prices, distance=lookback, prominence=prominence)
    
    # 找波峰
    peaks, _ = find_peaks(prices, distance=lookback, prominence=prominence)
    
    # 合併
    candidates = np.sort(np.concatenate([peaks, valleys]))
    
    print(f"找到 {len(candidates)} 個候選點")
    print(f"  波峰: {len(peaks)}")
    print(f"  波谷: {len(valleys)}")
    
    return candidates


def stage1_with_price_range_filter(df, lookback=20, min_range_pct=0.5):
    """
    版本1：價格波動幅度過濾
    
    過濾掉波動太小的極值點
    """
    print("\n" + "="*80)
    print(f"版本1：價格波動幅度過濾 (最小 {min_range_pct}%)")
    print("="*80)
    
    # 先找基礎候選點
    base_candidates = stage1_baseline(df, lookback)
    
    # 過濾
    filtered = []
    for idx in base_candidates:
        features = calculate_left_side_features(df, idx, lookback)
        if features and features['price_range_pct'] >= min_range_pct:
            filtered.append(idx)
    
    print(f"過濾後: {len(filtered)} 個候選點 (減少 {len(base_candidates) - len(filtered)})")
    
    return np.array(filtered)


def stage1_with_extreme_position_filter(df, lookback=20, extreme_threshold=0.25):
    """
    版本2：極端位置過濾
    
    只保留在極端位置的反轉點（< 25% 或 > 75%）
    """
    print("\n" + "="*80)
    print(f"版本2：極端位置過濾 (閾值 {extreme_threshold})")
    print("="*80)
    
    # 先找基礎候選點
    base_candidates = stage1_baseline(df, lookback)
    
    # 過濾
    filtered = []
    for idx in base_candidates:
        features = calculate_left_side_features(df, idx, lookback)
        if features:
            pos = features['price_position']
            if pos < extreme_threshold or pos > (1 - extreme_threshold):
                filtered.append(idx)
    
    print(f"過濾後: {len(filtered)} 個候選點 (減少 {len(base_candidates) - len(filtered)})")
    
    return np.array(filtered)


def stage1_with_volume_filter(df, lookback=20, min_volume_ratio=1.2):
    """
    版本3：成交量過濾
    
    只保留成交量放大的反轉點
    """
    print("\n" + "="*80)
    print(f"版本3：成交量過濾 (最小比率 {min_volume_ratio}x)")
    print("="*80)
    
    # 先找基礎候選點
    base_candidates = stage1_baseline(df, lookback)
    
    # 過濾
    filtered = []
    for idx in base_candidates:
        features = calculate_left_side_features(df, idx, lookback)
        if features and features['volume_ratio'] >= min_volume_ratio:
            filtered.append(idx)
    
    print(f"過濾後: {len(filtered)} 個候選點 (減少 {len(base_candidates) - len(filtered)})")
    
    return np.array(filtered)


def stage1_combined_filters(df, lookback=20, 
                            min_range_pct=0.3,
                            extreme_threshold=0.3,
                            min_volume_ratio=1.0):
    """
    版本4：組合過濾
    
    結合多個過濾條件（OR邏輯，只要滿足一個就保留）
    """
    print("\n" + "="*80)
    print(f"版本4：組合過濾")
    print(f"  價格波動 >= {min_range_pct}%")
    print(f"  極端位置 < {extreme_threshold} or > {1-extreme_threshold}")
    print(f"  成交量比 >= {min_volume_ratio}x")
    print("="*80)
    
    # 先找基礎候選點
    base_candidates = stage1_baseline(df, lookback)
    
    # 過濾（OR邏輯）
    filtered = []
    for idx in base_candidates:
        features = calculate_left_side_features(df, idx, lookback)
        if not features:
            continue
        
        # 任一條件滿足就保留
        keep = False
        
        # 條件1：價格波動夠大
        if features['price_range_pct'] >= min_range_pct:
            keep = True
        
        # 條件2：在極端位置
        pos = features['price_position']
        if pos < extreme_threshold or pos > (1 - extreme_threshold):
            keep = True
        
        # 條件3：成交量放大
        if features['volume_ratio'] >= min_volume_ratio:
            keep = True
        
        if keep:
            filtered.append(idx)
    
    print(f"過濾後: {len(filtered)} 個候選點 (減少 {len(base_candidates) - len(filtered)})")
    
    return np.array(filtered)


def stage1_strict_combined(df, lookback=20):
    """
    版本5：嚴格組合（AND邏輯）
    
    必須同時滿足多個條件
    """
    print("\n" + "="*80)
    print(f"版本5：嚴格組合過濾 (AND邏輯)")
    print("="*80)
    
    # 先找基礎候選點
    base_candidates = stage1_baseline(df, lookback)
    
    # 過濾（AND邏輯）
    filtered = []
    for idx in base_candidates:
        features = calculate_left_side_features(df, idx, lookback)
        if not features:
            continue
        
        # 必須同時滿足
        pos = features['price_position']
        
        if (features['price_range_pct'] >= 0.5 and
            (pos < 0.3 or pos > 0.7) and
            features['volume_ratio'] >= 1.1):
            filtered.append(idx)
    
    print(f"過濾後: {len(filtered)} 個候選點 (減少 {len(base_candidates) - len(filtered)})")
    
    return np.array(filtered)


def evaluate_version(candidates, true_reversals, version_name):
    """評估版本性能"""
    detected = np.sum(np.isin(candidates, true_reversals))
    recall = detected / len(true_reversals) * 100
    precision = detected / len(candidates) * 100 if len(candidates) > 0 else 0
    
    return {
        'version': version_name,
        'candidates': len(candidates),
        'detected': detected,
        'recall': recall,
        'precision': precision,
        'missed': len(true_reversals) - detected
    }


def compare_versions():
    """對比不同版本"""
    print("="*80)
    print("🔬 階段1：高召回率 + 降低訊號數量")
    print("="*80)
    
    # 載入數據
    df = load_data()
    if df is None:
        return
    
    true_reversals = np.where(df['reverse'].values)[0]
    
    print(f"\n數據概況:")
    print(f"  總數據點: {len(df)}")
    print(f"  真實反轉點: {len(true_reversals)} ({len(true_reversals)/len(df)*100:.2f}%)")
    
    # 測試不同版本
    versions = [
        ("基準: 基礎Peak-Valley", lambda: stage1_baseline(df)),
        ("版本1: 波動過濾(0.5%)", lambda: stage1_with_price_range_filter(df, min_range_pct=0.5)),
        ("版本1b: 波動過濾(1.0%)", lambda: stage1_with_price_range_filter(df, min_range_pct=1.0)),
        ("版本2: 極端位置(0.25)", lambda: stage1_with_extreme_position_filter(df, extreme_threshold=0.25)),
        ("版本2b: 極端位置(0.30)", lambda: stage1_with_extreme_position_filter(df, extreme_threshold=0.30)),
        ("版本3: 成交量(1.2x)", lambda: stage1_with_volume_filter(df, min_volume_ratio=1.2)),
        ("版本4: 組合(OR)", lambda: stage1_combined_filters(df)),
        ("版本5: 嚴格組合(AND)", lambda: stage1_strict_combined(df)),
    ]
    
    print("\n" + "="*80)
    print("📊 版本對比")
    print("="*80)
    print(f"\n{'版本':<30} {'候選點':>10} {'檢測到':>10} {'召回率':>10} {'精確度':>10} {'漏掉':>8}")
    print("-"*90)
    
    results = []
    for version_name, version_func in versions:
        candidates = version_func()
        result = evaluate_version(candidates, true_reversals, version_name)
        results.append(result)
        
        print(f"{result['version']:<30} {result['candidates']:>10} {result['detected']:>10} "
              f"{result['recall']:>9.1f}% {result['precision']:>9.1f}% {result['missed']:>8}")
    
    # 推薦
    print("\n" + "="*80)
    print("🎯 推薦")
    print("="*80)
    
    # 找出符合目標的版本（召回率 >= 95% 且候選點 < 10000）
    good_versions = [r for r in results if r['recall'] >= 95 and r['candidates'] < 10000]
    
    if good_versions:
        best = min(good_versions, key=lambda x: x['candidates'])
        print(f"\n✅ 最佳版本: {best['version']}")
        print(f"  召回率: {best['recall']:.1f}%")
        print(f"  候選點: {best['candidates']}")
        print(f"  精確度: {best['precision']:.1f}%")
        print(f"  漏掉: {best['missed']}")
    else:
        print("\n⚠️  沒有版本同時達到目標（召回率 >= 95% 且候選點 < 10000）")
        print("\n最接近的版本：")
        
        # 找召回率最高的
        best_recall = max(results, key=lambda x: x['recall'])
        print(f"\n最高召回率: {best_recall['version']}")
        print(f"  召回率: {best_recall['recall']:.1f}%")
        print(f"  候選點: {best_recall['candidates']}")
        
        # 找候選點最少但召回率還可以的
        decent_recall = [r for r in results if r['recall'] >= 90]
        if decent_recall:
            best_candidates = min(decent_recall, key=lambda x: x['candidates'])
            print(f"\n最少候選點 (召回率 >= 90%): {best_candidates['version']}")
            print(f"  召回率: {best_candidates['recall']:.1f}%")
            print(f"  候選點: {best_candidates['candidates']}")


def main():
    """主函數"""
    compare_versions()


if __name__ == '__main__':
    main()
