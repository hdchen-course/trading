#!/usr/bin/env python3
"""
改進的階段1檢測器

基於左側現象分析的發現：
1. 我們要找「前1-2根K線是反轉點」的時機
2. 94.6% 的反轉點在後續K線確認方向
3. 71.3% 有連續確認
4. 使用 peak_valley 原理 + 確認規則

策略：
- 使用 scipy.find_peaks 找局部極值
- 在當前K線檢查「前1-2根是否為反轉點」
- 用價格確認和連續確認來過濾
- 目標：召回率 > 95%，候選點 < 10,000
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


def find_local_extremes(prices, distance=20, prominence=None):
    """
    使用 scipy.find_peaks 找局部極值
    
    這是 peak_valley 的核心原理
    """
    if prominence is None:
        # 動態計算 prominence
        window_size = min(500, len(prices) // 4)
        if window_size < 100:
            window_size = len(prices)
        
        rolling_std = pd.Series(prices).rolling(window=window_size, min_periods=1).std()
        prominence = rolling_std.median() * 0.5
    
    # 找波谷
    valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)
    
    # 找波峰
    peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
    
    # 合併並排序
    extremes = np.sort(np.concatenate([peaks, valleys]))
    
    return extremes, peaks, valleys


def check_price_confirmation(df, rev_idx, curr_idx, is_peak, threshold=0.01):
    """
    檢查價格確認
    
    Args:
        df: DataFrame
        rev_idx: 反轉點索引
        curr_idx: 當前K線索引
        is_peak: 是否為波峰
        threshold: 確認閾值（1% = 0.01）
    """
    rev_price = df['close'].iloc[rev_idx]
    curr_price = df['close'].iloc[curr_idx]
    
    if is_peak:
        # 波峰後應該下跌
        return curr_price < rev_price * (1 - threshold)
    else:
        # 波谷後應該上漲
        return curr_price > rev_price * (1 + threshold)


def check_continuous_confirmation(df, rev_idx, curr_idx, is_peak):
    """
    檢查連續確認
    
    要求：反轉點後的K線連續朝反轉方向移動
    """
    if curr_idx != rev_idx + 2:
        return False
    
    rev_price = df['close'].iloc[rev_idx]
    mid_price = df['close'].iloc[rev_idx + 1]
    curr_price = df['close'].iloc[curr_idx]
    
    if is_peak:
        # 波峰後應該連續下跌
        return (mid_price < rev_price) and (curr_price < mid_price)
    else:
        # 波谷後應該連續上漲
        return (mid_price > rev_price) and (curr_price > mid_price)


def improved_stage1_v1(df, lookback=20, price_threshold=0.01):
    """
    版本1：基礎版本
    
    策略：
    - 找局部極值
    - 在當前K線檢查前1-2根是否為反轉點
    - 用1%價格確認過濾
    """
    print("\n" + "="*80)
    print("版本1：基礎版本（1%價格確認）")
    print("="*80)
    
    prices = df['close'].values
    n = len(df)
    
    # 找局部極值
    extremes, peaks, valleys = find_local_extremes(prices, distance=lookback)
    
    print(f"找到 {len(extremes)} 個局部極值")
    print(f"  波峰: {len(peaks)}")
    print(f"  波谷: {len(valleys)}")
    
    # 檢測候選點
    candidates = []
    
    for i in range(lookback, n):
        # 檢查前1-2根K線
        for offset in [1, 2]:
            check_idx = i - offset
            
            if check_idx not in extremes:
                continue
            
            # 判斷是波峰還是波谷
            is_peak = check_idx in peaks
            
            # 價格確認
            if check_price_confirmation(df, check_idx, i, is_peak, price_threshold):
                if i not in candidates:
                    candidates.append(i)
    
    return np.array(candidates)


def improved_stage1_v2(df, lookback=20, price_threshold=0.005):
    """
    版本2：放寬版本
    
    策略：
    - 降低價格確認閾值到0.5%
    - 提高召回率
    """
    print("\n" + "="*80)
    print("版本2：放寬版本（0.5%價格確認）")
    print("="*80)
    
    return improved_stage1_v1(df, lookback, price_threshold)


def improved_stage1_v3(df, lookback=20):
    """
    版本3：連續確認版本
    
    策略：
    - 使用連續確認（71.3%成功率）
    - 更嚴格但更準確
    """
    print("\n" + "="*80)
    print("版本3：連續確認版本")
    print("="*80)
    
    prices = df['close'].values
    n = len(df)
    
    # 找局部極值
    extremes, peaks, valleys = find_local_extremes(prices, distance=lookback)
    
    print(f"找到 {len(extremes)} 個局部極值")
    
    # 檢測候選點
    candidates = []
    
    for i in range(lookback, n):
        # 只檢查前2根K線（連續確認需要2根）
        check_idx = i - 2
        
        if check_idx not in extremes:
            continue
        
        # 判斷是波峰還是波谷
        is_peak = check_idx in peaks
        
        # 連續確認
        if check_continuous_confirmation(df, check_idx, i, is_peak):
            if i not in candidates:
                candidates.append(i)
    
    return np.array(candidates)


def improved_stage1_v4(df, lookback=20):
    """
    版本4：組合版本
    
    策略：
    - 價格確認 OR 連續確認
    - 平衡召回率和精確度
    """
    print("\n" + "="*80)
    print("版本4：組合版本（價格確認 OR 連續確認）")
    print("="*80)
    
    prices = df['close'].values
    n = len(df)
    
    # 找局部極值
    extremes, peaks, valleys = find_local_extremes(prices, distance=lookback)
    
    print(f"找到 {len(extremes)} 個局部極值")
    
    # 檢測候選點
    candidates = []
    
    for i in range(lookback, n):
        # 檢查前1-2根K線
        for offset in [1, 2]:
            check_idx = i - offset
            
            if check_idx not in extremes:
                continue
            
            # 判斷是波峰還是波谷
            is_peak = check_idx in peaks
            
            # 價格確認 OR 連續確認
            price_confirmed = check_price_confirmation(df, check_idx, i, is_peak, 0.005)
            continuous_confirmed = check_continuous_confirmation(df, check_idx, i, is_peak) if offset == 2 else False
            
            if price_confirmed or continuous_confirmed:
                if i not in candidates:
                    candidates.append(i)
    
    return np.array(candidates)


def improved_stage1_v5(df, lookback=20):
    """
    版本5：無過濾版本（基準線）
    
    策略：
    - 只找局部極值，不做任何確認
    - 用來對比其他版本的效果
    """
    print("\n" + "="*80)
    print("版本5：無過濾版本（基準線）")
    print("="*80)
    
    prices = df['close'].values
    n = len(df)
    
    # 找局部極值
    extremes, peaks, valleys = find_local_extremes(prices, distance=lookback)
    
    print(f"找到 {len(extremes)} 個局部極值")
    
    # 檢測候選點：在極值後1-2根K線
    candidates = []
    
    for i in range(lookback, n):
        for offset in [1, 2]:
            check_idx = i - offset
            
            if check_idx in extremes:
                if i not in candidates:
                    candidates.append(i)
    
    return np.array(candidates)


def evaluate_version(candidates, true_reversals, df, version_name):
    """
    評估版本性能
    
    正確理解：
    - 候選點 = 當前K線（檢測時機）
    - 召回率 = 候選點中，前1-2根K線是真實反轉點的比例
    - 召回率 = (前1-2根是真實反轉點的候選點數) / (候選點總數)
    """
    # 統計有多少候選點的前1-2根K線是真實反轉點
    valid_candidates = 0
    
    for cand_idx in candidates:
        # 檢查這個候選點的前1-2根K線
        is_valid = False
        for offset in [1, 2]:
            check_idx = cand_idx - offset
            if check_idx in true_reversals:
                is_valid = True
                break
        
        if is_valid:
            valid_candidates += 1
    
    # 計算召回率（命中率）
    recall = valid_candidates / len(candidates) * 100 if len(candidates) > 0 else 0
    
    # 額外統計：有多少真實反轉點被檢測到
    detected_reversals = set()
    for cand_idx in candidates:
        for offset in [1, 2]:
            check_idx = cand_idx - offset
            if check_idx in true_reversals:
                detected_reversals.add(check_idx)
    
    coverage = len(detected_reversals) / len(true_reversals) * 100 if len(true_reversals) > 0 else 0
    
    return {
        'version': version_name,
        'candidates': len(candidates),
        'valid_candidates': valid_candidates,
        'recall': recall,  # 前1-2根是真實反轉點的比例
        'coverage': coverage,  # 覆蓋了多少真實反轉點
        'detected_reversals': len(detected_reversals)
    }


def compare_versions():
    """對比不同版本"""
    print("="*80)
    print("🔬 改進的階段1檢測器對比")
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
        ("版本1: 1%價格確認", lambda: improved_stage1_v1(df)),
        ("版本2: 0.5%價格確認", lambda: improved_stage1_v2(df)),
        ("版本3: 連續確認", lambda: improved_stage1_v3(df)),
        ("版本4: 組合版本", lambda: improved_stage1_v4(df)),
        ("版本5: 無過濾（基準）", lambda: improved_stage1_v5(df)),
    ]
    
    print("\n" + "="*80)
    print("📊 版本對比")
    print("="*80)
    print(f"\n{'版本':<30} {'候選點':>10} {'有效候選':>10} {'召回率':>10} {'覆蓋率':>10}")
    print("-"*90)
    
    results = []
    for version_name, version_func in versions:
        candidates = version_func()
        result = evaluate_version(candidates, true_reversals, df, version_name)
        results.append(result)
        
        print(f"{result['version']:<30} {result['candidates']:>10} {result['valid_candidates']:>10} "
              f"{result['recall']:>9.1f}% {result['coverage']:>9.1f}%")
    
    # 推薦
    print("\n" + "="*80)
    print("🎯 推薦")
    print("="*80)
    
    # 找出最佳版本
    best_recall = max(results, key=lambda x: x['recall'])
    best_coverage = max(results, key=lambda x: x['coverage'])
    best_balance = max([r for r in results if r['recall'] >= 70 and r['candidates'] < 10000], 
                       key=lambda x: x['coverage'], 
                       default=None)
    
    print(f"\n最高召回率（命中率）: {best_recall['version']}")
    print(f"  召回率: {best_recall['recall']:.1f}% (前1-2根是真實反轉點)")
    print(f"  候選點: {best_recall['candidates']}")
    print(f"  覆蓋率: {best_recall['coverage']:.1f}% (覆蓋多少真實反轉點)")
    
    print(f"\n最高覆蓋率: {best_coverage['version']}")
    print(f"  覆蓋率: {best_coverage['coverage']:.1f}%")
    print(f"  召回率: {best_coverage['recall']:.1f}%")
    print(f"  候選點: {best_coverage['candidates']}")
    
    if best_balance:
        print(f"\n最佳平衡 (召回率 >= 70%, 候選點 < 10,000): {best_balance['version']}")
        print(f"  召回率: {best_balance['recall']:.1f}%")
        print(f"  覆蓋率: {best_balance['coverage']:.1f}%")
        print(f"  候選點: {best_balance['candidates']}")
    
    print("""
關鍵發現：
1. 召回率 = 候選點中，前1-2根K線是真實反轉點的比例
2. 覆蓋率 = 有多少真實反轉點在其後1-2根K線被檢測到
3. 目標：召回率 > 70%，候選點 < 10,000

建議：
- 如果召回率 >= 70% 且候選點 < 10,000：使用該版本
- 否則：調整參數或在階段2用更強的過濾
    """)


def main():
    """主函數"""
    compare_versions()


if __name__ == '__main__':
    main()
