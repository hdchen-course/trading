#!/usr/bin/env python3
"""
優化階段 1：候選點檢測

目標：
1. 召回率 > 95%（不漏掉真實反轉點）
2. 候選點數量 < 10,000（減少假陽性）

策略：
- 調整 lookback 參數
- 調整 prominence 閾值
- 增加額外的過濾條件
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def load_data():
    """載入訓練數據"""
    path = "training_data/reversal_training_data_2022-01-01_to_2025-09-01_15m.csv"
    
    if os.path.exists(path):
        print(f"✅ 載入數據: {path}")
        return pd.read_csv(path)
    
    return None


def stage1_v1_baseline(df, lookback=20, min_prominence_pct=0.01):
    """
    基線版本（當前實現）
    """
    n = len(df)
    prices = df['close'].values
    candidates = []
    
    for i in range(lookback, n):
        for offset in [1, 2]:
            check_idx = i - offset
            
            if check_idx < lookback:
                continue
            
            window_start = check_idx - lookback
            window_end = check_idx + 1
            window_prices = prices[window_start:window_end]
            check_price = prices[check_idx]
            
            prominence = np.std(window_prices) * min_prominence_pct
            
            # 波峰
            if check_price == np.max(window_prices):
                left_prices = window_prices[:-1]
                if len(left_prices) > 0:
                    height = check_price - np.mean(left_prices)
                    if height >= prominence and check_idx not in candidates:
                        candidates.append(check_idx)
            
            # 波谷
            elif check_price == np.min(window_prices):
                left_prices = window_prices[:-1]
                if len(left_prices) > 0:
                    depth = np.mean(left_prices) - check_price
                    if depth >= prominence and check_idx not in candidates:
                        candidates.append(check_idx)
    
    return np.array(candidates)


def stage1_v2_stricter_prominence(df, lookback=20, min_prominence_pct=0.02):
    """
    版本 2：提高 prominence 閾值
    
    更高的 prominence = 更顯著的極值 = 更少的候選點
    """
    return stage1_v1_baseline(df, lookback, min_prominence_pct)


def stage1_v3_longer_lookback(df, lookback=30, min_prominence_pct=0.01):
    """
    版本 3：增加 lookback 週期
    
    更長的 lookback = 更大的視野 = 更少的局部極值
    """
    return stage1_v1_baseline(df, lookback, min_prominence_pct)


def stage1_v4_min_price_change(df, lookback=20, min_prominence_pct=0.01, min_price_change=0.005):
    """
    版本 4：增加最小價格變化要求
    
    極值必須與周圍價格有至少 0.5% 的差異
    """
    n = len(df)
    prices = df['close'].values
    candidates = []
    
    for i in range(lookback, n):
        for offset in [1, 2]:
            check_idx = i - offset
            
            if check_idx < lookback:
                continue
            
            window_start = check_idx - lookback
            window_end = check_idx + 1
            window_prices = prices[window_start:window_end]
            check_price = prices[check_idx]
            
            prominence = np.std(window_prices) * min_prominence_pct
            
            # 波峰
            if check_price == np.max(window_prices):
                left_prices = window_prices[:-1]
                if len(left_prices) > 0:
                    height = check_price - np.mean(left_prices)
                    price_change = height / check_price
                    
                    if height >= prominence and price_change >= min_price_change:
                        if check_idx not in candidates:
                            candidates.append(check_idx)
            
            # 波谷
            elif check_price == np.min(window_prices):
                left_prices = window_prices[:-1]
                if len(left_prices) > 0:
                    depth = np.mean(left_prices) - check_price
                    price_change = depth / check_price
                    
                    if depth >= prominence and price_change >= min_price_change:
                        if check_idx not in candidates:
                            candidates.append(check_idx)
    
    return np.array(candidates)


def stage1_v5_volume_filter(df, lookback=20, min_prominence_pct=0.01, volume_threshold=0.8):
    """
    版本 5：增加成交量過濾
    
    反轉點通常伴隨成交量變化
    過濾掉低成交量的極值
    """
    n = len(df)
    prices = df['close'].values
    candidates = []
    
    # 計算成交量均線
    if 'volume' not in df.columns:
        return stage1_v1_baseline(df, lookback, min_prominence_pct)
    
    volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()
    
    for i in range(lookback, n):
        for offset in [1, 2]:
            check_idx = i - offset
            
            if check_idx < lookback:
                continue
            
            # 成交量過濾
            volume_ratio = df['volume'].iloc[check_idx] / volume_ma.iloc[check_idx]
            if volume_ratio < volume_threshold:
                continue
            
            window_start = check_idx - lookback
            window_end = check_idx + 1
            window_prices = prices[window_start:window_end]
            check_price = prices[check_idx]
            
            prominence = np.std(window_prices) * min_prominence_pct
            
            # 波峰
            if check_price == np.max(window_prices):
                left_prices = window_prices[:-1]
                if len(left_prices) > 0:
                    height = check_price - np.mean(left_prices)
                    if height >= prominence and check_idx not in candidates:
                        candidates.append(check_idx)
            
            # 波谷
            elif check_price == np.min(window_prices):
                left_prices = window_prices[:-1]
                if len(left_prices) > 0:
                    depth = np.mean(left_prices) - check_price
                    if depth >= prominence and check_idx not in candidates:
                        candidates.append(check_idx)
    
    return np.array(candidates)


def stage1_v6_combined(df, lookback=25, min_prominence_pct=0.015, 
                       min_price_change=0.003, volume_threshold=0.9):
    """
    版本 6：組合優化
    
    結合多個改進：
    - 稍長的 lookback (25)
    - 稍高的 prominence (0.015)
    - 最小價格變化 (0.3%)
    - 成交量過濾 (0.9)
    """
    n = len(df)
    prices = df['close'].values
    candidates = []
    
    # 成交量均線
    has_volume = 'volume' in df.columns
    if has_volume:
        volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()
    
    for i in range(lookback, n):
        for offset in [1, 2]:
            check_idx = i - offset
            
            if check_idx < lookback:
                continue
            
            # 成交量過濾
            if has_volume:
                volume_ratio = df['volume'].iloc[check_idx] / volume_ma.iloc[check_idx]
                if volume_ratio < volume_threshold:
                    continue
            
            window_start = check_idx - lookback
            window_end = check_idx + 1
            window_prices = prices[window_start:window_end]
            check_price = prices[check_idx]
            
            prominence = np.std(window_prices) * min_prominence_pct
            
            # 波峰
            if check_price == np.max(window_prices):
                left_prices = window_prices[:-1]
                if len(left_prices) > 0:
                    height = check_price - np.mean(left_prices)
                    price_change = height / check_price
                    
                    if height >= prominence and price_change >= min_price_change:
                        if check_idx not in candidates:
                            candidates.append(check_idx)
            
            # 波谷
            elif check_price == np.min(window_prices):
                left_prices = window_prices[:-1]
                if len(left_prices) > 0:
                    depth = np.mean(left_prices) - check_price
                    price_change = depth / check_price
                    
                    if depth >= prominence and price_change >= min_price_change:
                        if check_idx not in candidates:
                            candidates.append(check_idx)
    
    return np.array(candidates)


def evaluate_version(candidates, true_reversals, version_name):
    """評估版本性能"""
    recall = np.sum(np.isin(candidates, true_reversals)) / len(true_reversals) * 100
    
    return {
        'version': version_name,
        'candidates': len(candidates),
        'recall': recall,
        'missed': len(true_reversals) - np.sum(np.isin(candidates, true_reversals))
    }


def compare_all_versions():
    """對比所有版本"""
    print("="*80)
    print("🔬 階段 1 優化對比")
    print("="*80)
    
    # 載入數據
    df = load_data()
    if df is None:
        return
    
    true_reversals = np.where(df['reverse'].values)[0]
    
    print(f"\n數據概況:")
    print(f"  總數據點: {len(df)}")
    print(f"  真實反轉點: {len(true_reversals)} ({len(true_reversals)/len(df)*100:.2f}%)")
    
    # 測試所有版本
    versions = [
        ("V1: 基線", lambda: stage1_v1_baseline(df)),
        ("V2: 更高 prominence", lambda: stage1_v2_stricter_prominence(df)),
        ("V3: 更長 lookback", lambda: stage1_v3_longer_lookback(df)),
        ("V4: 最小價格變化", lambda: stage1_v4_min_price_change(df)),
        ("V5: 成交量過濾", lambda: stage1_v5_volume_filter(df)),
        ("V6: 組合優化", lambda: stage1_v6_combined(df)),
    ]
    
    print("\n" + "="*80)
    print("📊 版本對比")
    print("="*80)
    print(f"\n{'版本':<30} {'候選點數':>12} {'召回率':>10} {'漏掉':>8}")
    print("-"*80)
    
    results = []
    for version_name, version_func in versions:
        candidates = version_func()
        result = evaluate_version(candidates, true_reversals, version_name)
        results.append(result)
        
        print(f"{result['version']:<30} {result['candidates']:>12} "
              f"{result['recall']:>9.1f}% {result['missed']:>8}")
    
    # 找出最佳版本
    print("\n" + "="*80)
    print("🏆 推薦版本")
    print("="*80)
    
    # 目標：召回率 > 95%，候選點 < 10,000
    good_versions = [r for r in results if r['recall'] >= 95 and r['candidates'] < 10000]
    
    if good_versions:
        best = min(good_versions, key=lambda x: x['candidates'])
        print(f"\n✅ 推薦：{best['version']}")
        print(f"  候選點: {best['candidates']}")
        print(f"  召回率: {best['recall']:.1f}%")
        print(f"  漏掉: {best['missed']}")
    else:
        # 找召回率最高的
        best_recall = max(results, key=lambda x: x['recall'])
        # 找候選點最少的（召回率 > 90%）
        good_recall = [r for r in results if r['recall'] >= 90]
        if good_recall:
            best_candidates = min(good_recall, key=lambda x: x['candidates'])
            
            print(f"\n⚠️  沒有版本同時滿足兩個目標")
            print(f"\n最高召回率：{best_recall['version']}")
            print(f"  候選點: {best_recall['candidates']}")
            print(f"  召回率: {best_recall['recall']:.1f}%")
            
            print(f"\n最少候選點（召回率 > 90%）：{best_candidates['version']}")
            print(f"  候選點: {best_candidates['candidates']}")
            print(f"  召回率: {best_candidates['recall']:.1f}%")
    
    print("\n" + "="*80)
    print("💡 調優建議")
    print("="*80)
    
    print("""
參數調優方向：

1. lookback（回看週期）:
   - 增加 → 候選點減少，但可能降低召回率
   - 建議範圍：20-30

2. min_prominence_pct（突出度）:
   - 增加 → 候選點減少，召回率可能降低
   - 建議範圍：0.01-0.02

3. min_price_change（最小價格變化）:
   - 增加 → 候選點減少，過濾掉小幅波動
   - 建議範圍：0.003-0.01 (0.3%-1.0%)

4. volume_threshold（成交量閾值）:
   - 增加 → 候選點減少，但可能漏掉低量反轉
   - 建議範圍：0.8-1.2

下一步：
1. 微調參數找到最佳組合
2. 分析被漏掉的反轉點特徵
3. 針對性地改進檢測邏輯
    """)


def main():
    """主函數"""
    compare_all_versions()


if __name__ == '__main__':
    main()
