#!/usr/bin/env python3
"""
階段 1 整合 Task 2 的發現

Task 2 最佳特徵：
1. 極端價格位置（< 20% 或 > 80%）
2. ATR 二階導數（波動加速）

策略：
- 階段 1：使用這些特徵來過濾候選點
- 目標：保持高召回率，同時減少候選點數量
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np


def load_data():
    """載入訓練數據"""
    path = "training_data/reversal_training_data_2022-01-01_to_2025-09-01_15m.csv"
    
    if os.path.exists(path):
        print(f"✅ 載入數據: {path}")
        return pd.read_csv(path)
    
    return None


def calculate_atr_2nd_derivative(df, period=14):
    """
    計算 ATR 二階導數（波動加速）
    
    這是 Task 2 發現的關鍵特徵
    """
    if 'atr' in df.columns:
        atr = df['atr']
    else:
        # 計算 ATR
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
    
    # 一階導數
    atr_1st = atr.diff()
    
    # 二階導數（加速度）
    atr_2nd = atr_1st.diff()
    
    return atr_2nd


def calculate_price_position(df, lookback_periods=96):
    """
    計算價格位置（相對於 24 小時範圍）
    
    這是 Task 2 發現的另一個關鍵特徵
    """
    high_24h = df['close'].rolling(window=lookback_periods, min_periods=1).max()
    low_24h = df['close'].rolling(window=lookback_periods, min_periods=1).min()
    
    range_24h = high_24h - low_24h
    range_24h = range_24h.replace(0, 1e-10)
    
    position = (df['close'] - low_24h) / range_24h
    
    return position


def stage1_with_task2_features(df, 
                                lookback=20,
                                extreme_threshold=0.2,
                                atr_accel_threshold=0.5):
    """
    階段 1：整合 Task 2 特徵
    
    檢測邏輯：
    1. 找出局部極值（波峰/波谷）
    2. 檢查是否在極端價格位置
    3. 檢查 ATR 是否加速
    4. 只保留同時滿足的候選點
    
    Args:
        df: DataFrame
        lookback: 局部極值檢測的回看期
        extreme_threshold: 極端位置閾值
        atr_accel_threshold: ATR 加速閾值
        
    Returns:
        candidates: 候選點索引
    """
    print("\n" + "="*80)
    print("階段 1：整合 Task 2 特徵")
    print("="*80)
    
    n = len(df)
    prices = df['close'].values
    
    # 計算 Task 2 特徵
    print("\n計算 Task 2 特徵...")
    price_position = calculate_price_position(df, lookback_periods=96)
    atr_2nd = calculate_atr_2nd_derivative(df)
    
    # 極端位置
    is_extreme = (price_position < extreme_threshold) | (price_position > (1 - extreme_threshold))
    
    # ATR 加速
    is_accelerating = atr_2nd > atr_accel_threshold
    
    print(f"  極端位置點數: {is_extreme.sum()}")
    print(f"  ATR 加速點數: {is_accelerating.sum()}")
    print(f"  同時滿足: {(is_extreme & is_accelerating).sum()}")
    
    # 檢測局部極值
    print(f"\n檢測局部極值...")
    candidates = []
    
    for i in range(lookback, n):
        # 檢查 i-1 和 i-2
        for offset in [1, 2]:
            check_idx = i - offset
            
            if check_idx < lookback:
                continue
            
            # Task 2 特徵過濾
            if not is_extreme.iloc[check_idx]:
                continue
            
            if not is_accelerating.iloc[check_idx]:
                continue
            
            # 局部極值檢測
            window_start = check_idx - lookback
            window_end = check_idx + 1
            window_prices = prices[window_start:window_end]
            check_price = prices[check_idx]
            
            # 波峰或波谷
            if check_price == np.max(window_prices) or check_price == np.min(window_prices):
                if check_idx not in candidates:
                    candidates.append(check_idx)
    
    print(f"\n找到 {len(candidates)} 個候選點")
    
    return np.array(candidates)


def stage1_relaxed_task2(df,
                         lookback=20,
                         extreme_threshold=0.25,  # 放寬到 25%
                         atr_accel_threshold=0.3):  # 降低閾值
    """
    放寬版本：提高召回率
    
    使用更寬鬆的閾值來捕捉更多候選點
    """
    return stage1_with_task2_features(df, lookback, extreme_threshold, atr_accel_threshold)


def stage1_strict_task2(df,
                        lookback=20,
                        extreme_threshold=0.15,  # 更嚴格
                        atr_accel_threshold=0.7):  # 更高閾值
    """
    嚴格版本：減少候選點
    
    使用更嚴格的閾值來減少假陽性
    """
    return stage1_with_task2_features(df, lookback, extreme_threshold, atr_accel_threshold)


def evaluate_version(candidates, true_reversals, version_name):
    """評估版本性能"""
    recall = np.sum(np.isin(candidates, true_reversals)) / len(true_reversals) * 100
    
    return {
        'version': version_name,
        'candidates': len(candidates),
        'recall': recall,
        'missed': len(true_reversals) - np.sum(np.isin(candidates, true_reversals))
    }


def compare_versions():
    """對比不同版本"""
    print("="*80)
    print("🔬 Task 2 特徵整合對比")
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
        ("標準版本 (0.2, 0.5)", lambda: stage1_with_task2_features(df)),
        ("放寬版本 (0.25, 0.3)", lambda: stage1_relaxed_task2(df)),
        ("嚴格版本 (0.15, 0.7)", lambda: stage1_strict_task2(df)),
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
    
    # 推薦
    print("\n" + "="*80)
    print("🎯 推薦")
    print("="*80)
    
    print("""
Task 2 特徵的優勢：
1. 極端位置：反轉通常發生在價格極值
2. ATR 加速：波動增加表示動量變化

整合效果：
- 大幅減少候選點數量
- 保持合理的召回率

建議：
1. 使用「放寬版本」作為階段 1
   - 保持較高召回率
   - 候選點數量可控
   
2. 階段 2 使用更強的過濾
   - 成交量確認
   - RSI 確認
   - 趨勢確認
   - 或使用 ML

3. 參數可以根據實際表現微調
   - extreme_threshold: 0.2-0.3
   - atr_accel_threshold: 0.3-0.7
    """)


def main():
    """主函數"""
    compare_versions()


if __name__ == '__main__':
    main()
