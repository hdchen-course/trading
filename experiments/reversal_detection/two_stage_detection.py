#!/usr/bin/env python3
"""
兩階段反轉檢測系統

階段 1: 候選反轉點檢測（高召回率）
- 使用 peak_valley 邏輯，但只看左側數據
- 找出所有「可能」的反轉點

階段 2: 真實反轉點篩選（高精確度）
- 分析反轉點特徵
- 使用規則或 ML 過濾假信號

階段 3: 收益驗證
- 檢查入場後 15-30 分鐘的收益
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


def stage1_find_candidates(df, lookback=20, min_prominence_pct=0.01):
    """
    階段 1: 找出候選反轉點
    
    使用 peak_valley 邏輯，但只看左側數據（實時可用）
    
    對於每個位置 i，檢查：
    - 位置 i-1 或 i-2 是否可能是反轉點
    - 只使用 i 之前的數據
    
    Args:
        df: DataFrame with price data
        lookback: 回看週期
        min_prominence_pct: 最小突出度（百分比）
        
    Returns:
        candidates: 候選反轉點的索引列表
        candidate_types: 每個候選點的類型（'peak' 或 'valley'）
    """
    print("\n" + "="*80)
    print("階段 1: 找出候選反轉點")
    print("="*80)
    
    n = len(df)
    prices = df['close'].values
    
    candidates = []
    candidate_types = []
    candidate_positions = []  # 記錄在哪個位置檢測到的
    
    # 從 lookback 開始檢測
    for i in range(lookback, n):
        # 檢查 i-1 和 i-2 是否可能是反轉點
        for offset in [1, 2]:
            check_idx = i - offset
            
            if check_idx < lookback:
                continue
            
            # 只使用 check_idx 左側的數據
            window_start = check_idx - lookback
            window_end = check_idx + 1  # 包括 check_idx
            
            window_prices = prices[window_start:window_end]
            check_price = prices[check_idx]
            
            # 計算突出度
            prominence = np.std(window_prices) * min_prominence_pct
            
            # 檢查是否是波峰
            if check_price == np.max(window_prices):
                # 計算相對高度
                left_prices = window_prices[:-1]
                if len(left_prices) > 0:
                    height = check_price - np.mean(left_prices)
                    
                    if height >= prominence:
                        # 避免重複
                        if check_idx not in candidates:
                            candidates.append(check_idx)
                            candidate_types.append('peak')
                            candidate_positions.append(i)
            
            # 檢查是否是波谷
            elif check_price == np.min(window_prices):
                # 計算相對深度
                left_prices = window_prices[:-1]
                if len(left_prices) > 0:
                    depth = np.mean(left_prices) - check_price
                    
                    if depth >= prominence:
                        # 避免重複
                        if check_idx not in candidates:
                            candidates.append(check_idx)
                            candidate_types.append('valley')
                            candidate_positions.append(i)
    
    print(f"\n找到 {len(candidates)} 個候選反轉點")
    
    return candidates, candidate_types, candidate_positions


def analyze_candidate_features(df, candidates, candidate_types):
    """
    分析候選反轉點的特徵
    
    提取每個候選點的特徵，用於後續篩選
    """
    print("\n" + "="*80)
    print("分析候選反轉點特徵")
    print("="*80)
    
    features = []
    
    for idx, cand_type in zip(candidates, candidate_types):
        feature = {
            'index': idx,
            'type': cand_type,
            'price': df['close'].iloc[idx]
        }
        
        # 特徵 1: 價格位置（相對於最近 24 小時）
        lookback_24h = 96  # 24h / 15min
        start_idx = max(0, idx - lookback_24h)
        window_prices = df['close'].iloc[start_idx:idx+1]
        
        price_range = window_prices.max() - window_prices.min()
        if price_range > 0:
            feature['price_position'] = (feature['price'] - window_prices.min()) / price_range
        else:
            feature['price_position'] = 0.5
        
        # 特徵 2: 波動度
        if 'volatility' in df.columns:
            feature['volatility'] = df['volatility'].iloc[idx]
        else:
            feature['volatility'] = window_prices.std()
        
        # 特徵 3: 成交量
        if 'volume' in df.columns:
            volume_ma = df['volume'].iloc[start_idx:idx+1].mean()
            feature['volume_ratio'] = df['volume'].iloc[idx] / volume_ma if volume_ma > 0 else 1.0
        else:
            feature['volume_ratio'] = 1.0
        
        # 特徵 4: RSI
        if 'rsi' in df.columns:
            feature['rsi'] = df['rsi'].iloc[idx]
        else:
            feature['rsi'] = 50
        
        # 特徵 5: 趨勢強度（反轉前的趨勢）
        if idx >= 10:
            trend_window = df['close'].iloc[idx-10:idx]
            price_change = (trend_window.iloc[-1] - trend_window.iloc[0]) / trend_window.iloc[0]
            feature['trend_strength'] = abs(price_change)
        else:
            feature['trend_strength'] = 0
        
        # 特徵 6: 左側價格變化
        if idx >= 5:
            left_prices = df['close'].iloc[idx-5:idx]
            feature['left_price_change'] = (feature['price'] - left_prices.mean()) / left_prices.mean()
        else:
            feature['left_price_change'] = 0
        
        features.append(feature)
    
    features_df = pd.DataFrame(features)
    
    print(f"\n提取了 {len(features_df)} 個候選點的特徵")
    print(f"\n特徵統計:")
    print(features_df.describe())
    
    return features_df


def stage2_filter_candidates(features_df, df, true_reversals=None):
    """
    階段 2: 篩選真實反轉點
    
    使用規則過濾候選點
    
    Args:
        features_df: 候選點特徵
        df: 原始數據
        true_reversals: 真實反轉點（用於評估）
        
    Returns:
        filtered_indices: 篩選後的反轉點索引
    """
    print("\n" + "="*80)
    print("階段 2: 篩選真實反轉點")
    print("="*80)
    
    # 規則 1: 極端價格位置
    rule1 = (features_df['price_position'] < 0.2) | (features_df['price_position'] > 0.8)
    
    # 規則 2: 高波動
    volatility_threshold = features_df['volatility'].quantile(0.5)
    rule2 = features_df['volatility'] > volatility_threshold
    
    # 規則 3: 成交量確認
    rule3 = features_df['volume_ratio'] > 1.0
    
    # 規則 4: RSI 極端
    rule4 = (features_df['rsi'] < 35) | (features_df['rsi'] > 65)
    
    # 規則 5: 有明顯趨勢
    rule5 = features_df['trend_strength'] > 0.005  # 0.5%
    
    # 組合規則（至少滿足 3 個）
    rule_count = (
        rule1.astype(int) +
        rule2.astype(int) +
        rule3.astype(int) +
        rule4.astype(int) +
        rule5.astype(int)
    )
    
    filtered = rule_count >= 3
    
    filtered_features = features_df[filtered].copy()
    
    print(f"\n篩選結果:")
    print(f"  候選點: {len(features_df)}")
    print(f"  通過篩選: {len(filtered_features)}")
    print(f"  篩選率: {len(filtered_features)/len(features_df)*100:.1f}%")
    
    # 如果有真實標籤，評估性能
    if true_reversals is not None:
        candidate_indices = features_df['index'].values
        filtered_indices = filtered_features['index'].values
        
        # 候選點的召回率
        candidate_recall = np.sum(np.isin(candidate_indices, true_reversals)) / len(true_reversals) * 100
        
        # 篩選後的精確度和召回率
        filtered_tp = np.sum(np.isin(filtered_indices, true_reversals))
        filtered_precision = filtered_tp / len(filtered_indices) * 100 if len(filtered_indices) > 0 else 0
        filtered_recall = filtered_tp / len(true_reversals) * 100
        
        print(f"\n性能評估:")
        print(f"  候選點召回率: {candidate_recall:.1f}%")
        print(f"  篩選後精確度: {filtered_precision:.1f}%")
        print(f"  篩選後召回率: {filtered_recall:.1f}%")
    
    return filtered_features['index'].values


def stage3_verify_profit(df, filtered_indices, periods=[1, 2, 4]):
    """
    階段 3: 驗證收益
    
    檢查在篩選出的反轉點入場後，15-30-60 分鐘的收益
    
    Args:
        df: 原始數據
        filtered_indices: 篩選後的反轉點索引
        periods: 檢查的週期（K 線數）
        
    Returns:
        profit_analysis: 收益分析結果
    """
    print("\n" + "="*80)
    print("階段 3: 驗證收益")
    print("="*80)
    
    results = {}
    
    for period in periods:
        minutes = period * 15
        profits = []
        
        for idx in filtered_indices:
            if idx + period < len(df):
                entry_price = df['close'].iloc[idx]
                exit_price = df['close'].iloc[idx + period]
                profit = (exit_price - entry_price) / entry_price * 100
                profits.append(profit)
        
        if profits:
            win_rate = sum(1 for p in profits if p > 0) / len(profits) * 100
            
            results[period] = {
                'minutes': minutes,
                'avg_profit': np.mean(profits),
                'median_profit': np.median(profits),
                'win_rate': win_rate,
                'max_profit': np.max(profits),
                'max_loss': np.min(profits),
                'count': len(profits)
            }
            
            print(f"\n{minutes} 分鐘後:")
            print(f"  平均收益: {np.mean(profits):+.2f}%")
            print(f"  中位數收益: {np.median(profits):+.2f}%")
            print(f"  勝率: {win_rate:.1f}%")
            print(f"  最大收益: {np.max(profits):+.2f}%")
            print(f"  最大虧損: {np.min(profits):+.2f}%")
            print(f"  樣本數: {len(profits)}")
    
    return results


def main():
    """主函數"""
    print("="*80)
    print("🔬 兩階段反轉檢測系統")
    print("="*80)
    
    # 載入數據
    df = load_data()
    if df is None:
        print("❌ 找不到數據")
        return
    
    print(f"\n數據概況:")
    print(f"  總數據點: {len(df)}")
    
    true_reversals = None
    if 'reverse' in df.columns:
        true_reversals = np.where(df['reverse'].values)[0]
        print(f"  真實反轉點: {len(true_reversals)} ({len(true_reversals)/len(df)*100:.2f}%)")
    
    # 階段 1: 找出候選反轉點
    candidates, candidate_types, candidate_positions = stage1_find_candidates(
        df, 
        lookback=20,
        min_prominence_pct=0.01
    )
    
    # 分析候選點特徵
    features_df = analyze_candidate_features(df, candidates, candidate_types)
    
    # 階段 2: 篩選真實反轉點
    filtered_indices = stage2_filter_candidates(features_df, df, true_reversals)
    
    # 階段 3: 驗證收益
    profit_results = stage3_verify_profit(df, filtered_indices, periods=[1, 2, 4])
    
    # 總結
    print("\n" + "="*80)
    print("📊 總結")
    print("="*80)
    
    print(f"\n流程統計:")
    print(f"  候選點: {len(candidates)}")
    print(f"  篩選後: {len(filtered_indices)}")
    print(f"  篩選率: {len(filtered_indices)/len(candidates)*100:.1f}%")
    
    if true_reversals is not None:
        final_recall = np.sum(np.isin(filtered_indices, true_reversals)) / len(true_reversals) * 100
        print(f"  最終召回率: {final_recall:.1f}%")
    
    print("\n" + "="*80)
    print("💡 下一步")
    print("="*80)
    
    print("""
1. 調整階段 1 的參數
   - lookback: 影響候選點數量
   - min_prominence_pct: 影響候選點質量

2. 優化階段 2 的規則
   - 嘗試不同的規則組合
   - 調整閾值

3. 加入 Machine Learning
   - 使用候選點特徵訓練分類器
   - 預測哪些候選點是真實反轉點

4. 實時應用
   - 將這個流程應用到實盤
   - 每根新 K 線都運行檢測
    """)


if __name__ == '__main__':
    main()
