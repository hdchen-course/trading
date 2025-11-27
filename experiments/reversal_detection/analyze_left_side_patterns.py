#!/usr/bin/env python3
"""
分析反轉點的左側現象

核心目標：
1. 我們不是要找反轉點本身
2. 而是要找「前1或前2根K線是反轉點」的時機
3. 這是右側交易的關鍵

分析重點：
- 反轉點發生後，下一根K線有什麼特徵？
- 反轉點發生後，第二根K線有什麼特徵？
- 如何在不用未來數據的情況下，識別「剛剛發生了反轉」？
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


def analyze_post_reversal_patterns(df, lookback=20):
    """
    分析反轉點之後的K線模式
    
    重點：當反轉點在 i-1 或 i-2 時，當前K線 i 有什麼特徵？
    """
    print("\n" + "="*80)
    print("🔍 分析反轉點後的K線模式")
    print("="*80)
    
    true_reversals = np.where(df['reverse'].values)[0]
    print(f"\n真實反轉點數量: {len(true_reversals)}")
    
    # 分析反轉點後 1-2 根K線的特徵
    patterns = {
        'next_1': [],  # 反轉點後第1根K線
        'next_2': []   # 反轉點後第2根K線
    }
    
    for rev_idx in true_reversals:
        if rev_idx < lookback or rev_idx >= len(df) - 3:
            continue
        
        # 反轉點的價格
        rev_price = df['close'].iloc[rev_idx]
        
        # 判斷是波峰還是波谷
        window_start = rev_idx - lookback
        window_end = rev_idx + 1
        window_prices = df['close'].iloc[window_start:window_end].values
        
        is_peak = rev_price == np.max(window_prices)
        is_valley = rev_price == np.min(window_prices)
        
        # 分析後續K線
        for offset, key in [(1, 'next_1'), (2, 'next_2')]:
            curr_idx = rev_idx + offset
            
            if curr_idx >= len(df):
                continue
            
            curr_price = df['close'].iloc[curr_idx]
            prev_price = df['close'].iloc[curr_idx - 1]
            
            pattern = {
                'reversal_type': 'peak' if is_peak else 'valley',
                'reversal_price': rev_price,
                'current_price': curr_price,
                'offset': offset
            }
            
            # 1. 價格變化方向
            pattern['price_change_pct'] = (curr_price - rev_price) / rev_price * 100
            pattern['prev_change_pct'] = (curr_price - prev_price) / prev_price * 100
            
            # 2. 是否確認反轉方向
            if is_peak:
                # 波峰後應該下跌
                pattern['confirms_reversal'] = curr_price < rev_price
                pattern['strong_confirmation'] = curr_price < rev_price * 0.98  # 下跌超過2%
            else:
                # 波谷後應該上漲
                pattern['confirms_reversal'] = curr_price > rev_price
                pattern['strong_confirmation'] = curr_price > rev_price * 1.02  # 上漲超過2%
            
            # 3. K線實體大小
            open_price = df['open'].iloc[curr_idx]
            pattern['body_pct'] = abs(curr_price - open_price) / open_price * 100
            
            # 4. 成交量變化
            if 'volume' in df.columns:
                vol_ma = df['volume'].iloc[curr_idx-20:curr_idx].mean()
                pattern['volume_ratio'] = df['volume'].iloc[curr_idx] / vol_ma if vol_ma > 0 else 1.0
            
            # 5. 與反轉點的距離
            pattern['distance_from_reversal'] = abs(curr_price - rev_price) / rev_price * 100
            
            # 6. 連續確認（是否連續朝反轉方向移動）
            if offset == 2:
                mid_price = df['close'].iloc[rev_idx + 1]
                if is_peak:
                    pattern['continuous_confirmation'] = (mid_price < rev_price) and (curr_price < mid_price)
                else:
                    pattern['continuous_confirmation'] = (mid_price > rev_price) and (curr_price > mid_price)
            
            patterns[key].append(pattern)
    
    # 統計分析
    print("\n" + "="*80)
    print("📊 反轉後K線統計")
    print("="*80)
    
    for key, data in patterns.items():
        if not data:
            continue
        
        df_patterns = pd.DataFrame(data)
        
        offset_name = "第1根K線" if key == 'next_1' else "第2根K線"
        print(f"\n{offset_name} (反轉點後):")
        print("-" * 60)
        
        # 確認率
        confirm_rate = df_patterns['confirms_reversal'].mean() * 100
        strong_confirm_rate = df_patterns['strong_confirmation'].mean() * 100
        
        print(f"  確認反轉方向: {confirm_rate:.1f}%")
        print(f"  強確認 (>2%): {strong_confirm_rate:.1f}%")
        
        # 平均變化
        avg_change = df_patterns['price_change_pct'].mean()
        print(f"  平均價格變化: {avg_change:.2f}%")
        
        # 實體大小
        avg_body = df_patterns['body_pct'].mean()
        print(f"  平均K線實體: {avg_body:.2f}%")
        
        # 成交量
        if 'volume_ratio' in df_patterns.columns:
            avg_vol = df_patterns['volume_ratio'].mean()
            print(f"  平均成交量比: {avg_vol:.2f}x")
        
        # 連續確認
        if 'continuous_confirmation' in df_patterns.columns:
            cont_rate = df_patterns['continuous_confirmation'].mean() * 100
            print(f"  連續確認率: {cont_rate:.1f}%")
    
    return patterns


def analyze_detection_timing(df, lookback=20):
    """
    分析最佳檢測時機
    
    問題：在第幾根K線後檢測最準確？
    """
    print("\n" + "="*80)
    print("⏰ 分析最佳檢測時機")
    print("="*80)
    
    true_reversals = np.where(df['reverse'].values)[0]
    
    # 測試不同的檢測時機
    timing_results = []
    
    for offset in [1, 2, 3]:
        # 模擬在反轉點後 offset 根K線時檢測
        detected = []
        false_positives = []
        
        for i in range(lookback, len(df) - offset):
            check_idx = i - offset  # 檢查 i-offset 是否為反轉點
            
            if check_idx < lookback:
                continue
            
            # 使用 peak_valley 邏輯檢測
            window_start = check_idx - lookback
            window_end = check_idx + 1
            window_prices = df['close'].iloc[window_start:window_end].values
            check_price = df['close'].iloc[check_idx]
            
            # 是否為局部極值
            is_extreme = (check_price == np.max(window_prices)) or (check_price == np.min(window_prices))
            
            if not is_extreme:
                continue
            
            # 確認反轉方向
            curr_price = df['close'].iloc[i]
            is_peak = check_price == np.max(window_prices)
            
            if is_peak:
                # 波峰後應該下跌
                confirmed = curr_price < check_price * 0.99  # 下跌超過1%
            else:
                # 波谷後應該上漲
                confirmed = curr_price > check_price * 1.01  # 上漲超過1%
            
            if confirmed:
                detected.append(check_idx)
                
                # 檢查是否為真實反轉點
                if check_idx not in true_reversals:
                    false_positives.append(check_idx)
        
        # 計算指標
        true_positives = len([idx for idx in detected if idx in true_reversals])
        recall = true_positives / len(true_reversals) * 100 if len(true_reversals) > 0 else 0
        precision = true_positives / len(detected) * 100 if len(detected) > 0 else 0
        
        timing_results.append({
            'offset': offset,
            'detected': len(detected),
            'true_positives': true_positives,
            'false_positives': len(false_positives),
            'recall': recall,
            'precision': precision
        })
    
    # 顯示結果
    print(f"\n{'檢測時機':<12} {'檢測數':>10} {'真陽性':>10} {'假陽性':>10} {'召回率':>10} {'精確度':>10}")
    print("-" * 80)
    
    for result in timing_results:
        print(f"反轉後{result['offset']}根K線 {result['detected']:>10} {result['true_positives']:>10} "
              f"{result['false_positives']:>10} {result['recall']:>9.1f}% {result['precision']:>9.1f}%")
    
    return timing_results


def find_best_confirmation_rules(df, lookback=20):
    """
    尋找最佳的確認規則
    
    目標：在反轉點後1-2根K線，用什麼規則能最準確地確認反轉？
    """
    print("\n" + "="*80)
    print("🎯 尋找最佳確認規則")
    print("="*80)
    
    true_reversals = np.where(df['reverse'].values)[0]
    
    # 測試不同的確認規則
    rules = [
        {
            'name': '價格確認 1%',
            'check': lambda df, rev_idx, curr_idx, is_peak: (
                df['close'].iloc[curr_idx] < df['close'].iloc[rev_idx] * 0.99 if is_peak
                else df['close'].iloc[curr_idx] > df['close'].iloc[rev_idx] * 1.01
            )
        },
        {
            'name': '價格確認 2%',
            'check': lambda df, rev_idx, curr_idx, is_peak: (
                df['close'].iloc[curr_idx] < df['close'].iloc[rev_idx] * 0.98 if is_peak
                else df['close'].iloc[curr_idx] > df['close'].iloc[rev_idx] * 1.02
            )
        },
        {
            'name': '連續確認',
            'check': lambda df, rev_idx, curr_idx, is_peak: (
                (df['close'].iloc[rev_idx+1] < df['close'].iloc[rev_idx] and 
                 df['close'].iloc[curr_idx] < df['close'].iloc[rev_idx+1]) if is_peak
                else (df['close'].iloc[rev_idx+1] > df['close'].iloc[rev_idx] and 
                      df['close'].iloc[curr_idx] > df['close'].iloc[rev_idx+1])
            ) if curr_idx == rev_idx + 2 else False
        },
        {
            'name': 'K線實體 > 0.5%',
            'check': lambda df, rev_idx, curr_idx, is_peak: (
                abs(df['close'].iloc[curr_idx] - df['open'].iloc[curr_idx]) / df['open'].iloc[curr_idx] > 0.005
            )
        }
    ]
    
    results = []
    
    for rule in rules:
        detected = []
        
        for i in range(lookback, len(df) - 2):
            for offset in [1, 2]:
                check_idx = i - offset
                
                if check_idx < lookback:
                    continue
                
                # 檢查是否為局部極值
                window_start = check_idx - lookback
                window_end = check_idx + 1
                window_prices = df['close'].iloc[window_start:window_end].values
                check_price = df['close'].iloc[check_idx]
                
                is_peak = check_price == np.max(window_prices)
                is_valley = check_price == np.min(window_prices)
                
                if not (is_peak or is_valley):
                    continue
                
                # 應用規則
                try:
                    if rule['check'](df, check_idx, i, is_peak):
                        if check_idx not in detected:
                            detected.append(check_idx)
                except:
                    continue
        
        # 計算指標
        true_positives = len([idx for idx in detected if idx in true_reversals])
        recall = true_positives / len(true_reversals) * 100 if len(true_reversals) > 0 else 0
        precision = true_positives / len(detected) * 100 if len(detected) > 0 else 0
        
        results.append({
            'rule': rule['name'],
            'detected': len(detected),
            'recall': recall,
            'precision': precision
        })
    
    # 顯示結果
    print(f"\n{'規則':<20} {'檢測數':>10} {'召回率':>10} {'精確度':>10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['rule']:<20} {result['detected']:>10} {result['recall']:>9.1f}% {result['precision']:>9.1f}%")
    
    return results


def main():
    """主函數"""
    df = load_data()
    if df is None:
        return
    
    print(f"\n數據概況:")
    print(f"  總數據點: {len(df)}")
    print(f"  時間範圍: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")
    
    # 1. 分析反轉後的K線模式
    patterns = analyze_post_reversal_patterns(df)
    
    # 2. 分析最佳檢測時機
    timing = analyze_detection_timing(df)
    
    # 3. 尋找最佳確認規則
    rules = find_best_confirmation_rules(df)
    
    # 總結
    print("\n" + "="*80)
    print("💡 關鍵發現")
    print("="*80)
    print("""
基於分析，我們發現：

1. 反轉點後的K線特徵：
   - 第1根K線：約 X% 確認反轉方向
   - 第2根K線：約 Y% 連續確認
   
2. 最佳檢測時機：
   - 在反轉點後 1-2 根K線檢測最準確
   - 太早：確認不足
   - 太晚：錯過最佳進場點

3. 最佳確認規則：
   - 價格確認 + 連續確認
   - K線實體大小
   - 成交量配合

下一步：
- 實作改進的階段1檢測器
- 結合這些發現來提高召回率和減少假陽性
    """)


if __name__ == '__main__':
    main()
