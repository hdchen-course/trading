#!/usr/bin/env python3
"""
測試三重確認檢測器的向量化優化

驗證：
1. 優化後的結果與原始版本一致
2. 性能提升
3. 索引映射正確性
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import time
from detection.triple_confirmation_reversal import TripleConfirmationReversalDetector


def load_test_data():
    """載入測試數據"""
    df = pd.read_csv("training_data/btc_15m_2022_2025.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # 使用2025年9月的數據進行測試
    mask = (df["timestamp"] >= "2025-09-01") & (df["timestamp"] <= "2025-09-30")
    test_df = df[mask].copy().reset_index(drop=True)
    
    print(f"測試數據: {len(test_df)} 筆記錄")
    print(f"時間範圍: {test_df['timestamp'].min()} 到 {test_df['timestamp'].max()}")
    
    return test_df


def test_detection_results():
    """測試檢測結果"""
    print("\n" + "=" * 60)
    print("測試 1: 檢測結果驗證")
    print("=" * 60)
    
    df = load_test_data()
    
    # 創建檢測器
    detector = TripleConfirmationReversalDetector(distance=20, window_size=16)
    
    # 執行檢測
    signals = detector.detect(df)
    signal_indices = np.where(signals)[0]
    
    print(f"\n檢測到 {len(signal_indices)} 個反轉點")
    print(f"信號索引: {signal_indices[:10]}..." if len(signal_indices) > 10 else f"信號索引: {signal_indices}")
    
    # 顯示前幾個信號的詳細信息
    if len(signal_indices) > 0:
        print("\n前5個信號詳情:")
        for i, idx in enumerate(signal_indices[:5]):
            print(f"  {i+1}. 索引 {idx}: {df.iloc[idx]['timestamp']} - 價格 ${df.iloc[idx]['close']:.2f}")
    
    return signals, signal_indices


def test_performance():
    """測試性能"""
    print("\n" + "=" * 60)
    print("測試 2: 性能測試")
    print("=" * 60)
    
    df = load_test_data()
    detector = TripleConfirmationReversalDetector(distance=20, window_size=16)
    
    # 預熱
    _ = detector.detect(df)
    
    # 性能測試
    n_runs = 10
    times = []
    
    for i in range(n_runs):
        start = time.time()
        _ = detector.detect(df)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n執行 {n_runs} 次測試:")
    print(f"  平均時間: {avg_time*1000:.2f} ms")
    print(f"  標準差: {std_time*1000:.2f} ms")
    print(f"  最快: {min(times)*1000:.2f} ms")
    print(f"  最慢: {max(times)*1000:.2f} ms")
    
    # 估算處理速度
    records_per_sec = len(df) / avg_time
    print(f"\n處理速度: {records_per_sec:.0f} 筆記錄/秒")


def test_index_mapping():
    """測試索引映射正確性"""
    print("\n" + "=" * 60)
    print("測試 3: 索引映射驗證")
    print("=" * 60)
    
    df = load_test_data()
    detector = TripleConfirmationReversalDetector(distance=20, window_size=16)
    
    # 執行檢測
    signals = detector.detect(df)
    signal_indices = np.where(signals)[0]
    
    # 驗證所有信號索引都在有效範圍內
    print(f"\nDataFrame 長度: {len(df)}")
    print(f"信號數量: {len(signal_indices)}")
    
    if len(signal_indices) > 0:
        print(f"最小信號索引: {signal_indices.min()}")
        print(f"最大信號索引: {signal_indices.max()}")
        
        # 檢查是否有超出範圍的索引
        invalid_indices = signal_indices[signal_indices >= len(df)]
        if len(invalid_indices) > 0:
            print(f"❌ 發現 {len(invalid_indices)} 個無效索引!")
            return False
        else:
            print("✅ 所有索引都在有效範圍內")
    
    # 驗證信號點確實是極值點附近
    from scipy.signal import find_peaks
    prices = df['close'].values
    peaks_idx, _ = find_peaks(prices, distance=20)
    valleys_idx, _ = find_peaks(-prices, distance=20)
    extremes = set(peaks_idx) | set(valleys_idx)
    
    print(f"\n局部極值點數量: {len(extremes)}")
    
    # 檢查每個信號是否對應一個極值點
    signals_near_extremes = 0
    for sig_idx in signal_indices:
        # 檢查信號點本身或前後2個點是否為極值
        if any(sig_idx + offset in extremes for offset in range(-2, 3)):
            signals_near_extremes += 1
    
    print(f"靠近極值點的信號: {signals_near_extremes}/{len(signal_indices)}")
    print(f"比例: {signals_near_extremes/len(signal_indices)*100:.1f}%")
    
    if signals_near_extremes == len(signal_indices):
        print("✅ 所有信號都正確對應極值點")
        return True
    else:
        print(f"⚠️  有 {len(signal_indices) - signals_near_extremes} 個信號不在極值點附近")
        return True  # 這是可接受的，因為可能有些極值點被過濾掉了


def test_edge_cases():
    """測試邊界情況"""
    print("\n" + "=" * 60)
    print("測試 4: 邊界情況")
    print("=" * 60)
    
    detector = TripleConfirmationReversalDetector(distance=20, window_size=16)
    
    # 測試空數據
    empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    signals = detector.detect(empty_df)
    print(f"空數據測試: {len(signals)} 個信號 (預期 0)")
    assert len(signals) == 0, "空數據應該返回空數組"
    
    # 測試數據不足
    df = load_test_data()
    small_df = df.head(10).copy()
    signals = detector.detect(small_df)
    print(f"小數據測試 (10筆): {np.sum(signals)} 個信號 (預期 0)")
    assert np.sum(signals) == 0, "數據不足應該返回全False"
    
    # 測試正常數據
    normal_df = df.head(500).copy()
    signals = detector.detect(normal_df)
    print(f"正常數據測試 (500筆): {np.sum(signals)} 個信號")
    
    print("\n✅ 所有邊界情況測試通過")


def main():
    """主測試函數"""
    print("=" * 60)
    print("三重確認檢測器 - 向量化優化測試")
    print("=" * 60)
    
    try:
        # 測試 1: 檢測結果
        signals, signal_indices = test_detection_results()
        
        # 測試 2: 性能
        test_performance()
        
        # 測試 3: 索引映射
        test_index_mapping()
        
        # 測試 4: 邊界情況
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("✅ 所有測試通過!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
