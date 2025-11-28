#!/usr/bin/env python3
"""
評估 ML Reversal 檢測器在 2025 年數據上的表現

訓練數據：2023-2024 (2年)
驗證數據：2025 (未見過的數據)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np


def load_validation_data():
    """載入 2025 年的驗證數據"""
    print("=" * 80)
    print("📊 載入 2025 年驗證數據")
    print("=" * 80)
    
    data_file = "training_data/reversal_training_data_2025-01-01_to_2025-11-28_15m.csv"
    
    if not os.path.exists(data_file):
        print(f"\n❌ 找不到數據檔案: {data_file}")
        return None
    
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"\n✅ 數據載入成功")
    print(f"   數據行數: {len(df)} 行")
    print(f"   時間範圍: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"   真實反轉點: {df['reverse'].sum()} 個 ({df['reverse'].sum()/len(df)*100:.2f}%)")
    
    return df


def main():
    """主函數"""
    print("=" * 80)
    print("🤖 ML Reversal 檢測器評估")
    print("=" * 80)
    
    # 載入數據
    df = load_validation_data()
    
    if df is None:
        return
    
    # 使用 ML Reversal 檢測器產出訊號點
    print("\n" + "=" * 80)
    print("🔍 執行 ML Reversal 檢測")
    print("=" * 80)
    
    from detection.ml_reversal import MLReversalDetector
    
    detector = MLReversalDetector(distance=20, window_size=16)
    signals = detector.detect(df)
    
    # 獲取訊號點索引
    signal_indices = np.where(signals)[0]
    
    print(f"\n✅ 檢測完成")
    print(f"   檢測到的訊號點: {len(signal_indices)} 個")
    
    # 計算有多少訊號點的前 1 或前 2 個位置是真實反轉點
    print("\n" + "=" * 80)
    print("📊 計算命中率")
    print("=" * 80)
    
    # 獲取真實反轉點索引
    true_reversal_indices = set(df[df['reverse'] == True].index.tolist())
    
    print(f"\n真實反轉點: {len(true_reversal_indices)} 個")
    
    # 檢查每個訊號點 (向量化版本)
    # 為每個訊號點生成前 1 和前 2 的索引
    offset_1 = signal_indices - 1
    offset_2 = signal_indices - 2
    
    # 過濾掉負數索引
    valid_offset_1 = offset_1[offset_1 >= 0]
    valid_offset_2 = offset_2[offset_2 >= 0]
    
    # 檢查哪些索引是真實反轉點
    true_reversal_array = np.array(list(true_reversal_indices))
    
    # 使用 np.isin 快速檢查
    hits_offset_1 = np.isin(valid_offset_1, true_reversal_array)
    hits_offset_2 = np.isin(valid_offset_2, true_reversal_array)
    
    # 找出命中的訊號點（offset-1 或 offset-2 命中）
    # 需要對應回原始 signal_indices
    hit_mask = np.zeros(len(signal_indices), dtype=bool)
    hit_mask[signal_indices >= 1] |= hits_offset_1
    hit_mask[signal_indices >= 2] |= hits_offset_2
    
    hit_count = np.sum(hit_mask)
    
    # 找出所有被匹配到的真實反轉點
    matched_reversals = set()
    matched_reversals.update(valid_offset_1[hits_offset_1])
    matched_reversals.update(valid_offset_2[hits_offset_2])
    
    print(f"\n命中統計:")
    print(f"   檢測到的訊號點: {len(signal_indices)} 個")
    print(f"   命中的訊號點: {hit_count} 個")
    print(f"   命中率: {hit_count / len(signal_indices) * 100:.2f}%")
    print(f"\n召回率統計:")
    print(f"   真實反轉點: {len(true_reversal_indices)} 個")
    print(f"   被檢測到的反轉點: {len(matched_reversals)} 個")
    print(f"   召回率: {len(matched_reversals) / len(true_reversal_indices) * 100:.2f}%")


if __name__ == "__main__":
    main()
