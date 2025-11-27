#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
import os
sys.path.append('..')

# 直接從 detection 模組導入所需的類和函數
from detection import MacdReversalDetection
from utils import indicators as crypto_util

def generate_lstm_training_data():
    """生成LSTM訓練數據：2025/01-08的信號點前後24小時"""
    
    # 使用crypto_util中的函數載入數據
    print("載入原始數據...")
    # 檢查crypto_util中的函數
    print("Available functions in crypto_util:", [f for f in dir(crypto_util) if not f.startswith('_')])
    
    # 先檢查crypto_visualize.py如何載入數據
    # 暫時使用pandas直接讀取CSV
    try:
        df = pd.read_csv('../training_data/btc_12months_15m.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 篩選日期範圍
        start_date = pd.to_datetime("2025-01-01")
        end_date = pd.to_datetime("2025-08-31")
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        print(f"載入 {len(df)} 條原始數據")
        
    except FileNotFoundError:
        print("未找到btc_15min_data.csv，嘗試其他方式...")
        return
    
    # 計算技術指標
    print("計算技術指標...")
    df = crypto_util.calculate_technical_indicators(df)
    
    print("使用crypto_visualize.py的檢測邏輯...")
    detector = MacdReversalDetection()
    signal_mask = detector.detect(df)  # 返回布爾數組
    
    # 獲取信號位置
    signal_positions = np.where(signal_mask)[0].tolist()
    
    print(f"找到 {len(signal_positions)} 個信號點")
    if len(signal_positions) > 0:
        print("信號時間點:")
        for pos in signal_positions[:5]:  # 顯示前5個
            print(f"  {df.iloc[pos]['timestamp']}")
    
    # 需要analysis_df來獲取特徵
    analysis_df = crypto_util.analyze_cumulative_metrics(df, 16)
    
    # 特徵列
    feature_cols = [
        'close_price', 'ema_100', 'ema_50', 'avg_amplitude', 'avg_volume',
        'avg_macd', 'avg_rsi', 'price_above_sar_ratio', 'avg_bb_position',
        'volatility', 'price_change_pct', 'price_atr_ratio_1h', 'volume_trend',
        'macd_2nd', 'price_2nd', 'vol_2nd', 'atr_2nd'
    ]
    
    sequences = []
    labels = []
    
    # 每個序列96個連續時間步（24小時）
    window_size = 96  # 24小時 * 4個15分鐘
    
    # 信號位置需要映射到analysis_df的索引
    for signal_pos in signal_positions:
        # signal_pos是原始df的索引，需要轉換為analysis_df的索引
        analysis_pos = signal_pos - 15  # analysis_df比原始df少15行（window_size-1）
        
        # 確保有足夠的歷史數據（96個時間步）
        if analysis_pos >= window_size and analysis_pos < len(analysis_df):
            # 取信號點前96個時間步作為序列
            start_idx = analysis_pos - window_size
            end_idx = analysis_pos
            
            sequence = analysis_df.iloc[start_idx:end_idx][feature_cols].values
            sequences.append(sequence)
            labels.append(1)  # 信號
    
    # 生成非信號樣本（隨機選擇）
    non_signal_count = len(sequences) * 3  # 1:3比例
    
    for _ in range(non_signal_count):
        random_idx = np.random.randint(window_size, len(analysis_df))
        
        # 確保不是信號點附近
        signal_analysis_positions = [pos - 15 for pos in signal_positions if pos >= 15]
        if not any(abs(random_idx - sig_pos) < window_size//4 for sig_pos in signal_analysis_positions):
            start_idx = random_idx - window_size
            end_idx = random_idx
            
            sequence = analysis_df.iloc[start_idx:end_idx][feature_cols].values
            sequences.append(sequence)
            labels.append(0)  # 非信號
    
    # 轉換為numpy數組
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"生成訓練數據: {X.shape}, 標籤: {y.shape}")
    print(f"信號樣本: {np.sum(y)}, 非信號樣本: {len(y) - np.sum(y)}")
    
    # 保存數據
    np.save('model_training/lstm/data/X_train.npy', X)
    np.save('model_training/lstm/data/y_train.npy', y)
    
    print("訓練數據已保存到 X_train.npy 和 y_train.npy")
    
    return X, y

if __name__ == "__main__":
    generate_lstm_training_data()
