#!/usr/bin/env python3

import pandas as pd
import numpy as np
import talib

def extract_features(df, idx):
    """提取前4小時的特徵"""
    # 4小時 = 16個15分鐘
    start_idx = max(0, idx - 16)
    segment = df.iloc[start_idx:idx+1]
    
    if len(segment) < 10:
        return None
    
    features = {}
    
    # 價格趨勢特徵
    features['price_trend_4h'] = (segment['close'].iloc[-1] - segment['close'].iloc[0]) / segment['close'].iloc[0]
    features['price_volatility_4h'] = segment['close'].pct_change().std()
    features['high_low_ratio_4h'] = (segment['high'].max() - segment['low'].min()) / segment['close'].iloc[-1]
    
    # 成交量特徵
    features['volume_trend_4h'] = (segment['volume'].iloc[-1] - segment['volume'].iloc[0]) / segment['volume'].iloc[0]
    features['volume_spike_4h'] = segment['volume'].max() / segment['volume'].mean()
    
    # ATR相關特徵
    current_atr = segment['atr'].iloc[-1]
    if current_atr > 0:
        features['atr_ratio_4h'] = abs(segment['close'].iloc[-1] - segment['close'].iloc[0]) / current_atr
    else:
        features['atr_ratio_4h'] = 0
    
    # RSI變化
    if len(segment) >= 14:
        rsi_values = talib.RSI(segment['close'].values, timeperiod=14)
        if not np.isnan(rsi_values[-1]) and not np.isnan(rsi_values[0]):
            features['rsi_change_4h'] = rsi_values[-1] - rsi_values[0]
        else:
            features['rsi_change_4h'] = 0
    else:
        features['rsi_change_4h'] = 0
    
    # MACD特徵
    if len(segment) >= 26:
        macd, signal, hist = talib.MACD(segment['close'].values)
        if not np.isnan(macd[-1]):
            features['macd_4h'] = macd[-1]
            features['macd_signal_4h'] = signal[-1] if not np.isnan(signal[-1]) else 0
        else:
            features['macd_4h'] = 0
            features['macd_signal_4h'] = 0
    else:
        features['macd_4h'] = 0
        features['macd_signal_4h'] = 0
    
    # 布林帶特徵
    if len(segment) >= 20:
        bb_upper, bb_middle, bb_lower = talib.BBANDS(segment['close'].values, timeperiod=20)
        if not np.isnan(bb_upper[-1]):
            features['bb_position_4h'] = (segment['close'].iloc[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
        else:
            features['bb_position_4h'] = 0.5
    else:
        features['bb_position_4h'] = 0.5
    
    # 移動平均特徵
    if len(segment) >= 20:
        ma20 = segment['close'].rolling(20).mean().iloc[-1]
        features['ma20_distance_4h'] = (segment['close'].iloc[-1] - ma20) / ma20 if not np.isnan(ma20) else 0
    else:
        features['ma20_distance_4h'] = 0
    
    return features

def create_crash_dataset():
    """使用october_crash_detection.py的條件創建數據集，但修改為4h+1h窗口"""
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 計算技術指標
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    all_samples = []
    
    # 處理1-8月作為訓練數據
    for month in range(1, 9):
        month_df = df[df['timestamp'].dt.month == month].copy()
        
        if len(month_df) == 0:
            continue
            
        print(f"Processing month {month}...")
        
        # 檢測crash點 (使用october_crash_detection.py的條件)
        crash_points = []
        
        for i in range(192, len(month_df) - 4):  # 需要192個15分鐘(48小時)前數據，4個15分鐘(1小時)後數據
            current_row = month_df.iloc[i]
            two_days_ago_row = month_df.iloc[i-192]  # 48小時前
            eight_hours_ago_row = month_df.iloc[i-32]  # 8小時前
            
            current_price = current_row['close']
            two_days_ago_price = two_days_ago_row['close']
            eight_hours_ago_price = eight_hours_ago_row['close']
            
            # 計算48小時振幅和8小時漲幅
            price_range_48h = abs(current_price - two_days_ago_price)
            price_rise_8h = current_price - eight_hours_ago_price
            
            atr = current_row['atr']
            
            # 檢查條件：48小時振幅高於8倍ATR + 8小時漲幅高於2倍ATR
            if (price_range_48h > atr * 8.0 and 
                price_rise_8h > atr * 2.0 and 
                atr > 0):
                
                # 檢查1小時後是否下跌超過3倍ATR
                future_drop = False
                for j in range(1, 5):  # 4個15分鐘 = 1小時
                    if i + j < len(month_df):
                        future_price = month_df.iloc[i + j]['close']
                        price_drop = current_price - future_price
                        if price_drop > atr * 3.0:  # 下跌超過3倍ATR
                            future_drop = True
                            break
                
                if future_drop:
                    crash_points.append(i)
        
        print(f"Found {len(crash_points)} crash points in month {month}")
        
        # 為每個crash點提取特徵 (正樣本)
        for crash_idx in crash_points:
            features = extract_features(month_df, crash_idx)
            if features:
                features['is_crash'] = 1
                features['month'] = month
                features['timestamp'] = month_df.iloc[crash_idx]['timestamp']
                all_samples.append(features)
        
        # 生成負樣本 (非crash點)
        negative_count = len(crash_points) * 20
        negative_indices = []
        
        for _ in range(negative_count):
            # 隨機選擇一個不是crash點的位置
            while True:
                idx = np.random.randint(192, len(month_df) - 4)
                if idx not in crash_points:
                    # 確保這個點1小時後沒有大幅下跌
                    current_price = month_df.iloc[idx]['close']
                    atr = month_df.iloc[idx]['atr']
                    
                    has_big_drop = False
                    for j in range(1, 5):  # 檢查1小時內
                        if idx + j < len(month_df):
                            future_price = month_df.iloc[idx + j]['close']
                            price_drop = current_price - future_price
                            if price_drop > atr * 3.0:
                                has_big_drop = True
                                break
                    
                    if not has_big_drop:
                        negative_indices.append(idx)
                        break
        
        # 提取負樣本特徵
        for neg_idx in negative_indices:
            features = extract_features(month_df, neg_idx)
            if features:
                features['is_crash'] = 0
                features['month'] = month
                features['timestamp'] = month_df.iloc[neg_idx]['timestamp']
                all_samples.append(features)
    
    # 轉換為DataFrame
    dataset_df = pd.DataFrame(all_samples)
    
    print(f"\nDataset created:")
    print(f"Total samples: {len(dataset_df)}")
    print(f"Positive samples (crashes): {len(dataset_df[dataset_df['is_crash'] == 1])}")
    print(f"Negative samples: {len(dataset_df[dataset_df['is_crash'] == 0])}")
    
    # 保存數據集
    dataset_df.to_csv('crash_prediction_dataset_4h_1h.csv', index=False)
    print("Dataset saved to crash_prediction_dataset_4h_1h.csv")
    
    return dataset_df

if __name__ == "__main__":
    create_crash_dataset()
