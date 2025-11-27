#!/usr/bin/env python3

import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def extract_features(df, start_idx, window=16):
    """提取前4小時(16個15分鐘)的技術指標特徵"""
    if start_idx < window:
        return None
    
    # 取前4小時數據
    window_data = df.iloc[start_idx-window:start_idx].copy()
    
    features = {}
    
    # 價格特徵
    features['price_change_4h'] = (window_data['close'].iloc[-1] - window_data['close'].iloc[0]) / window_data['close'].iloc[0]
    features['price_volatility_4h'] = window_data['close'].pct_change().std()
    features['high_low_ratio'] = (window_data['high'].max() - window_data['low'].min()) / window_data['close'].iloc[-1]
    
    # 成交量特徵
    features['volume_trend'] = np.corrcoef(range(len(window_data)), window_data['volume'])[0,1]
    features['volume_ratio_avg'] = window_data['volume_ratio'].mean()
    features['volume_ratio_max'] = window_data['volume_ratio'].max()
    
    # 技術指標
    features['rsi_last'] = window_data['rsi'].iloc[-1] if 'rsi' in window_data.columns else 0
    features['rsi_change'] = window_data['rsi'].iloc[-1] - window_data['rsi'].iloc[0] if 'rsi' in window_data.columns else 0
    
    # ATR相關
    features['atr_ratio'] = window_data['atr'].iloc[-1] / window_data['close'].iloc[-1]
    features['candle_range_ratio'] = (window_data['high'].iloc[-1] - window_data['low'].iloc[-1]) / window_data['atr'].iloc[-1]
    
    # 趨勢特徵
    features['price_trend'] = np.corrcoef(range(len(window_data)), window_data['close'])[0,1]
    features['consecutive_up'] = sum(1 for i in range(1, len(window_data)) if window_data['close'].iloc[i] > window_data['close'].iloc[i-1])
    
    return features

def create_dataset():
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 計算技術指標
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    
    positive_samples = []
    negative_samples = []
    
    # 只使用1-8月數據作為訓練集
    train_months = [1, 2, 3, 4, 5, 6, 7, 8]
    
    for month in train_months:
        monthly_df = df[df['timestamp'].dt.month == month].copy()
        
        if len(monthly_df) == 0:
            continue
            
        print(f"Processing {month}月...")
        
        # 找出正樣本（會下跌的點）
        for i in range(96, len(monthly_df) - 16):  # 確保有足夠的前後數據
            current_row = monthly_df.iloc[i]
            day_ago_row = monthly_df.iloc[i-96]
            
            current_price = current_row['close']
            day_ago_price = day_ago_row['close']
            price_rise_24h = current_price - day_ago_price
            current_candle_range = current_row['high'] - current_row['low']
            
            atr = current_row['atr']
            volume_ratio = current_row['volume_ratio']
            
            # 檢查是否符合篩選條件
            if (volume_ratio > 1.3 and
                current_candle_range < atr and
                price_rise_24h > atr * 4.0 and
                price_rise_24h / day_ago_price > 0.005 and
                atr > 0):
                
                # 檢查4小時內是否下跌
                future_drop = False
                for j in range(1, 17):
                    if i + j < len(monthly_df):
                        future_price = monthly_df.iloc[i + j]['close']
                        if future_price < current_price * 0.99:
                            future_drop = True
                            break
                
                if future_drop:
                    # 提取前4小時特徵
                    features = extract_features(monthly_df, i)
                    if features:
                        features['label'] = 1  # 正樣本
                        positive_samples.append(features)
        
        # 生成負樣本（隨機選擇不會下跌的點）
        np.random.seed(42)
        negative_indices = np.random.choice(range(96, len(monthly_df) - 16), 
                                          size=min(200, len(monthly_df) - 112), 
                                          replace=False)
        
        for i in negative_indices:
            current_row = monthly_df.iloc[i]
            current_price = current_row['close']
            
            # 檢查4小時內是否沒有大幅下跌
            no_major_drop = True
            for j in range(1, 17):
                if i + j < len(monthly_df):
                    future_price = monthly_df.iloc[i + j]['close']
                    if future_price < current_price * 0.97:  # 下跌超過3%
                        no_major_drop = False
                        break
            
            if no_major_drop:
                features = extract_features(monthly_df, i)
                if features:
                    features['label'] = 0  # 負樣本
                    negative_samples.append(features)
    
    print(f"正樣本數量: {len(positive_samples)}")
    print(f"負樣本數量: {len(negative_samples)}")
    
    # 合併數據
    all_samples = positive_samples + negative_samples
    dataset_df = pd.DataFrame(all_samples)
    
    # 分離特徵和標籤
    X = dataset_df.drop('label', axis=1)
    y = dataset_df['label']
    
    # 分割訓練測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 訓練模型
    print("訓練隨機森林模型...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # 評估模型
    y_pred = rf.predict(X_test)
    print("\n模型評估結果:")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩陣:")
    print(confusion_matrix(y_test, y_pred))
    
    # 特徵重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特徵重要性 (前10名):")
    print(feature_importance.head(10))
    
    # 保存模型和特徵名稱
    joblib.dump(rf, 'model_training/output/crash/crash_prediction_model.pkl')
    joblib.dump(X.columns.tolist(), 'model_training/output/crash/feature_names.pkl')
    dataset_df.to_csv('crash_prediction_dataset.csv', index=False)
    
    print(f"\n模型已保存為 crash_prediction_model.pkl")
    print(f"數據集已保存為 crash_prediction_dataset.csv")
    
    return rf, X.columns.tolist()

if __name__ == "__main__":
    model, feature_names = create_dataset()
