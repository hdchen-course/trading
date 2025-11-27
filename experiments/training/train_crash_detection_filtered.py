#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def calculate_rsi(prices, period=14):
    """計算RSI指標"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stoch_rsi(prices, period=14, k_period=3, d_period=3):
    """計算StochRSI指標"""
    rsi = calculate_rsi(prices, period)
    min_rsi = rsi.rolling(window=period).min()
    max_rsi = rsi.rolling(window=period).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    k = stoch_rsi.rolling(window=k_period).mean()
    d = k.rolling(window=d_period).mean()
    return k.fillna(0), d.fillna(0)

def add_technical_indicators(df):
    """添加技術指標"""
    df['rsi'] = calculate_rsi(df['close'])
    df['stoch_k'], df['stoch_d'] = calculate_stoch_rsi(df['close'])
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    df['price_change_1h'] = df['close'].pct_change(periods=4)  # 4個15分鐘 = 1小時
    df['rsi_change'] = df['rsi'].diff()
    return df

def filter_crash_conditions(df):
    """篩選符合早期下跌警告條件的區間"""
    # 早期警告條件（基於之前分析的結果）
    crash_conditions = (
        (df['price_change_1h'] < -0.005) &  # 1小時下跌>0.5%
        (df['rsi_change'] < -3) &           # RSI下降>3點
        (df['volume_ratio'] > 1.2) &        # 成交量>1.2倍平均
        (df['rsi'].notna()) &               # 確保有RSI數據
        (df['stoch_k'].notna()) &           # 確保有StochRSI數據
        (df['volume_ratio'].notna())        # 確保有成交量比率
    )
    
    return df[crash_conditions].copy()

def prepare_features_and_labels(df):
    """準備特徵和標籤"""
    features = []
    labels = []
    
    for i in range(len(df) - 4):  # 需要未來1小時數據標記
        current_row = df.iloc[i]
        
        # 特徵向量
        feature_vector = [
            current_row['rsi'],
            current_row['stoch_k'],
            current_row['stoch_d'],
            current_row['price_change_1h'],
            current_row['volume_ratio'],
            current_row['rsi_change'],
            current_row['stoch_k'] - current_row['stoch_d'],  # StochRSI差值
        ]
        
        # 標籤：未來1小時內最大跌幅
        future_prices = df.iloc[i+1:i+5]['low']  # 未來4個15分鐘
        current_price = current_row['close']
        max_drop = (current_price - future_prices.min()) / current_price
        
        # 下跌超過2%為正樣本
        label = 1 if max_drop > 0.02 else 0
        
        features.append(feature_vector)
        labels.append(label)
    
    return np.array(features), np.array(labels)

def train_crash_model():
    """訓練下跌檢測模型"""
    print("載入原始K線數據...")
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    
    print(f"原始數據: {len(df)} 條記錄")
    
    # 計算技術指標
    print("計算技術指標...")
    df = add_technical_indicators(df)
    
    # 篩選符合條件的區間
    print("篩選早期警告條件...")
    filtered_df = filter_crash_conditions(df)
    print(f"符合條件的區間: {len(filtered_df)} 條記錄")
    
    if len(filtered_df) < 100:
        print("符合條件的數據太少，無法訓練模型")
        return None
    
    # 準備特徵和標籤
    print("準備特徵...")
    X, y = prepare_features_and_labels(filtered_df)
    
    print(f"訓練樣本: {len(X)}")
    print(f"正樣本比例: {y.mean():.3f}")
    
    if len(X) < 50:
        print("訓練樣本太少")
        return None
    
    # 分割訓練測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 訓練模型
    print("訓練隨機森林模型...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # 評估
    y_pred = model.predict(X_test)
    print("\n模型評估:")
    print(classification_report(y_test, y_pred))
    
    # 特徵重要性
    feature_names = ['RSI', 'StochK', 'StochD', 'Price_Change_1h', 
                    'Volume_Ratio', 'RSI_Change', 'StochRSI_Diff']
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特徵重要性:")
    print(importance)
    
    # 保存模型
    joblib.dump(model, 'crash_detection_filtered.pkl')
    print("\n模型已保存為 crash_detection_filtered.pkl")
    
    return model

if __name__ == "__main__":
    train_crash_model()
