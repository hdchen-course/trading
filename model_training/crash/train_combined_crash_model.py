#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from crypto_provider import BinanceDataProvider

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

def prepare_crash_features(df):
    """準備包含RSI和StochRSI的特徵數據"""
    # 計算技術指標
    df['rsi'] = calculate_rsi(df['close'])
    df['stoch_k'], df['stoch_d'] = calculate_stoch_rsi(df['close'])
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['price_change_1h'] = df['close'].pct_change(periods=12)  # 1小時變化
    
    # 特徵工程
    features = []
    labels = []
    
    for i in range(24, len(df) - 6):  # 需要6小時後續數據來標記
        # 當前特徵
        current_features = [
            df.iloc[i]['rsi'],
            df.iloc[i]['stoch_k'],
            df.iloc[i]['stoch_d'],
            df.iloc[i]['price_change_1h'],
            df.iloc[i]['volume'] / df.iloc[i]['volume_ma'] if df.iloc[i]['volume_ma'] > 0 else 1,
            df.iloc[i-1]['rsi'] - df.iloc[i]['rsi'],  # RSI變化
            df.iloc[i-1]['stoch_k'] - df.iloc[i]['stoch_k'],  # StochRSI變化
        ]
        
        # 標記：未來6小時內是否下跌超過2%
        future_low = df.iloc[i+1:i+7]['low'].min()
        current_price = df.iloc[i]['close']
        is_crash = (current_price - future_low) / current_price > 0.02
        
        features.append(current_features)
        labels.append(1 if is_crash else 0)
    
    return np.array(features), np.array(labels)

def train_model():
    """訓練結合RSI和StochRSI的下跌預測模型"""
    print("載入數據...")
    provider = BinanceDataProvider()
    df = provider.get_data('BTCUSDT', '5m', 8640)  # 30天數據
    
    print("準備特徵...")
    X, y = prepare_crash_features(df)
    
    print(f"特徵數量: {X.shape[0]}, 正樣本比例: {y.mean():.3f}")
    
    # 分割訓練測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 訓練隨機森林模型
    print("訓練模型...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # 評估模型
    y_pred = model.predict(X_test)
    print("\n模型評估:")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩陣:")
    print(confusion_matrix(y_test, y_pred))
    
    # 特徵重要性
    feature_names = ['RSI', 'StochK', 'StochD', 'Price_Change_1h', 
                    'Volume_Ratio', 'RSI_Change', 'StochK_Change']
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特徵重要性:")
    print(importance)
    
    # 保存模型
    joblib.dump(model, 'model_training/output/crash/crash_prediction_model.pkl')
    print("\n模型已保存為 crash_prediction_model.pkl")
    
    return model

if __name__ == "__main__":
    model = train_model()
