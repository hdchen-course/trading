#!/usr/bin/env python3

import pandas as pd
import numpy as np
import talib
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']

def extract_features(df, start_idx, window=16):
    """提取前4小時特徵"""
    if start_idx < window:
        return None
    
    window_data = df.iloc[start_idx-window:start_idx].copy()
    
    features = {}
    features['price_change_4h'] = (window_data['close'].iloc[-1] - window_data['close'].iloc[0]) / window_data['close'].iloc[0]
    features['price_volatility_4h'] = window_data['close'].pct_change().std()
    features['high_low_ratio'] = (window_data['high'].max() - window_data['low'].min()) / window_data['close'].iloc[-1]
    features['volume_trend'] = np.corrcoef(range(len(window_data)), window_data['volume'])[0,1]
    features['volume_ratio_avg'] = window_data['volume_ratio'].mean()
    features['volume_ratio_max'] = window_data['volume_ratio'].max()
    features['rsi_last'] = window_data['rsi'].iloc[-1]
    features['rsi_change'] = window_data['rsi'].iloc[-1] - window_data['rsi'].iloc[0]
    features['atr_ratio'] = window_data['atr'].iloc[-1] / window_data['close'].iloc[-1]
    features['candle_range_ratio'] = (window_data['high'].iloc[-1] - window_data['low'].iloc[-1]) / window_data['atr'].iloc[-1]
    features['price_trend'] = np.corrcoef(range(len(window_data)), window_data['close'])[0,1]
    features['consecutive_up'] = sum(1 for i in range(1, len(window_data)) if window_data['close'].iloc[i] > window_data['close'].iloc[i-1])
    
    return features

def predict_crashes():
    # 載入模型
    model = joblib.load('balanced_crash_prediction_model.pkl')
    feature_names = joblib.load('balanced_feature_names.pkl')
    
    # 載入數據
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 計算技術指標
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    
    # 測試9月和10月
    test_months = [9, 10]
    
    for month in test_months:
        monthly_df = df[df['timestamp'].dt.month == month].copy()
        
        if len(monthly_df) == 0:
            continue
            
        print(f"預測 {month}月...")
        
        predicted_crashes = []
        
        # 對每個時間點進行預測
        for i in range(16, len(monthly_df)):
            features = extract_features(monthly_df, i)
            if features:
                # 轉換為DataFrame並確保特徵順序
                feature_df = pd.DataFrame([features])
                feature_df = feature_df.reindex(columns=feature_names, fill_value=0)
                
                # 預測
                prediction = model.predict(feature_df)[0]
                probability = model.predict_proba(feature_df)[0][1]  # 下跌概率
                
                if prediction == 1 and probability > 0.7:  # 高信心預測
                    predicted_crashes.append({
                        'timestamp': monthly_df.iloc[i]['timestamp'],
                        'price': monthly_df.iloc[i]['close'],
                        'probability': probability
                    })
        
        print(f"{month}月預測到 {len(predicted_crashes)} 個下跌點")
        
        # 繪圖
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        # 灰色價格線
        ax.plot(monthly_df['timestamp'], monthly_df['close'], 'lightgray', linewidth=1, alpha=0.7)
        
        # 紅色預測點
        if predicted_crashes:
            crash_times = [c['timestamp'] for c in predicted_crashes]
            crash_prices = [c['price'] for c in predicted_crashes]
            ax.scatter(crash_times, crash_prices, color='red', s=50, alpha=0.8, zorder=5)
        
        ax.set_title(f'{month}月 BTC: ML預測下跌點 (概率>70%)', fontsize=14)
        ax.set_ylabel('Price (USDT)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 格式化x軸
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'ml_prediction_month_{month:02d}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 顯示預測詳情
        if predicted_crashes:
            print(f"\n{month}月預測結果:")
            for i, crash in enumerate(predicted_crashes[:10]):
                print(f"{i+1:2d}. {crash['timestamp'].strftime('%m-%d %H:%M')}: "
                      f"${crash['price']:,.0f}, 概率: {crash['probability']:.1%}")
            if len(predicted_crashes) > 10:
                print(f"... 還有 {len(predicted_crashes) - 10} 個預測點")

if __name__ == "__main__":
    predict_crashes()
