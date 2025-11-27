#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import talib
import joblib

plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']

def extract_features(df, idx):
    """提取前8小時的特徵 (與訓練時相同)"""
    start_idx = max(0, idx - 32)
    segment = df.iloc[start_idx:idx+1]
    
    if len(segment) < 10:
        return None
    
    features = {}
    
    # 價格趨勢特徵
    features['price_trend_8h'] = (segment['close'].iloc[-1] - segment['close'].iloc[0]) / segment['close'].iloc[0]
    features['price_volatility_8h'] = segment['close'].pct_change().std()
    features['high_low_ratio_8h'] = (segment['high'].max() - segment['low'].min()) / segment['close'].iloc[-1]
    
    # 成交量特徵
    features['volume_trend_8h'] = (segment['volume'].iloc[-1] - segment['volume'].iloc[0]) / segment['volume'].iloc[0]
    features['volume_spike_8h'] = segment['volume'].max() / segment['volume'].mean()
    
    # ATR相關特徵
    current_atr = segment['atr'].iloc[-1]
    if current_atr > 0:
        features['atr_ratio_8h'] = abs(segment['close'].iloc[-1] - segment['close'].iloc[0]) / current_atr
    else:
        features['atr_ratio_8h'] = 0
    
    # RSI變化
    if len(segment) >= 14:
        rsi_values = talib.RSI(segment['close'].values, timeperiod=14)
        if not np.isnan(rsi_values[-1]) and not np.isnan(rsi_values[0]):
            features['rsi_change_8h'] = rsi_values[-1] - rsi_values[0]
        else:
            features['rsi_change_8h'] = 0
    else:
        features['rsi_change_8h'] = 0
    
    # MACD特徵
    if len(segment) >= 26:
        macd, signal, hist = talib.MACD(segment['close'].values)
        if not np.isnan(macd[-1]):
            features['macd_8h'] = macd[-1]
            features['macd_signal_8h'] = signal[-1] if not np.isnan(signal[-1]) else 0
        else:
            features['macd_8h'] = 0
            features['macd_signal_8h'] = 0
    else:
        features['macd_8h'] = 0
        features['macd_signal_8h'] = 0
    
    # 布林帶特徵
    if len(segment) >= 20:
        bb_upper, bb_middle, bb_lower = talib.BBANDS(segment['close'].values, timeperiod=20)
        if not np.isnan(bb_upper[-1]):
            features['bb_position_8h'] = (segment['close'].iloc[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
        else:
            features['bb_position_8h'] = 0.5
    else:
        features['bb_position_8h'] = 0.5
    
    # 移動平均特徵
    if len(segment) >= 20:
        ma20 = segment['close'].rolling(20).mean().iloc[-1]
        features['ma20_distance_8h'] = (segment['close'].iloc[-1] - ma20) / ma20 if not np.isnan(ma20) else 0
    else:
        features['ma20_distance_8h'] = 0
    
    return features

def test_prediction():
    """測試模型在9/10月的預測效果"""
    
    # 載入模型和特徵名稱
    model = joblib.load('balanced_crash_prediction_model_8h_1h.pkl')
    feature_names = joblib.load('balanced_feature_names_8h_1h.pkl')
    
    # 讀取數據
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # 測試9月和10月
    for month in [9, 10]:
        month_df = df[df['timestamp'].dt.month == month].copy()
        
        if len(month_df) == 0:
            continue
            
        print(f"\n=== Testing Month {month} ===")
        
        predictions = []
        
        # 對每個時間點進行預測
        for i in range(32, len(month_df) - 4):  # 需要32個點的歷史數據，4個點的未來數據
            features = extract_features(month_df, i)
            
            if features:
                # 轉換為DataFrame並確保特徵順序正確
                feature_df = pd.DataFrame([features])
                feature_df = feature_df.reindex(columns=feature_names, fill_value=0)
                
                # 預測
                prob = model.predict_proba(feature_df)[0][1]  # 獲取crash的概率
                
                if prob > 0.7:  # 只保留高信心的預測
                    predictions.append({
                        'timestamp': month_df.iloc[i]['timestamp'],
                        'price': month_df.iloc[i]['close'],
                        'probability': prob
                    })
        
        print(f"High confidence predictions (>70%): {len(predictions)}")
        
        if predictions:
            print(f"\nTop predictions for month {month}:")
            sorted_predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)
            for i, pred in enumerate(sorted_predictions[:10]):
                print(f"{i+1:2d}. {pred['timestamp'].strftime('%m-%d %H:%M')}: "
                      f"${pred['price']:,.0f}, confidence: {pred['probability']:.1%}")
        
        # 繪圖
        if predictions:
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            
            # 灰色價格線
            ax.plot(month_df['timestamp'], month_df['close'], color='lightgray', linewidth=1, alpha=0.8)
            
            # 紅色預測點
            pred_times = [p['timestamp'] for p in predictions]
            pred_prices = [p['price'] for p in predictions]
            ax.scatter(pred_times, pred_prices, color='red', s=50, alpha=0.8, zorder=5)
            
            ax.set_title(f'{month}月 BTC: ML預測crash點 (8h+1h模型, 信心度>70%)', fontsize=14)
            ax.set_ylabel('Price (USDT)', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # 格式化x軸
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'ml_prediction_month_{month:02d}_8h_1h.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Chart saved to ml_prediction_month_{month:02d}_8h_1h.png")

if __name__ == "__main__":
    test_prediction()
