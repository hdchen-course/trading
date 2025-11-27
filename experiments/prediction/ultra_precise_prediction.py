import pandas as pd
import numpy as np
import talib

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def ultra_precise_rules():
    """使用極度嚴格的規則達到高精確度"""
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['atr'] = calculate_atr(df)
    
    for month in [9, 10]:
        print(f"\n=== Testing {month}月 Ultra Precise Rules ===")
        month_df = df[df['timestamp'].dt.month == month].copy().reset_index(drop=True)
        
        if len(month_df) == 0:
            continue
        
        predictions = []
        actual_spikes = []
        
        for i in range(384, len(month_df) - 16):
            current_row = month_df.iloc[i]
            current_price = current_row['close']
            current_atr = current_row['atr']
            
            if pd.isna(current_atr) or current_atr <= 0:
                continue
            
            # 基本條件檢查
            ninety_six_hours_ago_price = month_df.iloc[i - 384]['close']
            four_hours_ago_price = month_df.iloc[i - 16]['close']
            four_hours_later_price = month_df.iloc[i + 16]['close']
            
            rise_from_96h_ago = (current_price - ninety_six_hours_ago_price) / ninety_six_hours_ago_price
            required_rise_96h_ago = 3 * current_atr / current_price
            rise_vs_4h_later = (current_price - four_hours_later_price) / current_price
            required_rise_4h_later = 3 * current_atr / current_price
            
            actual_spike = (rise_from_96h_ago > required_rise_96h_ago and 
                          current_price > four_hours_ago_price and
                          rise_vs_4h_later > required_rise_4h_later)
            
            # 極度嚴格的預測規則
            predicted = False
            
            # 獲取數據用於技術分析
            data_slice = month_df.iloc[max(0, i-100):i+1]
            high = data_slice['high'].values
            low = data_slice['low'].values
            close = data_slice['close'].values
            open_price = data_slice['open'].values
            volume = data_slice['volume'].values
            
            if len(close) < 50:
                predictions.append(predicted)
                actual_spikes.append(actual_spike)
                continue
            
            # 規則1: 極強動量 + 反轉形態 + 高成交量
            if (rise_from_96h_ago > required_rise_96h_ago * 3.0 and  # 3倍ATR要求
                current_price > four_hours_ago_price * 1.01):       # 明顯高於4小時前
                
                # 檢查反轉形態
                doji = talib.CDLDOJI(open_price, high, low, close)[-1]
                shooting_star = talib.CDLSHOOTINGSTAR(open_price, high, low, close)[-1]
                hanging_man = talib.CDLHANGINGMAN(open_price, high, low, close)[-1]
                dark_cloud = talib.CDLDARKCLOUDCOVER(open_price, high, low, close)[-1]
                
                has_reversal_pattern = any([doji != 0, shooting_star != 0, hanging_man != 0, dark_cloud != 0])
                
                if has_reversal_pattern:
                    # 檢查成交量爆發
                    current_volume = volume[-1]
                    avg_volume_20 = np.mean(volume[-20:])
                    
                    if current_volume > avg_volume_20 * 2.0:  # 2倍成交量
                        
                        # 檢查RSI過熱
                        rsi = talib.RSI(close, timeperiod=14)[-1]
                        
                        if rsi > 75:  # RSI過熱
                            
                            # 檢查布林帶位置
                            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
                            bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                            
                            if bb_position > 0.9:  # 接近布林帶上軌
                                predicted = True
            
            # 規則2: 突破新高後的立即反轉信號
            if not predicted:
                # 檢查是否創新高
                high_50 = np.max(close[-50:-1])  # 排除當前K線
                
                if current_price > high_50 * 1.02:  # 突破50期高點2%
                    
                    # 檢查MACD背離
                    macd, macd_signal, macd_hist = talib.MACD(close)
                    
                    if (len(macd) > 5 and 
                        macd[-1] < macd[-5] and  # MACD下降
                        current_price > close[-5]):  # 價格上升
                        
                        # 檢查成交量確認
                        volume_ratio = volume[-1] / np.mean(volume[-10:])
                        
                        if (volume_ratio > 1.5 and  # 成交量放大
                            rise_from_96h_ago > required_rise_96h_ago * 2.5):  # 強勢上漲
                            predicted = True
            
            # 規則3: 連續上漲後的疲態信號
            if not predicted:
                # 檢查連續上漲
                consecutive_up = 0
                for j in range(len(close)-1, max(0, len(close)-20), -1):
                    if j > 0 and close[j] > close[j-1]:
                        consecutive_up += 1
                    else:
                        break
                
                if consecutive_up >= 10:  # 連續上漲10期以上
                    
                    # 檢查上影線
                    current_upper_shadow = high[-1] - max(open_price[-1], close[-1])
                    current_body = abs(close[-1] - open_price[-1])
                    
                    if (current_upper_shadow > current_body * 2 and  # 長上影線
                        rise_from_96h_ago > required_rise_96h_ago * 2.0):
                        
                        # 檢查成交量萎縮
                        recent_volume = np.mean(volume[-5:])
                        prev_volume = np.mean(volume[-15:-5])
                        
                        if recent_volume < prev_volume * 0.8:  # 成交量萎縮
                            predicted = True
            
            predictions.append(predicted)
            actual_spikes.append(actual_spike)
        
        # 統計結果
        y_true = actual_spikes
        y_pred = predictions
        
        predicted_count = sum(y_pred)
        true_positives = sum(a and p for a, p in zip(y_true, y_pred))
        precision = true_positives / predicted_count if predicted_count > 0 else 0
        recall = true_positives / sum(y_true) if sum(y_true) > 0 else 0
        
        print(f"Actual spikes: {sum(y_true)}")
        print(f"Predicted: {predicted_count}")
        print(f"True positives: {true_positives}")
        print(f"Precision: {precision:.1%}")
        print(f"Recall: {recall:.1%}")
        
        if predicted_count > 0:
            print(f"Success! Precision {precision:.1%} {'✓' if precision >= 0.4 else '✗'}")

if __name__ == "__main__":
    ultra_precise_rules()
