import pandas as pd
import numpy as np

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

def enhanced_rule_based_prediction():
    """使用增強規則進行預測"""
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['atr'] = calculate_atr(df)
    
    for month in [9, 10]:
        print(f"\n=== Testing {month}月 Enhanced Rules ===")
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
            
            # 增強規則預測
            predicted = False
            
            # 規則1: 極強動量 + 高成交量
            if (rise_from_96h_ago > required_rise_96h_ago * 2.0 and  # 2倍ATR要求
                current_price > four_hours_ago_price * 1.005):      # 明顯高於4小時前
                
                # 檢查成交量
                current_volume = month_df.iloc[i]['volume']
                avg_volume_24h = month_df.iloc[i-96:i]['volume'].mean()
                
                if current_volume > avg_volume_24h * 1.5:  # 1.5倍成交量
                    
                    # 檢查RSI不要過高
                    recent_closes = month_df.iloc[i-20:i]['close']
                    delta = recent_closes.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50
                    
                    if rsi < 80:  # RSI不要太高
                        
                        # 檢查價格位置
                        recent_high = month_df.iloc[i-96:i]['close'].max()
                        recent_low = month_df.iloc[i-96:i]['close'].min()
                        price_position = (current_price - recent_low) / (recent_high - recent_low)
                        
                        if 0.7 < price_position < 0.95:  # 在高位但不是最高位
                            predicted = True
            
            # 規則2: 連續上漲後的小幅回調
            if not predicted:
                # 檢查前4小時是否連續上漲
                recent_4h = month_df.iloc[i-16:i]['close']
                if len(recent_4h) >= 16:
                    consecutive_up = 0
                    for j in range(len(recent_4h)-1, 0, -1):
                        if recent_4h.iloc[j] > recent_4h.iloc[j-1]:
                            consecutive_up += 1
                        else:
                            break
                    
                    if (consecutive_up >= 8 and  # 至少連續上漲2小時
                        rise_from_96h_ago > required_rise_96h_ago * 1.8 and
                        current_price > four_hours_ago_price * 1.002):
                        predicted = True
            
            # 規則3: 突破關鍵阻力位
            if not predicted:
                # 檢查是否突破24小時高點
                high_24h = month_df.iloc[i-96:i-4]['close'].max()  # 排除最近1小時
                
                if (current_price > high_24h * 1.01 and  # 突破24小時高點1%
                    rise_from_96h_ago > required_rise_96h_ago * 1.5):
                    
                    # 檢查突破時的成交量
                    current_volume = month_df.iloc[i]['volume']
                    avg_volume_4h = month_df.iloc[i-16:i]['volume'].mean()
                    
                    if current_volume > avg_volume_4h * 1.3:
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

if __name__ == "__main__":
    enhanced_rule_based_prediction()
