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

def balanced_precise_rules():
    """平衡精確度和召回率的規則"""
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['atr'] = calculate_atr(df)
    
    for month in [9, 10]:
        print(f"\n=== Testing {month}月 Balanced Precise Rules ===")
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
            
            # 平衡的預測規則
            predicted = False
            score = 0  # 評分系統
            
            # 獲取技術分析數據
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
            
            # 評分因子1: 強勢動量 (0-3分)
            if rise_from_96h_ago > required_rise_96h_ago * 2.0:
                score += 3
            elif rise_from_96h_ago > required_rise_96h_ago * 1.5:
                score += 2
            elif rise_from_96h_ago > required_rise_96h_ago * 1.2:
                score += 1
            
            # 評分因子2: 成交量確認 (0-2分)
            current_volume = volume[-1]
            avg_volume_20 = np.mean(volume[-20:])
            volume_ratio = current_volume / avg_volume_20
            
            if volume_ratio > 2.0:
                score += 2
            elif volume_ratio > 1.5:
                score += 1
            
            # 評分因子3: RSI過熱 (0-2分)
            rsi = talib.RSI(close, timeperiod=14)[-1]
            if rsi > 80:
                score += 2
            elif rsi > 70:
                score += 1
            
            # 評分因子4: 反轉形態 (0-2分)
            doji = talib.CDLDOJI(open_price, high, low, close)[-1]
            shooting_star = talib.CDLSHOOTINGSTAR(open_price, high, low, close)[-1]
            hanging_man = talib.CDLHANGINGMAN(open_price, high, low, close)[-1]
            
            reversal_patterns = sum([abs(doji), abs(shooting_star), abs(hanging_man)])
            if reversal_patterns > 0:
                score += 2
            
            # 評分因子5: 布林帶位置 (0-1分)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            if bb_position > 0.8:
                score += 1
            
            # 評分因子6: 價格位置 (0-1分)
            high_24h = np.max(close[-96:])  # 24小時高點
            if current_price >= high_24h * 0.98:  # 接近24小時高點
                score += 1
            
            # 評分因子7: MACD信號 (0-1分)
            macd, macd_signal, macd_hist = talib.MACD(close)
            if len(macd) > 2 and macd[-1] < macd[-2]:  # MACD轉向下
                score += 1
            
            # 決策: 需要至少7分才預測為正
            if score >= 7:
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
        
        if precision >= 0.4:
            print(f"🎉 SUCCESS! Achieved {precision:.1%} precision!")
        else:
            print(f"❌ Precision {precision:.1%} still below 40% target")
        
        # 測試不同分數閾值
        print("\nTesting different score thresholds:")
        for threshold in [5, 6, 7, 8, 9]:
            y_pred_thresh = [score >= threshold for score in [
                sum([
                    3 if (month_df.iloc[i]['close'] - month_df.iloc[i-384]['close'])/month_df.iloc[i-384]['close'] > 3*month_df.iloc[i]['atr']/month_df.iloc[i]['close']*2.0 else
                    2 if (month_df.iloc[i]['close'] - month_df.iloc[i-384]['close'])/month_df.iloc[i-384]['close'] > 3*month_df.iloc[i]['atr']/month_df.iloc[i]['close']*1.5 else
                    1 if (month_df.iloc[i]['close'] - month_df.iloc[i-384]['close'])/month_df.iloc[i-384]['close'] > 3*month_df.iloc[i]['atr']/month_df.iloc[i]['close']*1.2 else 0,
                    2 if month_df.iloc[i]['volume'] > month_df.iloc[i-20:i]['volume'].mean() * 2.0 else
                    1 if month_df.iloc[i]['volume'] > month_df.iloc[i-20:i]['volume'].mean() * 1.5 else 0
                ]) for i in range(384, len(month_df) - 16) if not pd.isna(month_df.iloc[i]['atr']) and month_df.iloc[i]['atr'] > 0
            ]]
            
            if len(y_pred_thresh) == len(y_true):
                pred_count = sum(y_pred_thresh)
                tp = sum(a and p for a, p in zip(y_true, y_pred_thresh))
                prec = tp / pred_count if pred_count > 0 else 0
                print(f"  Threshold {threshold}: Predicted={pred_count}, TP={tp}, Precision={prec:.1%}")

if __name__ == "__main__":
    balanced_precise_rules()
