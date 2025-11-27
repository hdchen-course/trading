#!/usr/bin/env python3

import pandas as pd
import talib

def debug_october_signals():
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 10月份數據
    october_df = df[df['timestamp'].dt.month == 10].copy()
    
    # 計算技術指標
    october_df['atr'] = talib.ATR(october_df['high'], october_df['low'], october_df['close'], timeperiod=14)
    october_df['volume_ma20'] = october_df['volume'].rolling(window=20).mean()
    october_df['volume_ratio'] = october_df['volume'] / october_df['volume_ma20']
    
    # 檢查前幾個信號
    signals_found = 0
    
    for i in range(96, len(october_df)):
        current_row = october_df.iloc[i]
        day_ago_row = october_df.iloc[i-96]
        
        current_price = current_row['close']
        day_ago_price = day_ago_row['close']
        price_rise_24h = current_price - day_ago_price
        current_candle_range = current_row['high'] - current_row['low']
        
        atr = current_row['atr']
        volume_ratio = current_row['volume_ratio']
        
        # 檢查條件
        if (volume_ratio > 1.3 and
            current_candle_range < atr and
            price_rise_24h > atr * 4.0 and
            price_rise_24h / day_ago_price > 0.01 and
            atr > 0):
            
            signals_found += 1
            
            print(f"\n=== Signal {signals_found}: {current_row['timestamp'].strftime('%m-%d %H:%M')} ===")
            print(f"Current price: ${current_price:,.0f}")
            print(f"24h ago price: ${day_ago_price:,.0f}")
            print(f"24h rise: ${price_rise_24h:,.0f} ({price_rise_24h/day_ago_price*100:.2f}%)")
            print(f"ATR: ${atr:.0f}")
            print(f"24h rise / ATR: {price_rise_24h / atr:.2f}x")
            print(f"Volume ratio: {volume_ratio:.2f}x")
            print(f"Candle range / ATR: {current_candle_range / atr:.2f}x")
            
            # 檢查前後價格趨勢
            if i >= 4 and i < len(october_df) - 4:
                prev_prices = [october_df.iloc[i-j]['close'] for j in range(1, 5)]
                next_prices = [october_df.iloc[i+j]['close'] for j in range(1, 5)]
                
                print(f"Previous 4 candles: {[f'${p:,.0f}' for p in prev_prices]}")
                print(f"Next 4 candles: {[f'${p:,.0f}' for p in next_prices]}")
            
            if signals_found >= 6:  # 只看前6個信號
                break

if __name__ == "__main__":
    debug_october_signals()
