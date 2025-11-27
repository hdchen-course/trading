#!/usr/bin/env python3

import pandas as pd
import talib

def analyze_atr_ratio():
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 計算ATR (14期)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # 8/14 09:00 - 14:00
    start_time = pd.to_datetime('2025-08-14 09:00:00')
    end_time = pd.to_datetime('2025-08-14 14:00:00')
    
    period_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].copy()
    
    print("=== 1小時跌幅 vs ATR 比較 ===")
    print("時間\t\t1h跌幅%\tATR\t跌幅/ATR\t警告")
    print("-" * 60)
    
    warning_ratios = []
    
    for i in range(4, len(period_df)):
        current_row = period_df.iloc[i]
        hour_ago_row = period_df.iloc[i-4]
        
        current_price = current_row['close']
        hour_ago_price = hour_ago_row['close']
        change_1h_pct = (current_price - hour_ago_price) / hour_ago_price * 100
        change_1h_abs = abs(current_price - hour_ago_price)
        
        atr = current_row['atr']
        atr_ratio = change_1h_abs / atr if atr > 0 else 0
        
        is_warning = change_1h_pct < -0.5
        warning_flag = "***" if is_warning else ""
        
        print(f"{current_row['timestamp'].strftime('%H:%M')}\t\t{change_1h_pct:+.2f}\t{atr:.0f}\t{atr_ratio:.2f}\t\t{warning_flag}")
        
        if is_warning:
            warning_ratios.append({
                'time': current_row['timestamp'].strftime('%H:%M'),
                'drop_pct': change_1h_pct,
                'atr': atr,
                'ratio': atr_ratio
            })
    
    if warning_ratios:
        print(f"\n=== 警告訊號的ATR比率分析 ===")
        for w in warning_ratios:
            print(f"{w['time']}: 跌幅{w['drop_pct']:.2f}% = ATR的{w['ratio']:.2f}倍")
        
        avg_ratio = sum(w['ratio'] for w in warning_ratios) / len(warning_ratios)
        print(f"\n平均ATR比率: {avg_ratio:.2f}倍")
        print(f"ATR範圍: {min(w['atr'] for w in warning_ratios):.0f} - {max(w['atr'] for w in warning_ratios):.0f}")

if __name__ == "__main__":
    analyze_atr_ratio()
