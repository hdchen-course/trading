#!/usr/bin/env python3

import pandas as pd

def check_1h_changes():
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 8/14 09:00 - 14:00 (擴大範圍)
    start_time = pd.to_datetime('2025-08-14 09:00:00')
    end_time = pd.to_datetime('2025-08-14 14:00:00')
    
    period_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].copy()
    
    print("=== 1小時價格變化檢查 ===")
    print("時間\t\t當前價格\t1小時前價格\t1小時跌幅%")
    print("-" * 60)
    
    for i in range(4, len(period_df)):  # 從第5個開始，因為需要前4個15分鐘
        current_row = period_df.iloc[i]
        hour_ago_row = period_df.iloc[i-4]  # 4個15分鐘 = 1小時前
        
        current_price = current_row['close']
        hour_ago_price = hour_ago_row['close']
        change_1h = (current_price - hour_ago_price) / hour_ago_price * 100
        
        print(f"{current_row['timestamp'].strftime('%H:%M')}\t\t{current_price:,.0f}\t\t{hour_ago_price:,.0f}\t\t{change_1h:+.2f}")
        
        # 檢查是否符合警告條件
        if change_1h < -0.5:
            print(f"  *** 警告訊號: 1小時跌幅{change_1h:.2f}% ***")

if __name__ == "__main__":
    check_1h_changes()
