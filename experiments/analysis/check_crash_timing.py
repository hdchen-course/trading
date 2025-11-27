#!/usr/bin/env python3

import pandas as pd

def check_crash_timing():
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 8/14 10:00 - 14:00
    start_time = pd.to_datetime('2025-08-14 10:00:00')
    end_time = pd.to_datetime('2025-08-14 14:00:00')
    
    period_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].copy()
    
    print("=== 8/14 價格數據檢查 ===")
    print("時間\t\t開盤\t最高\t最低\t收盤\t跌幅%")
    print("-" * 70)
    
    prev_close = None
    for i, row in period_df.iterrows():
        timestamp = row['timestamp']
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']
        
        if prev_close is not None:
            change_pct = (close_price - prev_close) / prev_close * 100
        else:
            change_pct = 0
        
        print(f"{timestamp.strftime('%H:%M')}\t{open_price:,.0f}\t{high_price:,.0f}\t{low_price:,.0f}\t{close_price:,.0f}\t{change_pct:+.2f}")
        
        prev_close = close_price
    
    # 找出最大單期跌幅
    period_df['price_change'] = period_df['close'].pct_change() * 100
    max_drop_idx = period_df['price_change'].idxmin()
    max_drop_row = period_df.loc[max_drop_idx]
    
    print(f"\n=== 最大單期跌幅 ===")
    print(f"時間: {max_drop_row['timestamp'].strftime('%H:%M')}")
    print(f"跌幅: {max_drop_row['price_change']:.2f}%")
    print(f"價格: ${max_drop_row['open']:,.0f} -> ${max_drop_row['close']:,.0f}")

if __name__ == "__main__":
    check_crash_timing()
