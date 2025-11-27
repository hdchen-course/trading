#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']

def plot_candlestick(ax, df):
    """繪製K線圖"""
    timestamps = pd.to_datetime(df['timestamp'])
    
    for i in range(len(df)):
        row = df.iloc[i]
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']
        
        # 決定顏色
        color = 'red' if close_price < open_price else 'green'
        
        # 繪製影線
        ax.plot([timestamps.iloc[i], timestamps.iloc[i]], [low_price, high_price], 
                color='black', linewidth=1)
        
        # 繪製實體
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        ax.bar(timestamps.iloc[i], body_height, bottom=body_bottom, 
               color=color, alpha=0.7, width=pd.Timedelta(minutes=10))

def plot_specific_period():
    print("Loading data...")
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 篩選指定時間段
    start_time = pd.to_datetime('2025-08-14 07:00:00')
    end_time = pd.to_datetime('2025-08-15 00:00:00')
    
    period_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].copy()
    
    if len(period_df) == 0:
        print("No data found for the specified period")
        return
    
    print(f"Found {len(period_df)} candles for the period")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # K線圖
    plot_candlestick(ax1, period_df)
    ax1.set_title('BTC 15min Candles: 2025-08-14 07:00 - 2025-08-15 00:00', fontsize=14)
    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 成交量圖
    timestamps = pd.to_datetime(period_df['timestamp'])
    ax2.bar(timestamps, period_df['volume'], color='blue', alpha=0.6, width=pd.Timedelta(minutes=10))
    ax2.set_title('Volume', fontsize=12)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 格式化x軸
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('btc_0814_period.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 顯示價格統計
    print(f"\n=== Price Statistics ===")
    print(f"Period: {start_time} to {end_time}")
    print(f"Open: ${period_df.iloc[0]['open']:,.2f}")
    print(f"High: ${period_df['high'].max():,.2f}")
    print(f"Low: ${period_df['low'].min():,.2f}")
    print(f"Close: ${period_df.iloc[-1]['close']:,.2f}")
    print(f"Change: {(period_df.iloc[-1]['close'] - period_df.iloc[0]['open']) / period_df.iloc[0]['open'] * 100:.2f}%")
    print(f"Max drop: {(period_df.iloc[0]['open'] - period_df['low'].min()) / period_df.iloc[0]['open'] * 100:.2f}%")

if __name__ == "__main__":
    plot_specific_period()
