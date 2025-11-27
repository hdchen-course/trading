#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df):
    df['rsi'] = calculate_rsi(df['close'])
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    df['price_change_1h'] = df['close'].pct_change(periods=4)
    df['rsi_change'] = df['rsi'].diff()
    df['high_5'] = df['high'].rolling(window=5).max()  # 5期最高價
    return df

def detect_crash_start_points(df):
    """檢測下跌開始點，而不是下跌過程中的點"""
    crash_starts = []
    
    for i in range(24, len(df) - 24):
        current = df.iloc[i]
        
        # 檢查是否在相對高點附近（前5期最高價的95%以上）
        if current['close'] >= current['high_5'] * 0.95:
            
            # 檢查是否有下跌開始的訊號
            if (current['price_change_1h'] < -0.005 and  # 開始下跌
                current['rsi'] > 45 and                  # RSI還在中高位
                current['volume_ratio'] > 1.2):          # 成交量放大
                
                # 檢查未來是否真的持續下跌
                future_6h = df.iloc[i+1:i+25]
                if len(future_6h) >= 24:
                    current_price = current['close']
                    lowest_price = future_6h['low'].min()
                    drop_pct = (current_price - lowest_price) / current_price
                    
                    # 檢查是否是持續下跌而不是反彈
                    future_2h = df.iloc[i+1:i+9]  # 未來2小時
                    if len(future_2h) > 0:
                        max_rebound = (future_2h['high'].max() - current_price) / current_price
                        
                        # 如果未來2小時反彈不超過0.5%，且總跌幅>1%
                        if max_rebound < 0.005 and drop_pct > 0.01:
                            crash_starts.append({
                                'index': i,
                                'timestamp': current['timestamp'],
                                'price': current['close'],
                                'drop_pct': drop_pct,
                                'is_major_crash': drop_pct > 0.02
                            })
    
    return crash_starts

def visualize_crash_starts():
    print("Loading data...")
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df = add_technical_indicators(df)
    
    print("Detecting crash start points...")
    crash_starts = detect_crash_start_points(df)
    
    # 最近3個月
    recent_df = df.tail(8760)
    recent_start_idx = len(df) - 8760
    
    # 篩選最近3個月的crash starts
    recent_crashes = [c for c in crash_starts if c['index'] >= recent_start_idx]
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    timestamps = pd.to_datetime(recent_df['timestamp'])
    prices = recent_df['close'].values
    
    # 繪製基礎價格線
    ax.plot(timestamps, prices, color='lightgray', linewidth=0.8, alpha=0.6)
    
    # 標記crash start points
    major_crashes = [c for c in recent_crashes if c['is_major_crash']]
    minor_crashes = [c for c in recent_crashes if not c['is_major_crash']]
    
    if major_crashes:
        major_times = [pd.to_datetime(c['timestamp']) for c in major_crashes]
        major_prices = [c['price'] for c in major_crashes]
        ax.scatter(major_times, major_prices, color='red', s=100, alpha=0.8, 
                  label=f'Major Crash Start ({len(major_crashes)})', zorder=5)
    
    if minor_crashes:
        minor_times = [pd.to_datetime(c['timestamp']) for c in minor_crashes]
        minor_prices = [c['price'] for c in minor_crashes]
        ax.scatter(minor_times, minor_prices, color='green', s=60, alpha=0.7,
                  label=f'Minor Crash Start ({len(minor_crashes)})', zorder=5)
    
    ax.set_title('Crash Start Point Detection (Red: >2% drop, Green: 1-2% drop)', fontsize=14)
    ax.set_ylabel('Price (USDT)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('crash_start_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 統計
    total_crashes = len(recent_crashes)
    major_count = len(major_crashes)
    minor_count = len(minor_crashes)
    
    print(f"\n=== Crash Start Detection Results ===")
    print(f"Total crash starts detected: {total_crashes}")
    print(f"Major crashes (>2%): {major_count}")
    print(f"Minor crashes (1-2%): {minor_count}")
    
    if total_crashes > 0:
        avg_drop = np.mean([c['drop_pct'] for c in recent_crashes])
        print(f"Average drop after detection: {avg_drop*100:.2f}%")
        
        # 分析檢測時機
        print(f"\nDetection timing analysis:")
        for i, crash in enumerate(recent_crashes[:5]):  # 顯示前5個
            print(f"  {crash['timestamp']}: {crash['drop_pct']*100:.2f}% drop")

if __name__ == "__main__":
    visualize_crash_starts()
