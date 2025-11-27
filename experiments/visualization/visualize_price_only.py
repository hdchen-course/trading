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
    return df

def detect_crash_signals(df):
    df['crash_signal'] = False
    df['future_drop'] = 0.0
    
    for i in range(24, len(df) - 24):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        if (current['price_change_1h'] < -0.005 and
            current['rsi'] < prev['rsi'] - 3 and
            current['volume_ratio'] > 1.2):
            
            df.iloc[i, df.columns.get_loc('crash_signal')] = True
            
            future_6h = df.iloc[i+1:i+25]
            if len(future_6h) >= 24:
                current_price = current['close']
                lowest_price = future_6h['low'].min()
                drop_pct = (current_price - lowest_price) / current_price
                df.iloc[i, df.columns.get_loc('future_drop')] = drop_pct
    
    return df

def visualize_price_only():
    print("Loading data...")
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df = add_technical_indicators(df)
    df = detect_crash_signals(df)
    
    # 最近3個月
    recent_df = df.tail(8760)
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    timestamps = pd.to_datetime(recent_df['timestamp'])
    prices = recent_df['close'].values
    
    # 創建顏色數組
    colors = ['black'] * len(recent_df)
    drop_amounts = []
    
    for i in range(len(recent_df)):
        if recent_df.iloc[i]['crash_signal']:
            drop = recent_df.iloc[i]['future_drop']
            drop_amounts.append(drop * 100)
            if drop > 0.02:
                colors[i] = 'red'
            else:
                colors[i] = 'orange'
    
    # 分段繪製
    current_color = colors[0]
    start_idx = 0
    
    for i in range(1, len(colors)):
        if colors[i] != current_color or i == len(colors) - 1:
            end_idx = i if i != len(colors) - 1 else i
            
            segment_x = timestamps.iloc[start_idx:end_idx+1]
            segment_y = prices[start_idx:end_idx+1]
            
            if current_color == 'red':
                ax.plot(segment_x, segment_y, color='red', linewidth=2, alpha=0.9)
            elif current_color == 'orange':
                ax.plot(segment_x, segment_y, color='orange', linewidth=2, alpha=0.8)
            else:
                ax.plot(segment_x, segment_y, color='black', linewidth=1, alpha=0.7)
            
            start_idx = i
            current_color = colors[i]
    
    ax.set_title('BTC Price with Crash Detection (Red: >2% drop, Orange: <2% drop)', fontsize=14)
    ax.set_ylabel('Price (USDT)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 圖例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Normal'),
        Line2D([0], [0], color='red', lw=2, label='True Crash (>2%)'),
        Line2D([0], [0], color='orange', lw=2, label='Small Drop (<2%)')
    ]
    ax.legend(handles=legend_elements, fontsize=11)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('price_crash_segments.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 分析橙色段的下跌分布
    orange_drops = [d for i, d in enumerate(drop_amounts) if colors[i] == 'orange']
    red_drops = [d for i, d in enumerate(drop_amounts) if colors[i] == 'red']
    
    print(f"\n=== 詳細分析 ===")
    print(f"紅色段數量: {len(red_drops)}, 平均下跌: {np.mean(red_drops):.2f}%")
    print(f"橙色段數量: {len(orange_drops)}, 平均下跌: {np.mean(orange_drops):.2f}%")
    print(f"橙色段下跌範圍: {np.min(orange_drops):.2f}% - {np.max(orange_drops):.2f}%")
    
    # 橙色段中下跌超過1%的比例
    orange_over_1pct = sum(1 for d in orange_drops if d > 1.0)
    print(f"橙色段中下跌>1%的: {orange_over_1pct}/{len(orange_drops)} ({orange_over_1pct/len(orange_drops)*100:.1f}%)")

if __name__ == "__main__":
    visualize_price_only()
