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

def calculate_stoch_rsi(prices, period=14, k_period=3, d_period=3):
    rsi = calculate_rsi(prices, period)
    min_rsi = rsi.rolling(window=period).min()
    max_rsi = rsi.rolling(window=period).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    k = stoch_rsi.rolling(window=k_period).mean()
    d = k.rolling(window=d_period).mean()
    return k.fillna(0), d.fillna(0)

def add_technical_indicators(df):
    df['rsi'] = calculate_rsi(df['close'])
    df['stoch_k'], df['stoch_d'] = calculate_stoch_rsi(df['close'])
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
        
        # 早期警告條件
        if (current['price_change_1h'] < -0.005 and
            current['rsi'] < prev['rsi'] - 3 and
            current['volume_ratio'] > 1.2):
            
            df.iloc[i, df.columns.get_loc('crash_signal')] = True
            
            # 計算未來6小時下跌
            future_6h = df.iloc[i+1:i+25]
            if len(future_6h) >= 24:
                current_price = current['close']
                lowest_price = future_6h['low'].min()
                drop_pct = (current_price - lowest_price) / current_price
                df.iloc[i, df.columns.get_loc('future_drop')] = drop_pct
    
    return df

def plot_colored_segments(df):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 價格圖 - 分段著色
    timestamps = pd.to_datetime(df['timestamp'])
    prices = df['close'].values
    
    # 創建顏色數組
    colors = ['black'] * len(df)
    for i in range(len(df)):
        if df.iloc[i]['crash_signal']:
            drop = df.iloc[i]['future_drop']
            if drop > 0.02:  # 真實下跌超過2%
                colors[i] = 'red'
            else:  # 假訊號
                colors[i] = 'orange'
    
    # 分段繪製價格線
    current_color = colors[0]
    start_idx = 0
    
    for i in range(1, len(colors)):
        if colors[i] != current_color or i == len(colors) - 1:
            end_idx = i if i != len(colors) - 1 else i
            
            # 繪製當前段
            segment_x = timestamps[start_idx:end_idx+1]
            segment_y = prices[start_idx:end_idx+1]
            
            if current_color == 'red':
                ax1.plot(segment_x, segment_y, color='red', linewidth=1.5, alpha=0.8)
            elif current_color == 'orange':
                ax1.plot(segment_x, segment_y, color='orange', linewidth=1.2, alpha=0.7)
            else:
                ax1.plot(segment_x, segment_y, color='black', linewidth=0.8, alpha=0.7)
            
            start_idx = i
            current_color = colors[i]
    
    ax1.set_title('BTC Price with Crash Detection Segments')
    ax1.set_ylabel('Price (USDT)')
    ax1.grid(True, alpha=0.3)
    
    # 創建圖例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Normal'),
        Line2D([0], [0], color='red', lw=2, label='True Crash'),
        Line2D([0], [0], color='orange', lw=2, label='False Signal')
    ]
    ax1.legend(handles=legend_elements)
    
    # RSI圖 - 同樣分段著色
    rsi_values = df['rsi'].values
    start_idx = 0
    current_color = colors[0]
    
    for i in range(1, len(colors)):
        if colors[i] != current_color or i == len(colors) - 1:
            end_idx = i if i != len(colors) - 1 else i
            
            segment_x = timestamps[start_idx:end_idx+1]
            segment_y = rsi_values[start_idx:end_idx+1]
            
            if current_color == 'red':
                ax2.plot(segment_x, segment_y, color='red', linewidth=1.5, alpha=0.8)
            elif current_color == 'orange':
                ax2.plot(segment_x, segment_y, color='orange', linewidth=1.2, alpha=0.7)
            else:
                ax2.plot(segment_x, segment_y, color='blue', linewidth=0.8, alpha=0.7)
            
            start_idx = i
            current_color = colors[i]
    
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax2.set_title('RSI Indicator')
    ax2.set_ylabel('RSI')
    ax2.grid(True, alpha=0.3)
    
    # StochRSI圖
    ax3.plot(timestamps, df['stoch_k'], 'g-', linewidth=0.8, label='StochK')
    ax3.plot(timestamps, df['stoch_d'], 'm-', linewidth=0.8, label='StochD')
    ax3.set_title('StochRSI Indicator')
    ax3.set_ylabel('StochRSI')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 成交量比率圖
    ax4.plot(timestamps, df['volume_ratio'], 'purple', linewidth=0.8)
    ax4.axhline(y=1.2, color='r', linestyle='--', alpha=0.5, label='Threshold 1.2')
    ax4.set_title('Volume Ratio')
    ax4.set_ylabel('Volume Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 格式化x軸
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('colored_crash_segments.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_colored_segments():
    print("Loading data...")
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    
    print("Calculating indicators...")
    df = add_technical_indicators(df)
    
    print("Detecting crash signals...")
    df = detect_crash_signals(df)
    
    # 選擇最近3個月數據
    recent_df = df.tail(8760)
    
    print("Plotting colored segments...")
    plot_colored_segments(recent_df)
    
    # 統計
    total_signals = recent_df['crash_signal'].sum()
    true_crashes = (recent_df['crash_signal'] & (recent_df['future_drop'] > 0.02)).sum()
    false_signals = (recent_df['crash_signal'] & (recent_df['future_drop'] <= 0.02)).sum()
    
    print(f"\n=== Statistics ===")
    print(f"Total signals: {total_signals}")
    print(f"True crashes (red): {true_crashes}")
    print(f"False signals (orange): {false_signals}")
    print(f"Precision: {true_crashes/total_signals*100:.1f}%")

if __name__ == "__main__":
    visualize_colored_segments()
