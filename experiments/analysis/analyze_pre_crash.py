#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_pre_crash():
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 8/14 07:00 - 15:00 (擴大範圍看crash前後)
    start_time = pd.to_datetime('2025-08-14 07:00:00')
    end_time = pd.to_datetime('2025-08-14 15:00:00')
    
    period_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].copy()
    
    # 計算指標
    period_df['rsi'] = calculate_rsi(period_df['close'])
    period_df['volume_ma5'] = period_df['volume'].rolling(window=5).mean()
    period_df['volume_ratio'] = period_df['volume'] / period_df['volume_ma5']
    period_df['price_change_1h'] = period_df['close'].pct_change(periods=4)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10))
    
    timestamps = period_df['timestamp']
    crash_start = pd.to_datetime('2025-08-14 11:00:00')
    crash_end = pd.to_datetime('2025-08-14 13:00:00')
    
    # 價格圖
    ax1.plot(timestamps, period_df['close'], 'b-', linewidth=2)
    ax1.axvspan(crash_start, crash_end, alpha=0.3, color='red', label='Crash Period')
    ax1.set_title('BTC Price with Crash Period Highlighted')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RSI + 成交量比率
    ax2_twin = ax2.twinx()
    ax2.plot(timestamps, period_df['rsi'], 'purple', linewidth=2, label='RSI')
    ax2_twin.bar(timestamps, period_df['volume_ratio'], alpha=0.6, color='orange', 
                width=pd.Timedelta(minutes=10), label='Volume Ratio')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    ax2.axvspan(crash_start, crash_end, alpha=0.3, color='red')
    ax2.set_ylabel('RSI')
    ax2_twin.set_ylabel('Volume Ratio')
    ax2.set_title('RSI and Volume Ratio')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 1小時價格變化
    ax3.plot(timestamps, period_df['price_change_1h'] * 100, 'green', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.axhline(y=-0.5, color='r', linestyle='--', alpha=0.5, label='Warning -0.5%')
    ax3.axvspan(crash_start, crash_end, alpha=0.3, color='red')
    ax3.set_title('1-Hour Price Change (%)')
    ax3.set_ylabel('Change (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 添加關鍵時間點的垂直線
        ax.axvline(x=pd.to_datetime('2025-08-14 10:45:00'), color='orange', 
                  linestyle='--', alpha=0.7, label='Warning Signal')
        ax.axvline(x=crash_start, color='red', linestyle='-', alpha=0.8, label='Crash Start')
        
        # 設置相同的x軸範圍
        ax.set_xlim(start_time, end_time)
    
    plt.tight_layout()
    plt.savefig('pre_crash_analysis_0814.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 分析crash前徵兆
    pre_crash = period_df[period_df['timestamp'] < crash_start]
    
    print("=== Crash前徵兆分析 ===")
    
    # 檢查10:00-11:00的情況
    hour_before = pre_crash[pre_crash['timestamp'] >= pd.to_datetime('2025-08-14 10:00:00')]
    if len(hour_before) > 0:
        print(f"10:00-11:00 (crash前1小時):")
        print(f"  RSI: {hour_before.iloc[0]['rsi']:.1f} -> {hour_before.iloc[-1]['rsi']:.1f}")
        print(f"  價格: ${hour_before.iloc[0]['close']:,.0f} -> ${hour_before.iloc[-1]['close']:,.0f}")
        print(f"  最大成交量比率: {hour_before['volume_ratio'].max():.2f}")
    
    # 檢查早期警告訊號
    warnings = []
    for i in range(len(pre_crash)):
        row = pre_crash.iloc[i]
        if (row['price_change_1h'] < -0.005 and 
            row['volume_ratio'] > 1.2):
            warnings.append(f"  {row['timestamp'].strftime('%H:%M')}: 1h跌{row['price_change_1h']*100:.2f}%, 成交量比率{row['volume_ratio']:.2f}")
    
    print(f"\n早期警告訊號:")
    for w in warnings:
        print(w)

if __name__ == "__main__":
    analyze_pre_crash()
