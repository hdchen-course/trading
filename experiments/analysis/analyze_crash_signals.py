#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_crash_signals():
    print("Loading data...")
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 擴大時間範圍，看crash前的情況
    start_time = pd.to_datetime('2025-08-14 07:00:00')
    end_time = pd.to_datetime('2025-08-15 00:00:00')
    
    period_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].copy()
    
    # 計算技術指標
    period_df['rsi'] = calculate_rsi(period_df['close'])
    period_df['volume_ma5'] = period_df['volume'].rolling(window=5).mean()
    period_df['volume_ratio'] = period_df['volume'] / period_df['volume_ma5']
    period_df['price_change_1h'] = period_df['close'].pct_change(periods=4)
    period_df['rsi_change'] = period_df['rsi'].diff()
    
    # 找出11:00-13:00的crash時間點
    crash_start = pd.to_datetime('2025-08-14 11:00:00')
    crash_end = pd.to_datetime('2025-08-14 13:00:00')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    timestamps = period_df['timestamp']
    
    # 價格圖
    ax1.plot(timestamps, period_df['close'], 'b-', linewidth=1.5)
    ax1.axvspan(crash_start, crash_end, alpha=0.3, color='red', label='Crash Period')
    ax1.set_title('BTC Price', fontsize=12)
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RSI圖
    ax2.plot(timestamps, period_df['rsi'], 'purple', linewidth=1.5)
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
    ax2.axvspan(crash_start, crash_end, alpha=0.3, color='red')
    ax2.set_title('RSI', fontsize=12)
    ax2.set_ylabel('RSI')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 成交量比率圖
    ax3.bar(timestamps, period_df['volume_ratio'], color='orange', alpha=0.7, width=pd.Timedelta(minutes=10))
    ax3.axhline(y=1.5, color='r', linestyle='--', alpha=0.5, label='High Volume')
    ax3.axvspan(crash_start, crash_end, alpha=0.3, color='red')
    ax3.set_title('Volume Ratio (vs 5-period MA)', fontsize=12)
    ax3.set_ylabel('Volume Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 1小時價格變化
    ax4.plot(timestamps, period_df['price_change_1h'] * 100, 'green', linewidth=1.5)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.axhline(y=-0.5, color='r', linestyle='--', alpha=0.5, label='Warning Level')
    ax4.axvspan(crash_start, crash_end, alpha=0.3, color='red')
    ax4.set_title('1-Hour Price Change (%)', fontsize=12)
    ax4.set_ylabel('Price Change (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 格式化x軸
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('crash_signal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 分析crash前的徵兆
    pre_crash = period_df[period_df['timestamp'] < crash_start]
    crash_period = period_df[(period_df['timestamp'] >= crash_start) & (period_df['timestamp'] <= crash_end)]
    
    print(f"\n=== Crash前徵兆分析 ===")
    
    # 找出crash前1小時的情況
    pre_crash_1h = pre_crash.tail(4)  # 最後4個15分鐘
    if len(pre_crash_1h) > 0:
        print(f"Crash前1小時 (10:00-11:00):")
        print(f"  價格變化: {(pre_crash_1h.iloc[-1]['close'] - pre_crash_1h.iloc[0]['open']) / pre_crash_1h.iloc[0]['open'] * 100:.2f}%")
        print(f"  RSI變化: {pre_crash_1h.iloc[-1]['rsi'] - pre_crash_1h.iloc[0]['rsi']:.2f}")
        print(f"  平均成交量比率: {pre_crash_1h['volume_ratio'].mean():.2f}")
        print(f"  最後RSI: {pre_crash_1h.iloc[-1]['rsi']:.2f}")
    
    # 檢查是否有早期警告訊號
    warning_signals = []
    for i in range(len(pre_crash)):
        row = pre_crash.iloc[i]
        if (row['price_change_1h'] < -0.005 and 
            row['rsi_change'] < -3 and 
            row['volume_ratio'] > 1.2):
            warning_signals.append({
                'time': row['timestamp'],
                'price': row['close'],
                'rsi': row['rsi'],
                'volume_ratio': row['volume_ratio']
            })
    
    print(f"\n早期警告訊號數量: {len(warning_signals)}")
    for signal in warning_signals:
        print(f"  {signal['time'].strftime('%H:%M')}: 價格${signal['price']:,.0f}, RSI={signal['rsi']:.1f}, 成交量比率={signal['volume_ratio']:.2f}")

if __name__ == "__main__":
    analyze_crash_signals()
