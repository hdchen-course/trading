#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 設定字型避免中文顯示問題
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

def calculate_rsi(prices, period=14):
    """計算RSI指標"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stoch_rsi(prices, period=14, k_period=3, d_period=3):
    """計算StochRSI指標"""
    rsi = calculate_rsi(prices, period)
    min_rsi = rsi.rolling(window=period).min()
    max_rsi = rsi.rolling(window=period).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    k = stoch_rsi.rolling(window=k_period).mean()
    d = k.rolling(window=d_period).mean()
    return k.fillna(0), d.fillna(0)

def add_technical_indicators(df):
    """添加技術指標"""
    df['rsi'] = calculate_rsi(df['close'])
    df['stoch_k'], df['stoch_d'] = calculate_stoch_rsi(df['close'])
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    df['price_change_1h'] = df['close'].pct_change(periods=4)  # 4個15分鐘 = 1小時
    df['rsi_change'] = df['rsi'].diff()
    return df

def detect_early_crash_signals(df):
    """檢測早期下跌訊號（基於之前analyze_sustained_crashes.py的邏輯）"""
    signals = []
    
    for i in range(24, len(df) - 24):  # 需要前後6小時數據
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # 早期警告條件
        if (current['price_change_1h'] < -0.005 and  # 1小時下跌>0.5%
            current['rsi'] < prev['rsi'] - 3 and     # RSI下降>3點
            current['volume_ratio'] > 1.2):          # 成交量>1.2倍平均
            
            # 檢查後續6小時是否真的下跌
            future_6h = df.iloc[i+1:i+25]  # 未來24個15分鐘 = 6小時
            if len(future_6h) >= 24:
                lowest_price = future_6h['low'].min()
                current_price = current['close']
                drop_pct = (current_price - lowest_price) / current_price
                
                signals.append({
                    'timestamp': current['timestamp'],
                    'price': current['close'],
                    'rsi': current['rsi'],
                    'stoch_k': current['stoch_k'],
                    'volume_ratio': current['volume_ratio'],
                    'drop_pct': drop_pct,
                    'is_true_crash': drop_pct > 0.02  # 真實下跌超過2%
                })
    
    return pd.DataFrame(signals)

def visualize_filtered_samples():
    """可視化篩選出的樣本"""
    print("Loading data...")
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("Calculating technical indicators...")
    df = add_technical_indicators(df)
    
    print("Detecting crash signals...")
    signals_df = detect_early_crash_signals(df)
    
    if len(signals_df) == 0:
        print("No signals found!")
        return
    
    # 選擇最近3個月數據進行可視化
    recent_df = df.tail(8760)  # 3個月的15分鐘數據
    recent_signals = signals_df[signals_df['timestamp'] >= recent_df['timestamp'].iloc[0]]
    
    # 分離真實下跌和假訊號
    true_crashes = recent_signals[recent_signals['is_true_crash']]
    false_signals = recent_signals[~recent_signals['is_true_crash']]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 價格圖
    ax1.plot(recent_df['timestamp'], recent_df['close'], 'k-', linewidth=0.5, alpha=0.7)
    
    if len(true_crashes) > 0:
        ax1.scatter(pd.to_datetime(true_crashes['timestamp']), true_crashes['price'], 
                   color='red', s=30, alpha=0.8, label=f'True Crash ({len(true_crashes)})')
    if len(false_signals) > 0:
        ax1.scatter(pd.to_datetime(false_signals['timestamp']), false_signals['price'], 
                   color='orange', s=20, alpha=0.6, label=f'False Signal ({len(false_signals)})')
    
    ax1.set_title('BTC Price & Crash Signals')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RSI圖
    ax2.plot(recent_df['timestamp'], recent_df['rsi'], 'b-', linewidth=0.8)
    if len(true_crashes) > 0:
        ax2.scatter(pd.to_datetime(true_crashes['timestamp']), true_crashes['rsi'], 
                   color='red', s=30, alpha=0.8)
    if len(false_signals) > 0:
        ax2.scatter(pd.to_datetime(false_signals['timestamp']), false_signals['rsi'], 
                   color='orange', s=20, alpha=0.6)
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax2.set_title('RSI Indicator')
    ax2.set_ylabel('RSI')
    ax2.grid(True, alpha=0.3)
    
    # StochRSI圖
    ax3.plot(recent_df['timestamp'], recent_df['stoch_k'], 'g-', linewidth=0.8, label='StochK')
    ax3.plot(recent_df['timestamp'], recent_df['stoch_d'], 'm-', linewidth=0.8, label='StochD')
    if len(true_crashes) > 0:
        ax3.scatter(pd.to_datetime(true_crashes['timestamp']), true_crashes['stoch_k'], 
                   color='red', s=30, alpha=0.8)
    if len(false_signals) > 0:
        ax3.scatter(pd.to_datetime(false_signals['timestamp']), false_signals['stoch_k'], 
                   color='orange', s=20, alpha=0.6)
    ax3.set_title('StochRSI Indicator')
    ax3.set_ylabel('StochRSI')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 成交量比率圖
    ax4.plot(recent_df['timestamp'], recent_df['volume_ratio'], 'purple', linewidth=0.8)
    if len(true_crashes) > 0:
        ax4.scatter(pd.to_datetime(true_crashes['timestamp']), true_crashes['volume_ratio'], 
                   color='red', s=30, alpha=0.8)
    if len(false_signals) > 0:
        ax4.scatter(pd.to_datetime(false_signals['timestamp']), false_signals['volume_ratio'], 
                   color='orange', s=20, alpha=0.6)
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
    plt.savefig('filtered_crash_samples_corrected.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 統計信息
    total_signals = len(signals_df)
    true_positives = len(signals_df[signals_df['is_true_crash']])
    false_positives = len(signals_df[~signals_df['is_true_crash']])
    
    print(f"\n=== Statistics ===")
    print(f"Total signals: {total_signals}")
    print(f"True crashes: {true_positives} ({true_positives/total_signals*100:.1f}%)")
    print(f"False signals: {false_positives} ({false_positives/total_signals*100:.1f}%)")
    print(f"Precision: {true_positives/(true_positives+false_positives)*100:.1f}%")
    
    # 顯示平均下跌幅度
    if true_positives > 0:
        avg_drop = signals_df[signals_df['is_true_crash']]['drop_pct'].mean()
        print(f"Average true crash drop: {avg_drop*100:.2f}%")

if __name__ == "__main__":
    visualize_filtered_samples()
