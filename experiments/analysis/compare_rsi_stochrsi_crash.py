#!/usr/bin/env python3
"""
比較RSI vs StochRSI在早期下跌檢測策略中的表現
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import talib
import pandas_ta as ta

def compare_rsi_stochrsi_crash():
    """比較RSI和StochRSI在下跌檢測中的表現"""
    
    # 讀取數據
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ 載入15分鐘數據: {len(df)} 根K線")
    
    # 計算技術指標
    df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
    df['stochrsi_k'], df['stochrsi_d'] = talib.STOCHRSI(df['close'].values, timeperiod=14)
    df['stochrsi_k'] *= 100  # 轉換為0-100範圍
    df['stochrsi_d'] *= 100
    
    df['volume_ma'] = df['volume'].rolling(96).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['return_1h'] = df['close'].pct_change(4) * 100
    
    # RSI策略信號
    rsi_signals = []
    
    # StochRSI策略信號
    stochrsi_signals = []
    
    for i in range(96, len(df) - 24):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # RSI策略（原版）
        if (current['return_1h'] < -0.5 and
            current['rsi'] < prev['rsi'] - 3 and
            current['volume_ratio'] > 1.2):
            
            future_6h = df.iloc[i:i+24]
            if len(future_6h) >= 24:
                lowest_price = future_6h['low'].min()
                drop_pct = (lowest_price - current['close']) / current['close'] * 100
                
                if drop_pct < -1.5:
                    rsi_signals.append({
                        'timestamp': current['timestamp'],
                        'price': current['close'],
                        'drop_pct': drop_pct,
                        'rsi': current['rsi'],
                        'rsi_change': current['rsi'] - prev['rsi']
                    })
        
        # StochRSI策略（調整版）
        if (current['return_1h'] < -0.5 and
            current['stochrsi_k'] < prev['stochrsi_k'] - 5 and  # StochRSI更敏感，調整閾值
            current['volume_ratio'] > 1.2):
            
            future_6h = df.iloc[i:i+24]
            if len(future_6h) >= 24:
                lowest_price = future_6h['low'].min()
                drop_pct = (lowest_price - current['close']) / current['close'] * 100
                
                if drop_pct < -1.5:
                    stochrsi_signals.append({
                        'timestamp': current['timestamp'],
                        'price': current['close'],
                        'drop_pct': drop_pct,
                        'stochrsi': current['stochrsi_k'],
                        'stochrsi_change': current['stochrsi_k'] - prev['stochrsi_k']
                    })
    
    print(f"\n=== 策略比較結果 ===")
    print(f"RSI策略信號數量: {len(rsi_signals)}")
    print(f"StochRSI策略信號數量: {len(stochrsi_signals)}")
    
    # 統計分析
    if rsi_signals:
        rsi_drops = [s['drop_pct'] for s in rsi_signals]
        print(f"\nRSI策略:")
        print(f"  平均後續跌幅: {np.mean(rsi_drops):.2f}%")
        print(f"  最大後續跌幅: {np.min(rsi_drops):.2f}%")
        print(f"  成功率(跌>1.5%): {sum(1 for d in rsi_drops if d < -1.5)/len(rsi_drops)*100:.1f}%")
    
    if stochrsi_signals:
        stochrsi_drops = [s['drop_pct'] for s in stochrsi_signals]
        print(f"\nStochRSI策略:")
        print(f"  平均後續跌幅: {np.mean(stochrsi_drops):.2f}%")
        print(f"  最大後續跌幅: {np.min(stochrsi_drops):.2f}%")
        print(f"  成功率(跌>1.5%): {sum(1 for d in stochrsi_drops if d < -1.5)/len(stochrsi_drops)*100:.1f}%")
    
    # 可視化比較
    plt.figure(figsize=(20, 15))
    
    # 主圖：價格走勢 + 信號對比
    plt.subplot(4, 1, 1)
    plt.plot(df['timestamp'], df['close'], 'k-', linewidth=0.8, alpha=0.7, label='BTC價格')
    
    # RSI信號
    if rsi_signals:
        rsi_times = [s['timestamp'] for s in rsi_signals]
        rsi_prices = [s['price'] for s in rsi_signals]
        plt.scatter(rsi_times, rsi_prices, color='blue', s=50, marker='v', 
                   label=f'RSI策略 ({len(rsi_signals)}個)', alpha=0.7, zorder=5)
    
    # StochRSI信號
    if stochrsi_signals:
        stochrsi_times = [s['timestamp'] for s in stochrsi_signals]
        stochrsi_prices = [s['price'] for s in stochrsi_signals]
        plt.scatter(stochrsi_times, stochrsi_prices, color='red', s=50, marker='o',
                   label=f'StochRSI策略 ({len(stochrsi_signals)}個)', alpha=0.7, zorder=5)
    
    plt.title('RSI vs StochRSI 早期下跌檢測策略比較', fontsize=16)
    plt.ylabel('價格 (USDT)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # RSI指標
    plt.subplot(4, 1, 2)
    plt.plot(df['timestamp'], df['rsi'], 'b-', linewidth=0.8, label='RSI')
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    
    if rsi_signals:
        for signal in rsi_signals:
            idx = df[df['timestamp'] == signal['timestamp']].index[0]
            plt.scatter(signal['timestamp'], df.iloc[idx]['rsi'], color='blue', s=30, zorder=5)
    
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # StochRSI指標
    plt.subplot(4, 1, 3)
    plt.plot(df['timestamp'], df['stochrsi_k'], 'r-', linewidth=0.8, label='StochRSI %K')
    plt.plot(df['timestamp'], df['stochrsi_d'], 'orange', linewidth=0.8, label='StochRSI %D')
    plt.axhline(y=80, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=20, color='g', linestyle='--', alpha=0.5)
    
    if stochrsi_signals:
        for signal in stochrsi_signals:
            idx = df[df['timestamp'] == signal['timestamp']].index[0]
            plt.scatter(signal['timestamp'], df.iloc[idx]['stochrsi_k'], color='red', s=30, zorder=5)
    
    plt.ylabel('StochRSI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 成交量
    plt.subplot(4, 1, 4)
    plt.plot(df['timestamp'], df['volume_ratio'], 'purple', linewidth=0.8, label='成交量倍數')
    plt.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)
    plt.axhline(y=1.2, color='orange', linestyle='--', alpha=0.5, label='觸發閾值')
    
    plt.ylabel('成交量倍數')
    plt.xlabel('時間')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 格式化x軸
    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('data/rsi_vs_stochrsi_crash_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 信號重疊分析
    if rsi_signals and stochrsi_signals:
        rsi_times_set = set(s['timestamp'] for s in rsi_signals)
        stochrsi_times_set = set(s['timestamp'] for s in stochrsi_signals)
        
        overlap = rsi_times_set.intersection(stochrsi_times_set)
        rsi_only = rsi_times_set - stochrsi_times_set
        stochrsi_only = stochrsi_times_set - rsi_times_set
        
        print(f"\n=== 信號重疊分析 ===")
        print(f"重疊信號: {len(overlap)}個")
        print(f"僅RSI信號: {len(rsi_only)}個")
        print(f"僅StochRSI信號: {len(stochrsi_only)}個")
        print(f"重疊率: {len(overlap)/max(len(rsi_times_set), len(stochrsi_times_set))*100:.1f}%")
    
    return rsi_signals, stochrsi_signals

if __name__ == "__main__":
    rsi_signals, stochrsi_signals = compare_rsi_stochrsi_crash()
