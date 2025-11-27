#!/usr/bin/env python3
"""
可視化持續下跌檢測策略
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import talib

def visualize_crash_detection():
    """可視化持續下跌檢測策略"""
    
    # 讀取12個月的15分鐘訓練數據
    try:
        df = pd.read_csv('training_data/btc_12months_15m.csv')
        print(f"✅ 載入15分鐘數據: {len(df)} 根K線")
    except FileNotFoundError:
        print("❌ 找不到 training_data/btc_12months_15m.csv")
        return
    
    # 數據預處理
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 計算技術指標（基於15分鐘）
    df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'].values)
    df['volume_ma'] = df['volume'].rolling(96).mean()  # 24小時 = 96個15分鐘
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['return_15m'] = df['close'].pct_change() * 100
    df['return_1h'] = df['close'].pct_change(4) * 100  # 4個15分鐘 = 1小時
    
    # 策略1: 長期下跌預警信號（調整為15分鐘參數）
    strategy1_signals = []
    
    # 策略2: 早期階段判斷信號（調整為15分鐘參數）
    strategy2_signals = []
    
    for i in range(96, len(df) - 96):  # 需要前後24小時數據（96個15分鐘）
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # 策略1: 長期下跌預警（15分鐘版本）
        # RSI > 50 且 MACD轉負 且 成交量正常
        if (current['rsi'] > 50 and 
            current['macd'] < 0 and prev['macd'] >= 0 and  # MACD剛轉負
            0.8 <= current['volume_ratio'] <= 1.5):  # 成交量正常
            
            # 檢查後續24小時是否真的持續下跌
            future_24h = df.iloc[i:i+96]  # 96個15分鐘 = 24小時
            if len(future_24h) >= 96:
                lowest_price = future_24h['low'].min()
                drop_pct = (lowest_price - current['close']) / current['close'] * 100
                
                if drop_pct < -2:  # 24小時內跌超過2%
                    strategy1_signals.append({
                        'timestamp': current['timestamp'],
                        'price': current['close'],
                        'drop_pct': drop_pct,
                        'type': 'strategy1'
                    })
        
        # 策略2: 早期階段判斷（15分鐘版本）
        # 價格開始下跌且RSI快速下降
        if (current['return_1h'] < -0.5 and  # 1小時跌超過0.5%
            current['rsi'] < prev['rsi'] - 3 and  # RSI下降（調整為15分鐘適合的幅度）
            current['volume_ratio'] > 1.2):  # 成交量放大
            
            # 檢查後續6小時是否持續下跌
            future_6h = df.iloc[i:i+24]  # 24個15分鐘 = 6小時
            if len(future_6h) >= 24:
                lowest_price = future_6h['low'].min()
                drop_pct = (lowest_price - current['close']) / current['close'] * 100
                
                if drop_pct < -1.5:  # 6小時內繼續跌1.5%
                    strategy2_signals.append({
                        'timestamp': current['timestamp'],
                        'price': current['close'],
                        'drop_pct': drop_pct,
                        'type': 'strategy2'
                    })
    
    print(f"策略1信號數量: {len(strategy1_signals)}")
    print(f"策略2信號數量: {len(strategy2_signals)}")
    
    # 繪圖
    plt.figure(figsize=(20, 12))
    
    # 主圖：價格走勢
    plt.subplot(3, 1, 1)
    
    # 先畫完整的黑色價格線
    plt.plot(df['timestamp'], df['close'], 'k-', linewidth=0.8, alpha=0.7, label='BTC價格(15分鐘)')
    
    # 收集所有需要標記的時間段
    marked_segments = set()
    
    # 策略1信號的時間段（24小時）
    if strategy1_signals:
        for signal in strategy1_signals:
            start_idx = df[df['timestamp'] == signal['timestamp']].index[0]
            end_idx = min(start_idx + 96, len(df) - 1)  # 96個15分鐘 = 24小時
            for i in range(start_idx, end_idx + 1):
                marked_segments.add((i, 'strategy1'))
    
    # 策略2信號的時間段（6小時）
    if strategy2_signals:
        for signal in strategy2_signals:
            start_idx = df[df['timestamp'] == signal['timestamp']].index[0]
            end_idx = min(start_idx + 24, len(df) - 1)  # 24個15分鐘 = 6小時
            for i in range(start_idx, end_idx + 1):
                if (i, 'strategy1') not in marked_segments:  # 避免重疊
                    marked_segments.add((i, 'strategy2'))
    
    # 按策略類型分組並畫彩色線段
    current_segment = []
    current_type = None
    
    for i in range(len(df)):
        # 檢查當前點是否需要標記
        point_type = None
        for idx, seg_type in marked_segments:
            if idx == i:
                point_type = seg_type
                break
        
        if point_type == current_type:
            # 繼續當前線段
            current_segment.append(i)
        else:
            # 結束當前線段並開始新線段
            if current_segment and current_type:
                segment_data = df.iloc[current_segment]
                if current_type == 'strategy1':
                    plt.plot(segment_data['timestamp'], segment_data['close'], 
                            'r-', linewidth=3, alpha=0.8, zorder=3)
                elif current_type == 'strategy2':
                    plt.plot(segment_data['timestamp'], segment_data['close'], 
                            'orange', linewidth=3, alpha=0.8, zorder=3)
            
            # 開始新線段
            current_segment = [i] if point_type else []
            current_type = point_type
    
    # 處理最後一個線段
    if current_segment and current_type:
        segment_data = df.iloc[current_segment]
        if current_type == 'strategy1':
            plt.plot(segment_data['timestamp'], segment_data['close'], 
                    'r-', linewidth=3, alpha=0.8, zorder=3)
        elif current_type == 'strategy2':
            plt.plot(segment_data['timestamp'], segment_data['close'], 
                    'orange', linewidth=3, alpha=0.8, zorder=3)
    
    # 添加圖例
    legend_elements = [plt.Line2D([0], [0], color='k', linewidth=0.8, alpha=0.7, label='BTC價格(15分鐘)')]
    
    if strategy1_signals:
        legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=3, 
                                        label=f'策略1: 長期下跌預警 ({len(strategy1_signals)}個)'))
    
    if strategy2_signals:
        legend_elements.append(plt.Line2D([0], [0], color='orange', linewidth=3,
                                        label=f'策略2: 早期階段判斷 ({len(strategy2_signals)}個)'))
    
    plt.legend(handles=legend_elements)
    
    plt.title('BTC 12個月走勢 + 持續下跌檢測策略', fontsize=16)
    plt.ylabel('價格 (USDT)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子圖1：RSI
    plt.subplot(3, 1, 2)
    plt.plot(df['timestamp'], df['rsi'], 'b-', linewidth=0.8, label='RSI')
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    plt.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
    
    # 標記信號點的RSI
    if strategy1_signals:
        for signal in strategy1_signals:
            idx = df[df['timestamp'] == signal['timestamp']].index[0]
            plt.scatter(signal['timestamp'], df.iloc[idx]['rsi'], color='red', s=50, zorder=5)
    
    if strategy2_signals:
        for signal in strategy2_signals:
            idx = df[df['timestamp'] == signal['timestamp']].index[0]
            plt.scatter(signal['timestamp'], df.iloc[idx]['rsi'], color='orange', s=40, zorder=5)
    
    plt.ylabel('RSI', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子圖2：成交量
    plt.subplot(3, 1, 3)
    plt.plot(df['timestamp'], df['volume_ratio'], 'purple', linewidth=0.8, label='成交量倍數')
    plt.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)
    plt.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5)
    
    # 標記信號點的成交量
    if strategy1_signals:
        for signal in strategy1_signals:
            idx = df[df['timestamp'] == signal['timestamp']].index[0]
            plt.scatter(signal['timestamp'], df.iloc[idx]['volume_ratio'], color='red', s=50, zorder=5)
    
    if strategy2_signals:
        for signal in strategy2_signals:
            idx = df[df['timestamp'] == signal['timestamp']].index[0]
            plt.scatter(signal['timestamp'], df.iloc[idx]['volume_ratio'], color='orange', s=40, zorder=5)
    
    plt.ylabel('成交量倍數', fontsize=12)
    plt.xlabel('時間', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 格式化x軸
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('data/crash_detection_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 統計結果
    print(f"\n=== 策略效果統計 ===")
    
    if strategy1_signals:
        s1_drops = [s['drop_pct'] for s in strategy1_signals]
        print(f"策略1 - 平均後續跌幅: {np.mean(s1_drops):.2f}%")
        print(f"策略1 - 最大後續跌幅: {np.min(s1_drops):.2f}%")
        print(f"策略1 - 成功率(跌>2%): {sum(1 for d in s1_drops if d < -2)/len(s1_drops)*100:.1f}%")
    
    if strategy2_signals:
        s2_drops = [s['drop_pct'] for s in strategy2_signals]
        print(f"策略2 - 平均後續跌幅: {np.mean(s2_drops):.2f}%")
        print(f"策略2 - 最大後續跌幅: {np.min(s2_drops):.2f}%")
        print(f"策略2 - 成功率(跌>1.5%): {sum(1 for d in s2_drops if d < -1.5)/len(s2_drops)*100:.1f}%")

if __name__ == "__main__":
    visualize_crash_detection()
