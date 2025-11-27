import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def calculate_atr(df, period=14):
    """計算ATR (Average True Range)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def detect_price_spikes():
    """檢測價格突破條件"""
    
    # 讀取完整數據
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['atr'] = calculate_atr(df)
    
    # 按月份分組分析
    for month in range(1, 13):
        month_df = df[df['timestamp'].dt.month == month].copy()
        
        if len(month_df) == 0:
            continue
            
        month_df = month_df.reset_index(drop=True)
        
        print(f"=== {month}月 Price Spike Analysis ===")
        print(f"Total data points: {len(month_df)}")
        
        warning_points = []
        
        # 檢測條件 - 需要前96小時和後4小時的數據 (384個和16個15分鐘K線)
        for i in range(384, len(month_df) - 16):
            current_row = month_df.iloc[i]
            current_price = current_row['close']
            current_atr = current_row['atr']
            
            if pd.isna(current_atr) or current_atr <= 0:
                continue
            
            # 96小時前的價格 (384個15分鐘K線前)
            ninety_six_hours_ago_price = month_df.iloc[i - 384]['close']
            
            # 4小時前的價格 (16個15分鐘K線前)
            four_hours_ago_price = month_df.iloc[i - 16]['close']
            
            # 4小時後的價格 (16個15分鐘K線後)
            four_hours_later_price = month_df.iloc[i + 16]['close']
            
            # 條件1: 目前價格高於96小時前3倍ATR
            rise_from_96h_ago = (current_price - ninety_six_hours_ago_price) / ninety_six_hours_ago_price
            required_rise_96h_ago = 3 * current_atr / current_price
            
            if rise_from_96h_ago <= required_rise_96h_ago:
                continue
                
            # 條件2: 目前價格高於4小時前
            if current_price <= four_hours_ago_price:
                continue
                
            # 條件3: 目前價格高於4小時後3倍ATR
            rise_vs_4h_later = (current_price - four_hours_later_price) / current_price
            required_rise_4h_later = 3 * current_atr / current_price
            
            if rise_vs_4h_later <= required_rise_4h_later:
                continue
            
            # 所有條件都滿足，記錄突破點
            warning_points.append({
                'timestamp': current_row['timestamp'],
                'price': current_price,
                'ninety_six_hours_ago_price': ninety_six_hours_ago_price,
                'four_hours_ago_price': four_hours_ago_price,
                'four_hours_later_price': four_hours_later_price,
                'rise_from_96h_ago': rise_from_96h_ago * 100,
                'rise_from_4h_ago': (current_price - four_hours_ago_price) / four_hours_ago_price * 100,
                'rise_vs_4h_later': rise_vs_4h_later * 100,
                'atr': current_atr,
                'atr_multiple_ago': rise_from_96h_ago / (current_atr / current_price),
                'atr_multiple_later': rise_vs_4h_later / (current_atr / current_price)
            })
        
        # 統計信息
        if warning_points:
            avg_rise_96h_ago = sum(w['rise_from_96h_ago'] for w in warning_points) / len(warning_points)
            avg_rise_4h_ago = sum(w['rise_from_4h_ago'] for w in warning_points) / len(warning_points)
            avg_rise_4h_later = sum(w['rise_vs_4h_later'] for w in warning_points) / len(warning_points)
            avg_atr_multiple_ago = sum(w['atr_multiple_ago'] for w in warning_points) / len(warning_points)
            avg_atr_multiple_later = sum(w['atr_multiple_later'] for w in warning_points) / len(warning_points)
            
            print(f"Signals: {len(warning_points)}, 96h ago: {avg_rise_96h_ago:.1f}%, 4h ago: {avg_rise_4h_ago:.1f}%, 4h later: {avg_rise_4h_later:.1f}%, ATR×{avg_atr_multiple_ago:.1f}, ATR×{avg_atr_multiple_later:.1f}")
        else:
            print(f"Price spike signals detected: 0")
        
        if len(warning_points) == 0:
            continue
            
        # 繪製圖表 (使用october_crash_detection.py風格)
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 繪製價格線
        ax.plot(month_df['timestamp'], month_df['close'], 'b-', alpha=0.7, linewidth=1)
        
        # 標記突破點
        for point in warning_points:
            ax.scatter(point['timestamp'], point['price'], color='red', s=100, alpha=0.8, zorder=5)
        
        ax.set_title(f'{month}月 Enhanced Price Spike Detection')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price (USDT)')
        ax.grid(True, alpha=0.3)
        
        # 格式化x軸
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'month_{month:02d}_enhanced_spikes.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    detect_price_spikes()
