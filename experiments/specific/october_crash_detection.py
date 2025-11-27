#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import talib
import numpy as np

plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']

def calculate_sar(high, low, acceleration=0.02, maximum=0.2):
    """计算抛物线SAR指标"""
    return talib.SAR(high, low, acceleration=acceleration, maximum=maximum)

def detect_october_warnings():
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 計算技術指標
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['sar'] = calculate_sar(df['high'].values, df['low'].values)
    
    # 計算布林通道
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=48, nbdevup=2.4, nbdevdn=2.4, matype=0)
    
    # 按月份分組
    for month in range(1, 13):
        october_df = df[df['timestamp'].dt.month == month].copy()
        
        if len(october_df) == 0:
            continue
            
        print(f"=== {month}月 Enhanced Reversal Analysis ===")
        print(f"Total data points: {len(october_df)}")
        
        # 檢測反轉點訊號
        warning_points = []
        
        # 檢測條件 - 從第192個開始，到倒數第8個結束 (需要2小時後的數據)
        for i in range(192, len(october_df) - 8):
            current_row = october_df.iloc[i]
            current_price = current_row['close']
            
            # 一小時前的數據
            one_hour_ago_idx = i - 4
            one_hour_ago_row = october_df.iloc[one_hour_ago_idx]
            one_hour_ago_price = one_hour_ago_row['close']
            
            # 計算當前ATR
            current_atr = current_row['atr']
            
            # 條件1: 目前K線比一小時前K線價格低，但是不能低超過ATR
            price_drop = (current_price - one_hour_ago_price) / one_hour_ago_price
            max_drop = -current_atr / current_price
            
            if price_drop >= 0 or price_drop < max_drop:
                continue
            
            # 使用一小時前的數據作為檢測基準點
            check_idx = one_hour_ago_idx
            check_row = october_df.iloc[check_idx]
            check_price = check_row['close']
            check_atr = check_row['atr']
            check_volume = check_row['volume']
            
            # 條件2: 一小時前的K線比四小時前高2倍ATR
            if check_idx >= 16:
                four_hours_ago_price = october_df.iloc[check_idx - 16]['close']
                rise_4h = (check_price - four_hours_ago_price) / four_hours_ago_price
                required_rise_4h = 2 * check_atr / check_price
                if rise_4h <= required_rise_4h:
                    continue
            else:
                continue
            
            # 條件3: 一小時前的K線比24小時前高4倍ATR
            if check_idx >= 96:
                twenty_four_hours_ago_price = october_df.iloc[check_idx - 96]['close']
                rise_24h = (check_price - twenty_four_hours_ago_price) / twenty_four_hours_ago_price
                required_rise_24h = 4 * check_atr / check_price
                if rise_24h <= required_rise_24h:
                    continue
            else:
                continue
            
            # 條件4: 一小時前的K線比48小時前高6倍ATR
            if check_idx >= 192:
                forty_eight_hours_ago_price = october_df.iloc[check_idx - 192]['close']
                rise_48h = (check_price - forty_eight_hours_ago_price) / forty_eight_hours_ago_price
                required_rise_48h = 6 * check_atr / check_price
                if rise_48h <= required_rise_48h:
                    continue
            else:
                continue
            
            # 條件5: 一小時前K線成交量比均量大1.2倍以上
            if check_idx >= 96:
                past_24h_volumes = [october_df.iloc[j]['volume'] for j in range(check_idx-96, check_idx)]
                avg_volume_24h = sum(past_24h_volumes) / len(past_24h_volumes)
                if check_volume <= 1.2 * avg_volume_24h:
                    continue
            else:
                continue
            
            # 檢查2小時後價格下跌條件 (8個15分鐘K線後)
            two_hours_later_idx = i + 8
            if two_hours_later_idx < len(october_df):
                two_hours_later_price = october_df.iloc[two_hours_later_idx]['close']
                price_drop_2h = (two_hours_later_price - current_price) / current_price
                required_drop_2h = -2 * current_atr / current_price
                
                # 條件6: 2小時後價格下跌2倍ATR以上
                if price_drop_2h > required_drop_2h:
                    continue
                    
                # 所有條件都滿足，記錄警告點
                warning_points.append({
                    'timestamp': current_row['timestamp'],  # 當前時間
                    'price': current_price,  # 當前價格
                    'check_timestamp': check_row['timestamp'],  # 檢測時間點 (一小時前)
                    'check_price': check_price,  # 檢測價格
                    'forty_eight_hours_ago_price': forty_eight_hours_ago_price,
                    'twenty_four_hours_ago_price': twenty_four_hours_ago_price,
                    'four_hours_ago_price': four_hours_ago_price,
                    'price_rise_48h': rise_48h * 100,
                    'price_rise_24h': rise_24h * 100,
                    'price_rise_4h': rise_4h * 100,
                    'atr': check_atr,
                    'volume_ratio': check_volume / avg_volume_24h,
                    'price_drop_1h': price_drop * 100,
                    'price_drop_2h': price_drop_2h * 100,
                    'max_allowed_drop': max_drop * 100
                })
            else:
                continue
        
        # 統計信息
        if warning_points:
            avg_price_rise_48h = sum(w['price_rise_48h'] for w in warning_points) / len(warning_points)
            avg_price_rise_24h = sum(w['price_rise_24h'] for w in warning_points) / len(warning_points)
            avg_price_rise_4h = sum(w['price_rise_4h'] for w in warning_points) / len(warning_points)
            avg_volume_ratio = sum(w['volume_ratio'] for w in warning_points) / len(warning_points)
            avg_price_drop_1h = sum(w['price_drop_1h'] for w in warning_points) / len(warning_points)
            avg_price_drop_2h = sum(w['price_drop_2h'] for w in warning_points) / len(warning_points)
            
            print(f"Signals: {len(warning_points)}, 48h rise: {avg_price_rise_48h:.1f}%, 24h rise: {avg_price_rise_24h:.1f}%, 4h rise: {avg_price_rise_4h:.1f}%, VolumeRatio×{avg_volume_ratio:.1f}, 1h drop: {avg_price_drop_1h:.1f}%, 2h drop: {avg_price_drop_2h:.1f}%")
        else:
            print(f"SAR reversal signals detected: 0")
        
        if len(warning_points) == 0:
            print(f"SAR reversal signals detected: 0")
            continue
            
        # 繪圖
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        # 灰色價格線
        ax.plot(october_df['timestamp'], october_df['close'], color='lightgray', linewidth=1, alpha=0.8)
        
        # 紅色警告點
        warning_times = [w['timestamp'] for w in warning_points]
        warning_prices = [w['price'] for w in warning_points]
        ax.scatter(warning_times, warning_prices, color='red', s=50, alpha=0.8, zorder=5)
        
        ax.set_title(f'{month}月 BTC: SAR區間反轉信號 (前4K>SAR + 後4K<SAR + 4h漲4×ATR)', fontsize=14)
        ax.set_ylabel('Price (USDT)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 格式化x軸
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'month_{month:02d}_enhanced_reversal.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    detect_october_warnings()
