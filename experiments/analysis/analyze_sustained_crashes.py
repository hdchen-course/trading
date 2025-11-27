#!/usr/bin/env python3
"""
分析持續性大跌模式
重點：(1) 大跌前預警信號 (2) 早期階段判斷是否為持續下跌
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from crypto_provider import BinanceDataProvider
import talib

def analyze_sustained_crashes():
    """分析持續性大跌特徵"""
    
    # 持續下跌起始時間 (UTC+8) 和估計持續時間
    crash_events = [
        {"start": "2025-11-13 22:00", "duration_hours": 33},  # 到11/15 07:00
        {"start": "2025-11-12 22:00", "duration_hours": 3},   # 連續跌三小時
        {"start": "2025-11-11 09:00", "duration_hours": 8},   # 連續下跌
        {"start": "2025-11-10 22:00", "duration_hours": 1},   # 一小時巨幅下跌
        {"start": "2025-11-08 14:00", "duration_hours": 6},   # 連續下跌
        {"start": "2025-11-03 08:00", "duration_hours": 4},   # 連續下跌
        {"start": "2025-11-03 23:00", "duration_hours": 2},   # 大跌
        {"start": "2025-11-04 13:00", "duration_hours": 2},   # 大跌
        {"start": "2025-10-30 16:00", "duration_hours": 6},   # 連續下跌
        {"start": "2025-10-30 13:00", "duration_hours": 2},   # 大跌
        {"start": "2025-10-28 02:00", "duration_hours": 8},   # 連續下跌
        {"start": "2025-10-22 00:00", "duration_hours": 12},  # 連續大幅下跌
        {"start": "2025-10-21 07:00", "duration_hours": 6},   # 連續下跌
        {"start": "2025-10-17 12:00", "duration_hours": 8},   # 連續下跌
        {"start": "2025-10-16 21:00", "duration_hours": 10},  # 連續大跌
        {"start": "2025-10-15 17:00", "duration_hours": 6},   # 連續下跌
        {"start": "2025-10-14 06:00", "duration_hours": 8},   # 連續下跌好幾個小時
        {"start": "2025-10-10 22:00", "duration_hours": 6},   # 連續下跌
    ]
    
    provider = BinanceDataProvider()
    
    # 獲取足夠的數據
    start_date = datetime.strptime("2025-10-01", "%Y-%m-%d")
    days_needed = (datetime.now() - start_date).days + 1
    df = provider.get_data("BTCUSDT", "1h", days_needed * 24)
    
    if df.empty:
        print("❌ 無法獲取數據")
        return
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 計算技術指標
    df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'].values)
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'].values, timeperiod=20)
    df['volume_ma'] = df['volume'].rolling(24).mean()
    df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
    
    # 價格變化率
    for period in [1, 2, 4, 6, 12, 24]:
        df[f'return_{period}h'] = df['close'].pct_change(period) * 100
    
    print(f"📊 分析 {len(crash_events)} 個持續下跌事件")
    
    analysis_results = []
    
    for i, event in enumerate(crash_events):
        print(f"\n=== 事件 #{i+1}: {event['start']} (持續{event['duration_hours']}小時) ===")
        
        # 轉換為UTC時間
        crash_time = datetime.strptime(event['start'], "%Y-%m-%d %H:%M") - timedelta(hours=8)
        
        # 找到最接近的時間點
        time_diff = abs(df['timestamp'] - crash_time)
        crash_idx = time_diff.idxmin()
        
        if crash_idx < 24 or crash_idx + event['duration_hours'] >= len(df):
            continue
        
        crash_row = df.loc[crash_idx]
        print(f"開始時間: {crash_row['timestamp']} UTC")
        print(f"開始價格: ${crash_row['close']:.2f}")
        
        # 計算實際跌幅
        end_idx = min(crash_idx + event['duration_hours'], len(df) - 1)
        lowest_price = df.loc[crash_idx:end_idx]['low'].min()
        total_drop = (lowest_price - crash_row['close']) / crash_row['close'] * 100
        
        print(f"最大跌幅: {total_drop:.2f}% (持續{event['duration_hours']}小時)")
        
        # === (1) 分析大跌前的預警信號 ===
        pre_signals = {}
        
        for hours_before in [1, 2, 4, 6]:
            if crash_idx - hours_before >= 0:
                before_idx = crash_idx - hours_before
                before_row = df.loc[before_idx]
                
                # RSI異常
                pre_signals[f'rsi_{hours_before}h_before'] = before_row['rsi']
                
                # MACD背離
                pre_signals[f'macd_{hours_before}h_before'] = before_row['macd']
                pre_signals[f'macd_hist_{hours_before}h_before'] = before_row['macd_hist']
                
                # 成交量異常
                vol_ratio = before_row['volume'] / before_row['volume_ma'] if before_row['volume_ma'] > 0 else 1
                pre_signals[f'volume_ratio_{hours_before}h_before'] = vol_ratio
                
                # 布林帶位置
                bb_pos = (before_row['close'] - before_row['bb_lower']) / (before_row['bb_upper'] - before_row['bb_lower'])
                pre_signals[f'bb_position_{hours_before}h_before'] = bb_pos
                
                # 價格動能
                pre_signals[f'return_1h_{hours_before}h_before'] = before_row['return_1h']
                pre_signals[f'return_4h_{hours_before}h_before'] = before_row['return_4h']
        
        # === (2) 分析早期下跌階段特徵 ===
        early_signals = {}
        
        # 下跌開始後1-2小時的特徵
        for hours_after in [1, 2]:
            if crash_idx + hours_after < len(df):
                after_idx = crash_idx + hours_after
                after_row = df.loc[after_idx]
                
                # 跌幅速度
                drop_speed = (after_row['close'] - crash_row['close']) / crash_row['close'] * 100
                early_signals[f'drop_speed_{hours_after}h'] = drop_speed
                
                # 成交量持續性
                vol_ratio = after_row['volume'] / after_row['volume_ma'] if after_row['volume_ma'] > 0 else 1
                early_signals[f'volume_sustained_{hours_after}h'] = vol_ratio
                
                # RSI下降速度
                rsi_drop = crash_row['rsi'] - after_row['rsi']
                early_signals[f'rsi_drop_speed_{hours_after}h'] = rsi_drop
                
                # 反彈力度（是否有明顯反彈）
                rebound = (after_row['high'] - after_row['low']) / after_row['low'] * 100
                early_signals[f'rebound_strength_{hours_after}h'] = rebound
        
        # 合併結果
        result = {
            'event_id': i + 1,
            'start_time': event['start'],
            'duration_hours': event['duration_hours'],
            'total_drop': total_drop,
            **pre_signals,
            **early_signals
        }
        
        analysis_results.append(result)
        
        # 顯示關鍵特徵
        if crash_idx - 1 >= 0:
            before_1h = df.loc[crash_idx - 1]
            print(f"大跌前1小時預警 - RSI: {before_1h['rsi']:.1f}, 成交量: {before_1h['volume']/before_1h['volume_ma']:.1f}x, MACD: {before_1h['macd']:.3f}")
        
        if crash_idx + 1 < len(df):
            after_1h = df.loc[crash_idx + 1]
            drop_1h = (after_1h['close'] - crash_row['close']) / crash_row['close'] * 100
            print(f"開始1小時後 - 跌幅: {drop_1h:.2f}%, RSI: {after_1h['rsi']:.1f}")
    
    # 統計分析
    if analysis_results:
        results_df = pd.DataFrame(analysis_results)
        
        print(f"\n=== 預警信號統計分析 ===")
        
        # 按持續時間分類
        short_crashes = results_df[results_df['duration_hours'] <= 3]  # 短期下跌
        long_crashes = results_df[results_df['duration_hours'] > 6]    # 長期下跌
        
        print(f"短期下跌({len(short_crashes)}個): 持續≤3小時")
        print(f"長期下跌({len(long_crashes)}個): 持續>6小時")
        
        # 預警信號對比
        if len(short_crashes) > 0 and len(long_crashes) > 0:
            print(f"\n大跌前1小時RSI對比:")
            print(f"短期下跌 - 平均RSI: {short_crashes['rsi_1h_before'].mean():.1f}")
            print(f"長期下跌 - 平均RSI: {long_crashes['rsi_1h_before'].mean():.1f}")
            
            print(f"\n大跌前1小時成交量對比:")
            print(f"短期下跌 - 平均成交量倍數: {short_crashes['volume_ratio_1h_before'].mean():.1f}x")
            print(f"長期下跌 - 平均成交量倍數: {long_crashes['volume_ratio_1h_before'].mean():.1f}x")
            
            print(f"\n早期跌幅速度對比:")
            print(f"短期下跌 - 1小時跌幅: {short_crashes['drop_speed_1h'].mean():.2f}%")
            print(f"長期下跌 - 1小時跌幅: {long_crashes['drop_speed_1h'].mean():.2f}%")
        
        return results_df
    
    return None

if __name__ == "__main__":
    results = analyze_sustained_crashes()
