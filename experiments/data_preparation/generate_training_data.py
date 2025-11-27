#!/usr/bin/env python3
import pandas as pd
import numpy as np
from crypto_util import calculate_technical_indicators, analyze_cumulative_metrics
import argparse

def load_data(start_date, end_date):
    """載入數據"""
    df = pd.read_csv("training_data/btc_15m_2022_2025.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # 過濾日期範圍
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
    
    if df.empty:
        return None
    
    # 計算技術指標
    df = calculate_technical_indicators(df)
    return df

def generate_signals(start_date, end_date):
    """生成信號數據"""
    print(f"Generating signals from {start_date} to {end_date}")
    
    # 載入數據
    df = load_data(start_date, end_date)
    if df is None or df.empty:
        print("No data found for the specified period")
        return pd.DataFrame()
    
    # 分析數據
    analysis_df = analyze_cumulative_metrics(df)
    
    if analysis_df is None or analysis_df.empty:
        print("No analysis data generated")
        return pd.DataFrame()
    
    # 條件2：MACD二階導數z-score從負轉正
    macd_current = analysis_df["macd_2nd"] > 0.5
    macd_prev = analysis_df["macd_2nd"].shift(1) > 0.5
    price_prev = analysis_df["price_2nd"].shift(1) > 0.5
    price_current = analysis_df["price_2nd"] < -0.5
    
    condition2 = macd_current & macd_prev & price_prev & price_current
    
    # 其他過濾條件
    price_above_ema = analysis_df["close_price"] > analysis_df["ema_50"]
    ema_bullish = analysis_df["ema_50"] > analysis_df["ema_100"]
    sar_bullish = analysis_df["price_above_sar_ratio"] > 0.6
    price_2nd_limit = analysis_df["price_2nd"] > -1.5
    
    # 組合條件
    signals = condition2 & price_above_ema & ema_bullish & sar_bullish & price_2nd_limit
    
    # 提取信號點
    signal_data = analysis_df[signals].copy()
    signal_data['signal'] = 1
    
    # 添加非信號點（隨機採樣）
    non_signals = analysis_df[~signals].sample(n=min(len(signal_data)*3, len(analysis_df[~signals])), random_state=42)
    non_signals['signal'] = 0
    
    # 合併數據
    training_data = pd.concat([signal_data, non_signals]).sort_index()
    
    print(f"Found {len(signal_data)} signals, {len(non_signals)} non-signals")
    return training_data

def main():
    # 生成訓練數據 (1-8月)
    train_data = generate_signals("2025-01-01", "2025-08-31")
    if not train_data.empty:
        train_data.to_csv("training_data.csv", index=False)
        print(f"Training data saved: {len(train_data)} records")
    
    # 生成驗證數據 (9-10月)
    val_data = generate_signals("2025-09-01", "2025-10-31")
    if not val_data.empty:
        val_data.to_csv("validation_data.csv", index=False)
        print(f"Validation data saved: {len(val_data)} records")

if __name__ == "__main__":
    main()
