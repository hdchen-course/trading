#!/usr/bin/env python3
"""
抓取Binance BTC 12個月15分K線數據
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import os


def fetch_binance_klines(symbol, interval, start_time, end_time):
    """從Binance API獲取K線數據"""
    url = "https://api.binance.com/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_time.timestamp() * 1000),
        "endTime": int(end_time.timestamp() * 1000),
        "limit": 1000,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ API請求失敗: {e}")
        return None


def fetch_12months_btc_data():
    """抓取12個月的BTC 15分K線數據"""
    print("🚀 開始抓取BTC 12個月15分K線數據...")

    end_time = datetime.now()
    start_time = end_time - timedelta(days=365)

    print(
        f"📅 時間範圍: {start_time.strftime('%Y-%m-%d')} 到 {end_time.strftime('%Y-%m-%d')}"
    )

    all_data = []
    current_time = start_time
    batch_size = timedelta(days=10)
    batch_count = 0

    while current_time < end_time:
        batch_end = min(current_time + batch_size, end_time)

        print(
            f"📊 抓取批次 {batch_count + 1}: {current_time.strftime('%Y-%m-%d')} 到 {batch_end.strftime('%Y-%m-%d')}"
        )

        klines = fetch_binance_klines(
            "BTCUSDT", "15m", current_time, batch_end
        )

        if klines:
            all_data.extend(klines)
            print(f"✅ 成功抓取 {len(klines)} 條記錄")
        else:
            print("❌ 抓取失敗，跳過此批次")

        current_time = batch_end
        batch_count += 1
        time.sleep(0.1)

    if not all_data:
        print("❌ 沒有抓取到任何數據")
        return False

    print(f"🔄 處理 {len(all_data)} 條原始數據...")

    df = pd.DataFrame(
        all_data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    os.makedirs("data", exist_ok=True)

    filename = "data/btc_12months_15m.csv"
    df.to_csv(filename, index=False)

    print(f"💾 數據已保存至: {filename}")
    print(f"📊 總記錄數: {len(df)}")
    print(f"📅 數據範圍: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"💰 價格範圍: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return True


if __name__ == "__main__":
    success = fetch_12months_btc_data()

    if success:
        print("\n✅ BTC 12個月數據抓取完成！")
        print("🎯 現在可以使用此數據進行LSTM訓練")
    else:
        print("\n❌ 數據抓取失敗")
