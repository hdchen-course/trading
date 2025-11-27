#!/usr/bin/env python3
"""
分析ReversalStrategy錯過的潛在收益
"""

import pandas as pd
import numpy as np


def analyze_missed_profits():
    # 載入數據
    df = pd.read_csv("training_data/btc_12months_15m.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 交易記錄
    trades = [
        {
            "entry_time": "2025-11-05 01:15:00",
            "exit_time": "2025-11-05 01:45:00",
            "entry_price": 99113.39,
            "exit_price": 100104.52,
            "type": "long",
        },
        {
            "entry_time": "2025-11-06 15:15:00",
            "exit_time": "2025-11-06 16:30:00",
            "entry_price": 102438.02,
            "exit_price": 100901.45,
            "type": "short",
        },
    ]

    print("=== 分析錯過的潛在收益 ===\n")

    for i, trade in enumerate(trades, 1):
        entry_time = pd.to_datetime(trade["entry_time"])
        exit_time = pd.to_datetime(trade["exit_time"])
        entry_price = trade["entry_price"]
        exit_price = trade["exit_price"]
        trade_type = trade["type"]

        print(f"📊 交易 {i} ({trade_type.upper()}):")
        print(f"   進場: {entry_time} @ {entry_price:.2f}")
        print(f"   出場: {exit_time} @ {exit_price:.2f}")

        # 實際收益
        if trade_type == "long":
            actual_return = (exit_price - entry_price) / entry_price
        else:
            actual_return = (entry_price - exit_price) / entry_price
        print(f"   實際收益: {actual_return*100:.2f}%")

        # 分析後續價格走勢
        exit_idx = df[df["timestamp"] == exit_time].index[0]

        # 檢查後續24小時的價格
        future_24h = df.iloc[exit_idx : exit_idx + 96]  # 24小時 = 96個15分鐘

        if trade_type == "long":
            max_price = future_24h["high"].max()
            potential_return = (max_price - entry_price) / entry_price
            missed_profit = potential_return - actual_return
            print(f"   24小時內最高價: {max_price:.2f}")
            print(f"   潛在最大收益: {potential_return*100:.2f}%")
            print(f"   錯過收益: {missed_profit*100:.2f}%")

            # 找到最高價的時間
            max_idx = future_24h["high"].idxmax()
            max_time = df.iloc[max_idx]["timestamp"]
            print(f"   最高價時間: {max_time}")

        else:  # short
            min_price = future_24h["low"].min()
            potential_return = (entry_price - min_price) / entry_price
            missed_profit = potential_return - actual_return
            print(f"   24小時內最低價: {min_price:.2f}")
            print(f"   潛在最大收益: {potential_return*100:.2f}%")
            print(f"   錯過收益: {missed_profit*100:.2f}%")

            # 找到最低價的時間
            min_idx = future_24h["low"].idxmin()
            min_time = df.iloc[min_idx]["timestamp"]
            print(f"   最低價時間: {min_time}")

        print()


if __name__ == "__main__":
    analyze_missed_profits()
