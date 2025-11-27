#!/usr/bin/env python3
"""
調試移動停利邏輯
"""

import pandas as pd


def debug_trailing_stop():
    # 載入數據
    df = pd.read_csv("training_data/btc_12months_15m.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 第一筆交易: 2025-11-05 01:15 進場 @ 99113.39, 01:45 出場 @ 100104.52
    entry_time = pd.to_datetime("2025-11-05 01:15:00")
    exit_time = pd.to_datetime("2025-11-05 01:45:00")
    entry_price = 99113.39
    exit_price = 100104.52

    # 找到交易期間的數據
    entry_idx = df[df["timestamp"] == entry_time].index[0]
    exit_idx = df[df["timestamp"] == exit_time].index[0]

    trade_data = df.iloc[entry_idx : exit_idx + 1].copy()

    print("=== 第一筆交易移動停利調試 ===")
    print(f"進場: {entry_time} @ {entry_price}")
    print(f"出場: {exit_time} @ {exit_price}")
    print(f"實際收益: {(exit_price - entry_price) / entry_price * 100:.2f}%")
    print()

    # 模擬移動停利邏輯
    original_take_profit = 0.01  # 1%
    x = original_take_profit
    data = [
        x * 0.25,
        x * 0.5,
        x * 1,
        x * 1.5,
        x * 2,
        x * 2.5,
        x * 3,
        x * 3.5,
        x * 4,
        x * 4.5,
        x * 5,
        x * 6,
        x * 7,
        x * 8,
        x * 9,
        x * 10,
        x * 11,
        x * 12,
        x * 13,
        x * 14,
        x * 15,
        x * 16,
        x * 17,
        x * 18,
        x * 19,
        x * 20,
        x * 21,
        x * 22,
        x * 23,
        x * 24,
    ]

    print("移動停利階梯:")
    for i, level in enumerate(data[:10]):
        print(f"  Level {i}: {level*100:.2f}%")
    print()

    trailing_activated = False
    current_level = 0
    stop_loss_price = entry_price * 0.99  # 初始1%停損
    take_profit_price = entry_price * 1.01  # 初始1%停利

    print("價格變化過程:")
    for _, row in trade_data.iterrows():
        current_price = row["close"]
        profit_pct = (current_price - entry_price) / entry_price

        print(
            f"{row['timestamp']}: {current_price:.2f} (收益: {profit_pct*100:.2f}%)"
        )

        # 檢查是否達到0.5%觸發移動停損
        if profit_pct >= data[1] and not trailing_activated:  # 0.5%
            trailing_activated = True
            current_level = 1
            stop_loss_price = entry_price * (1 + data[0])  # 0.25%
            take_profit_price = entry_price * (1 + data[3])  # 1.5%
            print(
                f"  ✅ 觸發移動停利! 停損: {stop_loss_price:.2f}, 停利: {take_profit_price:.2f}"
            )

        # 檢查是否觸發停利
        if current_price >= take_profit_price:
            print(
                f"  🎯 觸發停利! 價格 {current_price:.2f} >= 停利 {take_profit_price:.2f}"
            )
            break

        print(
            f"  當前停損: {stop_loss_price:.2f}, 停利: {take_profit_price:.2f}"
        )
        print()


if __name__ == "__main__":
    debug_trailing_stop()
