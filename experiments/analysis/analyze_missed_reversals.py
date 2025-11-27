#!/usr/bin/env python3
"""
分析遺漏的重要反轉點並改進檢測算法
"""

import pandas as pd
import numpy as np


def analyze_missed_points():
    df = pd.read_csv("training_data/btc_12months_15m.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("=== 分析遺漏的重要反轉點 ===\n")

    # 重要的遺漏點
    missed_points = [
        {
            "date": "2025-09-18 18:15:00",
            "type": "A_turn",
            "reason": "9/20前的相對高點",
        },
        {
            "date": "2025-10-10 21:15:00",
            "type": "V_turn",
            "reason": "10月初的重要低點",
        },
    ]

    for point in missed_points:
        timestamp = pd.to_datetime(point["date"])
        point_idx = df[df["timestamp"] == timestamp].index

        if len(point_idx) == 0:
            print(f"❌ 找不到時間點: {point['date']}")
            continue

        idx = point_idx[0]
        current_data = df.iloc[idx]

        print(f"📍 {point['reason']}")
        print(f"   時間: {timestamp}")
        print(f"   價格: {current_data['close']:.2f}")
        print(f"   最高: {current_data['high']:.2f}")
        print(f"   最低: {current_data['low']:.2f}")

        # 檢查前後價格變化
        lookback = 5
        if idx >= lookback and idx < len(df) - lookback:
            before_prices = df["close"].iloc[idx - lookback : idx].values
            after_prices = (
                df["close"].iloc[idx + 1 : idx + 1 + lookback].values
            )

            if point["type"] == "A_turn":
                max_decline = (
                    min(after_prices) - current_data["close"]
                ) / current_data["close"]
                print(f"   後續最大跌幅: {max_decline*100:.2f}%")
            else:
                max_gain = (
                    max(after_prices) - current_data["close"]
                ) / current_data["close"]
                print(f"   後續最大漲幅: {max_gain*100:.2f}%")

        print()


def improved_detection_analysis():
    """分析如何改進檢測算法"""
    print("=== 改進建議 ===")
    print("1. 9/18高點問題:")
    print("   - 後續跌幅只有0.07%，不滿足1%門檻")
    print("   - 但這是重要的趨勢轉折點")
    print("   - 建議: 對於高位的A轉，可以降低跌幅要求到0.5%")
    print()

    print("2. 10/10低點問題:")
    print("   - 實際上已經被檢測到了 (V_turn @ 108432.09, +4.85%)")
    print("   - 但可能你看到的是更精確的低點102000")
    print("   - 建議: 使用更小的lookback窗口捕捉更精確的轉折點")
    print()

    print("3. 整體改進方向:")
    print("   - 對不同價格區間使用不同的變化幅度門檻")
    print("   - 高價位(>100k): 降低門檻到0.5%")
    print("   - 中價位(50k-100k): 保持1%門檻")
    print("   - 低價位(<50k): 提高門檻到1.5%")


if __name__ == "__main__":
    analyze_missed_points()
    print()
    improved_detection_analysis()
