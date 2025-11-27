#!/usr/bin/env python3
"""
深度分析StochRSI在支撐壓力中的作用
"""

import pandas as pd
import numpy as np


def analyze_stochrsi_patterns():
    """分析StochRSI模式"""
    df = pd.read_csv("support_resistance_analysis.csv")

    print("=== StochRSI 深度分析 ===\n")

    # 定義StochRSI區間
    def get_stochrsi_zone(k_val):
        if pd.isna(k_val):
            return "Unknown"
        elif k_val <= 20:
            return "Oversold"
        elif k_val >= 80:
            return "Overbought"
        else:
            return "Neutral"

    df["stochrsi_zone"] = df["stochrsi_k"].apply(get_stochrsi_zone)

    # 按觸碰類型和StochRSI區間分析
    for touch_type in ["bb_upper_touch", "bb_lower_touch"]:
        print(f"📈 {touch_type.upper()} + StochRSI 組合分析:")
        type_data = df[df["type"] == touch_type]

        for zone in ["Oversold", "Neutral", "Overbought"]:
            zone_data = type_data[type_data["stochrsi_zone"] == zone]
            if len(zone_data) == 0:
                continue

            print(f"   {zone} 區間 ({len(zone_data)}次):")
            print(f"     支撐效果: {zone_data['support_count'].mean():.2f}/5")
            print(
                f"     壓力效果: {zone_data['resistance_count'].mean():.2f}/5"
            )
            print(
                f"     5期最大漲幅: {zone_data['max_return_5'].mean()*100:.2f}%"
            )
            print(
                f"     5期最大跌幅: {zone_data['min_return_5'].mean()*100:.2f}%"
            )
            print(f"     平均StochRSI K: {zone_data['stochrsi_k'].mean():.1f}")
            print(
                f"     平均成交量比率: {zone_data['volume_ratio'].mean():.2f}"
            )
        print()

    # 尋找最佳交易信號組合
    print("🎯 最佳交易信號組合:")

    # 布林上軌 + 超買
    bb_upper_overbought = df[
        (df["type"] == "bb_upper_touch")
        & (df["stochrsi_zone"] == "Overbought")
    ]
    if len(bb_upper_overbought) > 0:
        print(f"   布林上軌 + StochRSI超買 ({len(bb_upper_overbought)}次):")
        print(
            f"     壓力效果: {bb_upper_overbought['resistance_count'].mean():.2f}/5"
        )
        print(
            f"     5期後跌幅: {bb_upper_overbought['min_return_5'].mean()*100:.2f}%"
        )
        print(
            f"     勝率指標: {(bb_upper_overbought['min_return_5'] < -0.005).mean()*100:.1f}%"
        )

    # 布林下軌 + 超賣
    bb_lower_oversold = df[
        (df["type"] == "bb_lower_touch") & (df["stochrsi_zone"] == "Oversold")
    ]
    if len(bb_lower_oversold) > 0:
        print(f"   布林下軌 + StochRSI超賣 ({len(bb_lower_oversold)}次):")
        print(
            f"     支撐效果: {bb_lower_oversold['support_count'].mean():.2f}/5"
        )
        print(
            f"     5期後漲幅: {bb_lower_oversold['max_return_5'].mean()*100:.2f}%"
        )
        print(
            f"     勝率指標: {(bb_lower_oversold['max_return_5'] > 0.005).mean()*100:.1f}%"
        )

    print()

    # K線與D線交叉分析
    print("📊 StochRSI K/D線交叉分析:")
    df["kd_diff"] = df["stochrsi_k"] - df["stochrsi_d"]
    df["kd_signal"] = np.where(df["kd_diff"] > 0, "K>D", "K<D")

    for touch_type in ["bb_upper_touch", "bb_lower_touch"]:
        type_data = df[df["type"] == touch_type]
        print(f"   {touch_type.upper()}:")

        for signal in ["K>D", "K<D"]:
            signal_data = type_data[type_data["kd_signal"] == signal]
            if len(signal_data) > 0:
                print(
                    f"     {signal} ({len(signal_data)}次): 5期漲幅 {signal_data['max_return_5'].mean()*100:.2f}%, 跌幅 {signal_data['min_return_5'].mean()*100:.2f}%"
                )

    return df


if __name__ == "__main__":
    analyze_stochrsi_patterns()
