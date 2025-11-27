#!/usr/bin/env python3
"""
分析MACD和其他指標在支撐壓力中的作用
"""

import pandas as pd
import numpy as np


def analyze_indicators():
    """分析各指標效果"""
    df = pd.read_csv("support_resistance_analysis.csv")

    print("=== 技術指標效果分析 ===\n")

    # MACD分析
    print("📊 MACD指標分析:")
    for touch_type in ["bb_upper_touch", "bb_lower_touch"]:
        type_data = df[df["type"] == touch_type].dropna(
            subset=["macd", "macd_signal", "macd_hist"]
        )
        if len(type_data) == 0:
            continue

        print(f"   {touch_type.upper()} ({len(type_data)}次):")
        print(f"     平均MACD: {type_data['macd'].mean():.6f}")
        print(f"     平均MACD Signal: {type_data['macd_signal'].mean():.6f}")
        print(f"     平均MACD Hist: {type_data['macd_hist'].mean():.6f}")

        # MACD柱狀圖正負分析
        positive_hist = type_data[type_data["macd_hist"] > 0]
        negative_hist = type_data[type_data["macd_hist"] < 0]

        if len(positive_hist) > 0:
            print(
                f"     MACD柱正值時 ({len(positive_hist)}次): 支撐{positive_hist['support_count'].mean():.2f}/5, 壓力{positive_hist['resistance_count'].mean():.2f}/5"
            )
            print(
                f"       5期漲幅: {positive_hist['max_return_5'].mean()*100:.2f}%, 跌幅: {positive_hist['min_return_5'].mean()*100:.2f}%"
            )

        if len(negative_hist) > 0:
            print(
                f"     MACD柱負值時 ({len(negative_hist)}次): 支撐{negative_hist['support_count'].mean():.2f}/5, 壓力{negative_hist['resistance_count'].mean():.2f}/5"
            )
            print(
                f"       5期漲幅: {negative_hist['max_return_5'].mean()*100:.2f}%, 跌幅: {negative_hist['min_return_5'].mean()*100:.2f}%"
            )
        print()

    # 布林位置分析
    print("📈 布林通道位置分析:")
    for touch_type in ["bb_upper_touch", "bb_lower_touch"]:
        type_data = df[df["type"] == touch_type].dropna(subset=["bb_position"])
        if len(type_data) == 0:
            continue

        print(f"   {touch_type.upper()}:")
        print(f"     平均布林位置: {type_data['bb_position'].mean():.2f}")

        # 按布林位置分組
        high_pos = type_data[type_data["bb_position"] > 0.8]
        mid_pos = type_data[
            (type_data["bb_position"] >= 0.2)
            & (type_data["bb_position"] <= 0.8)
        ]
        low_pos = type_data[type_data["bb_position"] < 0.2]

        for pos_name, pos_data in [
            ("高位>0.8", high_pos),
            ("中位0.2-0.8", mid_pos),
            ("低位<0.2", low_pos),
        ]:
            if len(pos_data) > 0:
                print(
                    f"     {pos_name} ({len(pos_data)}次): 支撐{pos_data['support_count'].mean():.2f}/5, 壓力{pos_data['resistance_count'].mean():.2f}/5"
                )
        print()

    # 多指標組合分析
    print("🎯 多指標組合效果:")

    # 布林上軌 + StochRSI超賣 + MACD負值
    combo1 = df[
        (df["type"] == "bb_upper_touch")
        & (df["stochrsi_k"] < 20)
        & (df["macd_hist"] < 0)
        & (df["volume_ratio"] > 1.5)
    ].dropna()

    if len(combo1) > 0:
        print(f"   上軌+超賣+MACD負+大量 ({len(combo1)}次):")
        print(f"     壓力效果: {combo1['resistance_count'].mean():.2f}/5")
        print(f"     5期跌幅: {combo1['min_return_5'].mean()*100:.2f}%")
        print(f"     勝率: {(combo1['min_return_5'] < -0.01).mean()*100:.1f}%")

    # 布林下軌 + StochRSI超買 + MACD正值
    combo2 = df[
        (df["type"] == "bb_lower_touch")
        & (df["stochrsi_k"] > 80)
        & (df["macd_hist"] > 0)
        & (df["volume_ratio"] > 1.5)
    ].dropna()

    if len(combo2) > 0:
        print(f"   下軌+超買+MACD正+大量 ({len(combo2)}次):")
        print(f"     支撐效果: {combo2['support_count'].mean():.2f}/5")
        print(f"     5期漲幅: {combo2['max_return_5'].mean()*100:.2f}%")
        print(f"     勝率: {(combo2['max_return_5'] > 0.01).mean()*100:.1f}%")


if __name__ == "__main__":
    analyze_indicators()
