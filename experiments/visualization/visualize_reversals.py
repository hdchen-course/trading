#!/usr/bin/env python3
"""
在BTC價格圖上可視化A轉/V轉反轉點
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import talib


def detect_reversal_patterns(df, lookback=5, min_change=0.015):
    """檢測A轉和V轉模式"""
    reversals = []

    for i in range(lookback, len(df) - lookback):
        current_price = df["close"].iloc[i]
        before_prices = df["close"].iloc[i - lookback : i].values
        after_prices = df["close"].iloc[i + 1 : i + 1 + lookback].values

        # A轉檢測 (尖頂反轉)
        if (
            current_price
            == max(df["close"].iloc[i - lookback : i + 1 + lookback])
            and all(before_prices[-3:] < current_price)
            and all(after_prices[:3] < current_price)
            and (current_price - min(after_prices)) / current_price
            >= min_change
        ):

            reversals.append(
                {
                    "timestamp": df["timestamp"].iloc[i],
                    "type": "A_turn",
                    "price": current_price,
                    "index": i,
                    "change": (min(after_prices) - current_price)
                    / current_price,
                }
            )

        # V轉檢測 (尖底反轉)
        elif (
            current_price
            == min(df["close"].iloc[i - lookback : i + 1 + lookback])
            and all(before_prices[-3:] > current_price)
            and all(after_prices[:3] > current_price)
            and (max(after_prices) - current_price) / current_price
            >= min_change
        ):

            reversals.append(
                {
                    "timestamp": df["timestamp"].iloc[i],
                    "type": "V_turn",
                    "price": current_price,
                    "index": i,
                    "change": (max(after_prices) - current_price)
                    / current_price,
                }
            )

    return pd.DataFrame(reversals)


def calculate_reversal_features(df, reversals_df):
    """計算反轉點的技術指標特徵"""
    # 計算技術指標
    df["stochrsi_k"], df["stochrsi_d"] = talib.STOCHRSI(
        df["close"], timeperiod=14, fastk_period=3, fastd_period=3
    )
    df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(df["close"])
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    features = []
    for _, reversal in reversals_df.iterrows():
        idx = reversal["index"]
        current_data = df.iloc[idx]

        features.append(
            {
                "timestamp": reversal["timestamp"],
                "type": reversal["type"],
                "price": reversal["price"],
                "change": reversal["change"],
                "stochrsi_k": current_data["stochrsi_k"],
                "volume_ratio": current_data["volume_ratio"],
                "macd_hist": current_data["macd_hist"],
            }
        )

    return pd.DataFrame(features)


def plot_btc_with_reversals(df, reversals_df):
    """繪製BTC價格走勢並標記反轉點"""
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(20, 12), height_ratios=[3, 1]
    )

    # 主圖：價格走勢
    ax1.plot(
        df["timestamp"],
        df["close"],
        color="white",
        linewidth=0.8,
        alpha=0.7,
        label="BTC Price",
    )

    # 標記A轉點 (紅色向下箭頭)
    a_turns = reversals_df[reversals_df["type"] == "A_turn"]
    if len(a_turns) > 0:
        ax1.scatter(
            a_turns["timestamp"],
            a_turns["price"],
            color="red",
            marker="v",
            s=100,
            alpha=0.8,
            label=f"A轉 ({len(a_turns)}次)",
            zorder=5,
        )

    # 標記V轉點 (綠色向上箭頭)
    v_turns = reversals_df[reversals_df["type"] == "V_turn"]
    if len(v_turns) > 0:
        ax1.scatter(
            v_turns["timestamp"],
            v_turns["price"],
            color="lime",
            marker="^",
            s=100,
            alpha=0.8,
            label=f"V轉 ({len(v_turns)}次)",
            zorder=5,
        )

    ax1.set_title(
        "BTC價格走勢 - A轉/V轉反轉點標記", fontsize=16, color="white"
    )
    ax1.set_ylabel("價格 (USDT)", fontsize=12, color="white")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 格式化x軸日期 - 清楚的時間標籤
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(
        mdates.DayLocator(interval=15)
    )  # 每15天一個刻度

    # 副圖：成交量
    ax2.bar(df["timestamp"], df["volume"], color="gray", alpha=0.6, width=0.8)

    # 標記反轉點的成交量
    if len(a_turns) > 0:
        a_volumes = df[df["timestamp"].isin(a_turns["timestamp"])]["volume"]
        ax2.bar(
            a_turns["timestamp"], a_volumes, color="red", alpha=0.8, width=0.8
        )

    if len(v_turns) > 0:
        v_volumes = df[df["timestamp"].isin(v_turns["timestamp"])]["volume"]
        ax2.bar(
            v_turns["timestamp"], v_volumes, color="lime", alpha=0.8, width=0.8
        )

    ax2.set_ylabel("成交量", fontsize=12, color="white")
    ax2.set_xlabel("時間", fontsize=12, color="white")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=15))

    plt.tight_layout()
    plt.xticks(rotation=45)

    # 保存圖表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"btc_reversals_visualization_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="black")
    print(f"📊 圖表已保存: {filename}")

    return filename


def analyze_reversal_characteristics(features_df):
    """分析反轉特徵"""
    print("=== A轉/V轉特徵分析 ===\n")

    for pattern_type in ["A_turn", "V_turn"]:
        pattern_data = features_df[
            features_df["type"] == pattern_type
        ].dropna()
        if len(pattern_data) == 0:
            continue

        print(f"📊 {pattern_type.upper()} 特徵 (共{len(pattern_data)}次):")
        print(f"   平均變化幅度: {pattern_data['change'].mean()*100:.2f}%")
        print(f"   最大變化幅度: {pattern_data['change'].max()*100:.2f}%")
        print(f"   平均StochRSI K: {pattern_data['stochrsi_k'].mean():.1f}")
        print(f"   平均成交量比率: {pattern_data['volume_ratio'].mean():.2f}")

        # 極端值統計
        if pattern_type == "A_turn":
            extreme = pattern_data[pattern_data["stochrsi_k"] > 80]
            print(
                f"   StochRSI>80時: {len(extreme)}次 ({len(extreme)/len(pattern_data)*100:.1f}%)"
            )
        else:
            extreme = pattern_data[pattern_data["stochrsi_k"] < 20]
            print(
                f"   StochRSI<20時: {len(extreme)}次 ({len(extreme)/len(pattern_data)*100:.1f}%)"
            )

        print()


def main():
    # 載入數據
    print("載入BTC一年數據...")
    df = pd.read_csv("training_data/btc_12months_15m.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"數據範圍: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"總數據點: {len(df)}個")

    # 檢測反轉模式
    print("\n檢測A轉/V轉反轉模式...")
    reversals_df = detect_reversal_patterns(df, lookback=5, min_change=0.015)
    print(f"發現反轉點: {len(reversals_df)}個")

    if len(reversals_df) == 0:
        print("未發現符合條件的反轉點")
        return

    # 計算特徵
    print("計算反轉點特徵...")
    features_df = calculate_reversal_features(df, reversals_df)

    # 分析特徵
    analyze_reversal_characteristics(features_df)

    # 繪製圖表
    print("繪製價格走勢圖...")
    plot_btc_with_reversals(df, reversals_df)

    # 保存結果
    reversals_df.to_csv("btc_reversals_detected.csv", index=False)
    features_df.to_csv("btc_reversals_features.csv", index=False)
    print(
        "📄 反轉點數據已保存: btc_reversals_detected.csv, btc_reversals_features.csv"
    )


if __name__ == "__main__":
    main()
