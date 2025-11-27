#!/usr/bin/env python3
"""
平衡版A轉/V轉檢測 - 抓住重要轉折，避免過度標記
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def detect_reversal_patterns_balanced(df, lookback=4):
    """平衡版反轉檢測"""
    reversals = []

    for i in range(lookback, len(df) - lookback):
        current_price = df["close"].iloc[i]
        before_prices = df["close"].iloc[i - lookback : i].values
        after_prices = df["close"].iloc[i + 1 : i + 1 + lookback].values

        # 平衡門檻 - 比自適應版本更嚴格
        if current_price > 100000:
            min_change = 0.008  # 0.8%
        elif current_price > 50000:
            min_change = 0.012  # 1.2%
        else:
            min_change = 0.018  # 1.8%

        # 更嚴格的趨勢確認
        before_trend_strong = (
            before_prices[-1] - before_prices[0]
        ) / before_prices[0]
        after_trend_strong = (
            after_prices[-1] - after_prices[0]
        ) / after_prices[0]

        # A轉檢測 - 更嚴格條件
        if (
            current_price
            >= max(df["close"].iloc[i - lookback : i + 1 + lookback])
            * 0.998  # 更接近最高
            and before_trend_strong > 0.002  # 前面確實上漲
            and after_trend_strong < -0.002  # 後面確實下跌
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
                    "threshold": min_change,
                }
            )

        # V轉檢測 - 更嚴格條件
        elif (
            current_price
            <= min(df["close"].iloc[i - lookback : i + 1 + lookback])
            * 1.002  # 更接近最低
            and before_trend_strong < -0.002  # 前面確實下跌
            and after_trend_strong > 0.002  # 後面確實上漲
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
                    "threshold": min_change,
                }
            )

    return pd.DataFrame(reversals)


def plot_btc_with_balanced_reversals(df, reversals_df):
    """繪製平衡版反轉點圖表"""
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

    # 標記A轉點
    a_turns = reversals_df[reversals_df["type"] == "A_turn"]
    if len(a_turns) > 0:
        ax1.scatter(
            a_turns["timestamp"],
            a_turns["price"],
            color="red",
            marker="v",
            s=100,
            alpha=0.9,
            label=f"A轉 ({len(a_turns)}次)",
            zorder=5,
        )

    # 標記V轉點
    v_turns = reversals_df[reversals_df["type"] == "V_turn"]
    if len(v_turns) > 0:
        ax1.scatter(
            v_turns["timestamp"],
            v_turns["price"],
            color="lime",
            marker="^",
            s=100,
            alpha=0.9,
            label=f"V轉 ({len(v_turns)}次)",
            zorder=5,
        )

    ax1.set_title(
        "BTC價格走勢 - 平衡版A轉/V轉檢測", fontsize=16, color="white"
    )
    ax1.set_ylabel("價格 (USDT)", fontsize=12, color="white")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # x軸格式
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=21))

    # 副圖：成交量
    ax2.bar(df["timestamp"], df["volume"], color="gray", alpha=0.6, width=0.8)

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
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=21))

    plt.tight_layout()
    plt.xticks(rotation=45)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"btc_reversals_balanced_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="black")
    print(f"📊 平衡版圖表已保存: {filename}")

    return filename


def main():
    # 載入數據
    print("載入BTC一年數據...")
    df = pd.read_csv("training_data/btc_12months_15m.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 平衡版檢測
    print("執行平衡版A轉/V轉檢測...")
    reversals_df = detect_reversal_patterns_balanced(df)
    print(f"平衡版檢測到反轉點: {len(reversals_df)}個")

    # 統計
    a_count = len(reversals_df[reversals_df["type"] == "A_turn"])
    v_count = len(reversals_df[reversals_df["type"] == "V_turn"])
    print(f"A轉: {a_count}次, V轉: {v_count}次")

    # 檢查9/20-10/11期間
    start_date = pd.to_datetime("2025-09-15")
    end_date = pd.to_datetime("2025-10-15")
    period_reversals = reversals_df[
        (reversals_df["timestamp"] >= start_date)
        & (reversals_df["timestamp"] <= end_date)
    ]

    print(f"\n9/15-10/15期間的反轉點: {len(period_reversals)}個")
    for _, row in period_reversals.iterrows():
        print(
            f"  {row['timestamp']}: {row['type']} @ {row['price']:.2f}, 變化 {row['change']*100:.2f}%"
        )

    # 繪製圖表
    print("\n繪製平衡版圖表...")
    plot_btc_with_balanced_reversals(df, reversals_df)

    # 保存結果
    reversals_df.to_csv("btc_reversals_balanced.csv", index=False)
    print("📄 平衡版數據已保存: btc_reversals_balanced.csv")


if __name__ == "__main__":
    main()
