#!/usr/bin/env python3
"""
增強版A轉/V轉檢測 - 降低門檻捕捉更多反轉點
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import talib


def detect_reversal_patterns_enhanced(df, lookback=3, min_change=0.01):
    """增強版反轉檢測 - 更寬鬆的條件"""
    reversals = []

    for i in range(lookback, len(df) - lookback):
        current_price = df["close"].iloc[i]
        before_prices = df["close"].iloc[i - lookback : i].values
        after_prices = df["close"].iloc[i + 1 : i + 1 + lookback].values

        # A轉檢測 (更寬鬆)
        if (
            current_price
            >= max(df["close"].iloc[i - lookback : i + 1 + lookback])
            * 0.999  # 接近最高
            and np.mean(before_prices[-2:]) < current_price  # 前面上漲趨勢
            and np.mean(after_prices[:2]) < current_price  # 後面下跌趨勢
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

        # V轉檢測 (更寬鬆)
        elif (
            current_price
            <= min(df["close"].iloc[i - lookback : i + 1 + lookback])
            * 1.001  # 接近最低
            and np.mean(before_prices[-2:]) > current_price  # 前面下跌趨勢
            and np.mean(after_prices[:2]) > current_price  # 後面上漲趨勢
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


def plot_btc_with_enhanced_reversals(df, reversals_df):
    """繪製增強版反轉點圖表"""
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
            s=80,
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
            s=80,
            alpha=0.8,
            label=f"V轉 ({len(v_turns)}次)",
            zorder=5,
        )

    ax1.set_title(
        "BTC價格走勢 - 增強版A轉/V轉檢測", fontsize=16, color="white"
    )
    ax1.set_ylabel("價格 (USDT)", fontsize=12, color="white")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 格式化x軸日期 - 每三週一個標籤
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(
        mdates.DayLocator(interval=21)
    )  # 每21天一個刻度

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
    ax2.xaxis.set_major_locator(
        mdates.DayLocator(interval=21)
    )  # 每21天一個刻度

    plt.tight_layout()
    plt.xticks(rotation=45)

    # 保存圖表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"btc_reversals_enhanced_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="black")
    print(f"📊 增強版圖表已保存: {filename}")

    return filename


def main():
    # 載入數據
    print("載入BTC一年數據...")
    df = pd.read_csv("training_data/btc_12months_15m.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 增強版檢測
    print("執行增強版A轉/V轉檢測...")
    reversals_df = detect_reversal_patterns_enhanced(
        df, lookback=3, min_change=0.01
    )  # 降低到1%
    print(f"增強版檢測到反轉點: {len(reversals_df)}個")

    if len(reversals_df) == 0:
        print("未發現符合條件的反轉點")
        return

    # 統計
    a_count = len(reversals_df[reversals_df["type"] == "A_turn"])
    v_count = len(reversals_df[reversals_df["type"] == "V_turn"])
    print(f"A轉: {a_count}次, V轉: {v_count}次")

    # 檢查6/24附近
    target_date = pd.to_datetime("2025-06-24")
    nearby = reversals_df[
        (reversals_df["timestamp"] >= target_date - pd.Timedelta(days=2))
        & (reversals_df["timestamp"] <= target_date + pd.Timedelta(days=2))
    ]

    if len(nearby) > 0:
        print(f"\n6/24附近的反轉點:")
        for _, row in nearby.iterrows():
            print(
                f"  {row['timestamp']}: {row['type']} @ {row['price']:.2f}, 變化 {row['change']*100:.2f}%"
            )

    # 繪製圖表
    print("繪製增強版圖表...")
    plot_btc_with_enhanced_reversals(df, reversals_df)

    # 保存結果
    reversals_df.to_csv("btc_reversals_enhanced.csv", index=False)
    print("📄 增強版數據已保存: btc_reversals_enhanced.csv")


if __name__ == "__main__":
    main()
