#!/usr/bin/env python3
"""
最終版A轉/V轉檢測 - 使用更大區間捕捉明顯轉折
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def detect_reversal_patterns_final(df, lookback=8):
    """最終版反轉檢測 - 更大區間窗口"""
    reversals = []

    for i in range(lookback, len(df) - lookback):
        current_price = df["close"].iloc[i]

        # 檢查是否為區間內的極值點
        # 檢查是否為真正的局部峰谷
        window_data = df.iloc[i - lookback : i + 1 + lookback]
        is_local_high = all(
            current_price >= df.iloc[j]["close"]
            for j in range(i - lookback, i + lookback + 1)
            if j != i
        )
        is_local_low = all(
            current_price <= df.iloc[j]["close"]
            for j in range(i - lookback, i + lookback + 1)
            if j != i
        )

        # 計算趨勢斜率
        before_prices = df["close"].iloc[i - 5 : i].values
        after_prices = df["close"].iloc[i + 1 : i + 6].values

        if len(before_prices) >= 3 and len(after_prices) >= 3:
            before_slope = (before_prices[-1] - before_prices[0]) / len(
                before_prices
            )
            after_slope = (after_prices[-1] - after_prices[0]) / len(
                after_prices
            )

            # 平衡的動態門檻
            base_change = 0.012 if current_price > 100000 else 0.015

            # 檢查是否為重要區域（多次測試的支撐/阻力）
            extended_window = df.iloc[max(0, i - 80) : min(len(df), i + 80)]
            price_range = current_price * 0.02  # 2%範圍
            similar_prices = extended_window[
                (extended_window["low"] >= current_price - price_range)
                & (extended_window["low"] <= current_price + price_range)
            ]

            # 只對非常重要的區域適度降低門檻
            if len(similar_prices) >= 15:  # 非常重要的區域
                min_change = base_change * 0.8  # 只降低20%
            else:
                min_change = base_change

            # A轉：局部高點 + 上升轉下降
            if is_local_high and before_slope > 0 and after_slope < 0:
                max_decline = (
                    min(after_prices) - current_price
                ) / current_price
                if abs(max_decline) >= min_change:
                    # 檢查是否與最近的轉折點太接近
                    too_close = False
                    for prev_rev in reversals[-3:]:  # 檢查最近3個
                        time_diff = abs(
                            (
                                df["timestamp"].iloc[i] - prev_rev["timestamp"]
                            ).total_seconds()
                        )
                        if time_diff < 7200:  # 2小時內
                            too_close = True
                            break

                    if not too_close:
                        reversals.append(
                            {
                                "timestamp": df["timestamp"].iloc[i],
                                "type": "A_turn",
                                "price": current_price,
                                "index": i,
                                "change": max_decline,
                                "slopes": f"{before_slope:.1f}→{after_slope:.1f}",
                            }
                        )

            # V轉：局部低點 + 下降轉上升
            elif is_local_low and before_slope < 0 and after_slope > 0:
                max_gain = (max(after_prices) - current_price) / current_price
                if max_gain >= min_change:
                    # 檢查是否與最近的轉折點太接近
                    too_close = False
                    for prev_rev in reversals[-3:]:  # 檢查最近3個
                        time_diff = abs(
                            (
                                df["timestamp"].iloc[i] - prev_rev["timestamp"]
                            ).total_seconds()
                        )
                        if time_diff < 7200:  # 2小時內
                            too_close = True
                            break

                    if not too_close:
                        reversals.append(
                            {
                                "timestamp": df["timestamp"].iloc[i],
                                "type": "V_turn",
                                "price": current_price,
                                "index": i,
                                "change": max_gain,
                                "slopes": f"{before_slope:.1f}→{after_slope:.1f}",
                            }
                        )

    return pd.DataFrame(reversals)


def plot_btc_with_final_reversals(df, reversals_df):
    """繪製最終版反轉點圖表"""
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
            s=120,
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
            s=120,
            alpha=0.9,
            label=f"V轉 ({len(v_turns)}次)",
            zorder=5,
        )

    ax1.set_title(
        "BTC價格走勢 - 最終版A轉/V轉檢測 (20K線區間)",
        fontsize=16,
        color="white",
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
    filename = f"btc_reversals_final_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="black")
    print(f"📊 最終版圖表已保存: {filename}")

    return filename


def main():
    # 載入數據
    print("載入BTC一年數據...")
    df = pd.read_csv("training_data/btc_12months_15m.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 最終版檢測
    print("執行最終版A轉/V轉檢測 (20K線區間)...")
    reversals_df = detect_reversal_patterns_final(
        df, lookback=10
    )  # 10*2+1 = 21K線區間
    print(f"最終版檢測到反轉點: {len(reversals_df)}個")

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
    print("\n繪製最終版圖表...")
    plot_btc_with_final_reversals(df, reversals_df)

    # 保存結果
    reversals_df.to_csv("btc_reversals_final.csv", index=False)
    print("📄 最終版數據已保存: btc_reversals_final.csv")


if __name__ == "__main__":
    main()
