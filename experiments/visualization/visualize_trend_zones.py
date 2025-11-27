import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def identify_trend_zones(df, window=50, min_duration=20):
    """
    識別趨勢上漲和下跌區間
    window: 計算趨勢的窗口大小
    min_duration: 最小趨勢持續時間（K線數）
    """
    # 計算移動平均線斜率
    df["ma"] = df["close"].rolling(window=window).mean()
    df["ma_slope"] = df["ma"].diff(5) / df["ma"].shift(5)  # 5期斜率

    # 定義趨勢
    trend_threshold = 0.002  # 0.2%的斜率閾值
    df["trend"] = 0  # 0=橫盤, 1=上漲, -1=下跌
    df.loc[df["ma_slope"] > trend_threshold, "trend"] = 1
    df.loc[df["ma_slope"] < -trend_threshold, "trend"] = -1

    # 識別趨勢區間
    zones = []
    current_trend = None
    start_idx = None

    for i in range(len(df)):
        trend = df.iloc[i]["trend"]

        if current_trend != trend:
            # 趨勢改變
            if start_idx is not None and i - start_idx >= min_duration:
                # 保存前一個趨勢區間
                zones.append(
                    {
                        "start_idx": start_idx,
                        "end_idx": i - 1,
                        "start_time": df.iloc[start_idx]["timestamp"],
                        "end_time": df.iloc[i - 1]["timestamp"],
                        "trend": current_trend,
                        "start_price": df.iloc[start_idx]["close"],
                        "end_price": df.iloc[i - 1]["close"],
                        "duration": i - start_idx,
                    }
                )

            current_trend = trend
            start_idx = i

    # 處理最後一個區間
    if start_idx is not None and len(df) - start_idx >= min_duration:
        zones.append(
            {
                "start_idx": start_idx,
                "end_idx": len(df) - 1,
                "start_time": df.iloc[start_idx]["timestamp"],
                "end_time": df.iloc[-1]["timestamp"],
                "trend": current_trend,
                "start_price": df.iloc[start_idx]["close"],
                "end_price": df.iloc[-1]["close"],
                "duration": len(df) - start_idx,
            }
        )

    return zones


def plot_trend_zones(df, zones, reversals_df=None):
    """繪製趨勢區間圖表"""
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 12), height_ratios=[3, 1]
    )

    # 主圖：價格和趨勢區間
    ax1.plot(
        df["timestamp"],
        df["close"],
        color="white",
        linewidth=1,
        alpha=0.8,
        label="BTC價格",
    )
    ax1.plot(
        df["timestamp"],
        df["ma"],
        color="orange",
        linewidth=1.5,
        alpha=0.7,
        label=f"MA{50}",
    )

    # 繪製趨勢區間
    for zone in zones:
        start_time = zone["start_time"]
        end_time = zone["end_time"]

        if zone["trend"] == 1:  # 上漲趨勢
            ax1.axvspan(
                start_time,
                end_time,
                alpha=0.2,
                color="green",
                label="上漲趨勢" if zone == zones[0] else "",
            )
        elif zone["trend"] == -1:  # 下跌趨勢
            ax1.axvspan(
                start_time,
                end_time,
                alpha=0.2,
                color="red",
                label="下跌趨勢" if zone == zones[0] else "",
            )

    # 標記A轉V轉點
    if reversals_df is not None and len(reversals_df) > 0:
        recent_reversals = reversals_df[
            reversals_df["timestamp"] >= df["timestamp"].min()
        ]

        a_turns = recent_reversals[recent_reversals["type"] == "A_turn"]
        v_turns = recent_reversals[recent_reversals["type"] == "V_turn"]

        if len(a_turns) > 0:
            ax1.scatter(
                a_turns["timestamp"],
                a_turns["price"],
                color="red",
                s=80,
                marker="v",
                label=f"A轉 ({len(a_turns)})",
                zorder=5,
            )

        if len(v_turns) > 0:
            ax1.scatter(
                v_turns["timestamp"],
                v_turns["price"],
                color="lime",
                s=80,
                marker="^",
                label=f"V轉 ({len(v_turns)})",
                zorder=5,
            )

    ax1.set_title("BTC 趨勢區間分析", fontsize=16, color="white")
    ax1.set_ylabel("價格 (USD)", fontsize=12, color="white")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 下圖：趨勢斜率
    ax2.plot(df["timestamp"], df["ma_slope"], color="cyan", linewidth=1)
    ax2.axhline(
        y=0.002, color="green", linestyle="--", alpha=0.7, label="上漲閾值"
    )
    ax2.axhline(
        y=-0.002, color="red", linestyle="--", alpha=0.7, label="下跌閾值"
    )
    ax2.axhline(y=0, color="white", linestyle="-", alpha=0.5)
    ax2.fill_between(
        df["timestamp"],
        0,
        df["ma_slope"],
        where=(df["ma_slope"] > 0),
        color="green",
        alpha=0.3,
    )
    ax2.fill_between(
        df["timestamp"],
        0,
        df["ma_slope"],
        where=(df["ma_slope"] < 0),
        color="red",
        alpha=0.3,
    )

    ax2.set_title("趨勢斜率", fontsize=12, color="white")
    ax2.set_xlabel("時間", fontsize=12, color="white")
    ax2.set_ylabel("斜率", fontsize=12, color="white")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# 讀取數據
print("📊 讀取BTC數據...")
df = pd.read_csv("training_data/btc_12months_15m.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 讀取轉折點數據
try:
    reversals_df = pd.read_csv("btc_reversals_final.csv")
    reversals_df["timestamp"] = pd.to_datetime(reversals_df["timestamp"])
    print(f"📈 載入轉折點: {len(reversals_df)} 個")
except:
    reversals_df = None
    print("⚠️ 未找到轉折點數據")

# 使用最近6個月數據
recent_data = df.tail(17280)  # 6個月 * 30天 * 24小時 * 4 (15分鐘)
print(f"分析數據: {len(recent_data)} 行 (最近6個月)")

# 識別趨勢區間
print("🔍 識別趨勢區間...")
zones = identify_trend_zones(recent_data, window=50, min_duration=20)

print(f"\n📊 趨勢區間統計:")
uptrend_zones = [z for z in zones if z["trend"] == 1]
downtrend_zones = [z for z in zones if z["trend"] == -1]
sideways_zones = [z for z in zones if z["trend"] == 0]

print(f"上漲趨勢: {len(uptrend_zones)} 個區間")
print(f"下跌趨勢: {len(downtrend_zones)} 個區間")
print(f"橫盤整理: {len(sideways_zones)} 個區間")

# 顯示最近的趨勢區間
print(f"\n🔍 最近10個趨勢區間:")
for i, zone in enumerate(zones[-10:]):
    trend_name = {1: "上漲", -1: "下跌", 0: "橫盤"}[zone["trend"]]
    change = (zone["end_price"] - zone["start_price"]) / zone["start_price"]
    duration_hours = zone["duration"] * 0.25  # 15分鐘 = 0.25小時

    print(f"  {zone['start_time']} - {zone['end_time']}")
    print(
        f"    {trend_name}趨勢: {zone['start_price']:.0f} → {zone['end_price']:.0f} ({change:+.1%}), 持續 {duration_hours:.1f}h"
    )

# 繪製圖表
print(f"\n📊 繪製趨勢區間圖表...")
fig = plot_trend_zones(recent_data, zones, reversals_df)

# 保存圖表
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"btc_trend_zones_{timestamp}.png"
plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="black")
print(f"📊 趨勢區間圖表已保存: {filename}")

# 保存趨勢區間數據
zones_df = pd.DataFrame(zones)
if len(zones_df) > 0:
    zones_df.to_csv("btc_trend_zones.csv", index=False)
    print(f"📄 趨勢區間數據已保存: btc_trend_zones.csv")

plt.show()
