import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def identify_trend_by_slope(df, window=10):
    """
    基於移動平均斜率識別趨勢
    """
    # 計算短期移動平均
    df["ma_short"] = df["close"].rolling(window=window).mean()

    # 計算斜率（每5個點的變化率）
    df["slope"] = df["ma_short"].diff(5) / df["ma_short"].shift(5)

    # 定義趨勢
    trend_threshold = 0.001  # 0.1%的斜率閾值
    df["trend"] = 0  # 默認橫盤
    df.loc[df["slope"] > trend_threshold, "trend"] = 1  # 上漲
    df.loc[df["slope"] < -trend_threshold, "trend"] = -1  # 下跌

    return df


def plot_colored_trend_lines(df, reversals_df=None):
    """用顏色變化的線段繪製趨勢"""
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 12), height_ratios=[4, 1]
    )

    # 主圖：根據趨勢著色的價格線
    for i in range(1, len(df)):
        prev_time = df.iloc[i - 1]["timestamp"]
        curr_time = df.iloc[i]["timestamp"]
        prev_price = df.iloc[i - 1]["close"]
        curr_price = df.iloc[i]["close"]
        trend = df.iloc[i]["trend"]

        # 根據趨勢選擇顏色
        if trend == 1:  # 上漲
            color = "lime"
            alpha = 0.8
        elif trend == -1:  # 下跌
            color = "red"
            alpha = 0.8
        else:  # 橫盤
            color = "white"
            alpha = 0.6

        ax1.plot(
            [prev_time, curr_time],
            [prev_price, curr_price],
            color=color,
            linewidth=1.5,
            alpha=alpha,
        )

    # 添加移動平均線
    ax1.plot(
        df["timestamp"],
        df["ma_short"],
        color="orange",
        linewidth=1,
        alpha=0.7,
        label="MA10",
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
                color="orange",
                s=80,
                marker="v",
                label=f"A轉 ({len(a_turns)})",
                zorder=5,
                edgecolors="black",
                linewidth=1,
            )

        if len(v_turns) > 0:
            ax1.scatter(
                v_turns["timestamp"],
                v_turns["price"],
                color="cyan",
                s=80,
                marker="^",
                label=f"V轉 ({len(v_turns)})",
                zorder=5,
                edgecolors="black",
                linewidth=1,
            )

    # 添加圖例
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="lime", lw=3, label="上漲趨勢"),
        Line2D([0], [0], color="red", lw=3, label="下跌趨勢"),
        Line2D([0], [0], color="white", lw=3, label="橫盤整理"),
        Line2D([0], [0], color="orange", lw=2, label="MA10"),
    ]

    if reversals_df is not None:
        if len(a_turns) > 0:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="v",
                    color="w",
                    markerfacecolor="orange",
                    markersize=8,
                    label=f"A轉 ({len(a_turns)})",
                    linestyle="None",
                )
            )
        if len(v_turns) > 0:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor="cyan",
                    markersize=8,
                    label=f"V轉 ({len(v_turns)})",
                    linestyle="None",
                )
            )

    ax1.legend(handles=legend_elements, loc="upper left")
    ax1.set_title(
        "BTC 趨勢線段分析 (綠色=上漲, 紅色=下跌, 白色=橫盤)",
        fontsize=16,
        color="white",
    )
    ax1.set_ylabel("價格 (USD)", fontsize=12, color="white")
    ax1.grid(True, alpha=0.3)

    # 下圖：趨勢斜率
    colors = [
        "red" if s < -0.001 else "lime" if s > 0.001 else "gray"
        for s in df["slope"]
    ]
    ax2.scatter(df["timestamp"], df["slope"], c=colors, s=1, alpha=0.7)
    ax2.axhline(
        y=0.001,
        color="lime",
        linestyle="--",
        alpha=0.7,
        label="上漲閾值 (+0.1%)",
    )
    ax2.axhline(
        y=-0.001,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="下跌閾值 (-0.1%)",
    )
    ax2.axhline(y=0, color="white", linestyle="-", alpha=0.5)

    ax2.set_title("趨勢斜率指標", fontsize=12, color="white")
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

# 使用最近2個月數據
recent_data = df.tail(5760).copy()  # 2個月 * 30天 * 24小時 * 4 (15分鐘)
print(f"分析數據: {len(recent_data)} 行 (最近2個月)")

# 識別趨勢
print("🔍 識別趨勢...")
recent_data = identify_trend_by_slope(recent_data, window=10)

# 統計趨勢分布
uptrend_count = sum(recent_data["trend"] == 1)
downtrend_count = sum(recent_data["trend"] == -1)
sideways_count = sum(recent_data["trend"] == 0)
total_count = len(recent_data)

print(f"\n📊 趨勢分布統計:")
print(f"上漲時段: {uptrend_count} 個 ({uptrend_count/total_count*100:.1f}%)")
print(
    f"下跌時段: {downtrend_count} 個 ({downtrend_count/total_count*100:.1f}%)"
)
print(f"橫盤時段: {sideways_count} 個 ({sideways_count/total_count*100:.1f}%)")

# 分析趨勢轉換點
trend_changes = []
for i in range(1, len(recent_data)):
    if recent_data.iloc[i]["trend"] != recent_data.iloc[i - 1]["trend"]:
        trend_changes.append(
            {
                "timestamp": recent_data.iloc[i]["timestamp"],
                "price": recent_data.iloc[i]["close"],
                "from_trend": recent_data.iloc[i - 1]["trend"],
                "to_trend": recent_data.iloc[i]["trend"],
            }
        )

print(f"\n🔄 趨勢轉換點: {len(trend_changes)} 個")
print(f"最近5個趨勢轉換:")
for change in trend_changes[-5:]:
    trend_names = {1: "上漲", -1: "下跌", 0: "橫盤"}
    from_name = trend_names[change["from_trend"]]
    to_name = trend_names[change["to_trend"]]
    print(
        f"  {change['timestamp']}: {from_name} → {to_name} @ {change['price']:.0f}"
    )

# 繪製圖表
print(f"\n📊 繪製趨勢圖表...")
fig = plot_colored_trend_lines(recent_data, reversals_df)

# 保存圖表
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"btc_trend_colored_{timestamp}.png"
plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="black")
print(f"📊 趨勢圖表已保存: {filename}")

plt.show()
