import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def identify_trend_segments(df, window=20):
    """
    識別趨勢線段，基於價格變化方向
    """
    segments = []

    # 計算每個窗口的趨勢
    for i in range(window, len(df) - window):
        current_price = df.iloc[i]["close"]
        past_price = df.iloc[i - window]["close"]
        future_price = df.iloc[i + window]["close"]

        # 計算趨勢方向
        past_trend = (current_price - past_price) / past_price
        future_trend = (future_price - current_price) / current_price

        # 綜合趨勢判斷
        if past_trend > 0.01 and future_trend > 0.01:  # 持續上漲
            trend = 1
        elif past_trend < -0.01 and future_trend < -0.01:  # 持續下跌
            trend = -1
        else:  # 橫盤或轉折
            trend = 0

        segments.append(
            {
                "index": i,
                "timestamp": df.iloc[i]["timestamp"],
                "price": current_price,
                "trend": trend,
            }
        )

    return segments


def plot_trend_lines(df, segments, reversals_df=None):
    """用不同顏色的線段繪製趨勢"""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 10))

    # 將segments轉換為DataFrame方便處理
    seg_df = pd.DataFrame(segments)

    # 繪製不同趨勢的線段
    for i in range(len(seg_df) - 1):
        current_idx = seg_df.iloc[i]["index"]
        next_idx = seg_df.iloc[i + 1]["index"]
        trend = seg_df.iloc[i]["trend"]

        # 獲取線段數據
        segment_data = df.iloc[current_idx : next_idx + 1]

        # 根據趨勢選擇顏色
        if trend == 1:  # 上漲
            color = "lime"
            alpha = 0.8
        elif trend == -1:  # 下跌
            color = "red"
            alpha = 0.8
        else:  # 橫盤
            color = "gray"
            alpha = 0.6

        ax.plot(
            segment_data["timestamp"],
            segment_data["close"],
            color=color,
            linewidth=2,
            alpha=alpha,
        )

    # 標記A轉V轉點
    if reversals_df is not None and len(reversals_df) > 0:
        recent_reversals = reversals_df[
            reversals_df["timestamp"] >= df["timestamp"].min()
        ]

        a_turns = recent_reversals[recent_reversals["type"] == "A_turn"]
        v_turns = recent_reversals[recent_reversals["type"] == "V_turn"]

        if len(a_turns) > 0:
            ax.scatter(
                a_turns["timestamp"],
                a_turns["price"],
                color="orange",
                s=100,
                marker="v",
                label=f"A轉 ({len(a_turns)})",
                zorder=5,
                edgecolors="white",
                linewidth=1,
            )

        if len(v_turns) > 0:
            ax.scatter(
                v_turns["timestamp"],
                v_turns["price"],
                color="cyan",
                s=100,
                marker="^",
                label=f"V轉 ({len(v_turns)})",
                zorder=5,
                edgecolors="white",
                linewidth=1,
            )

    # 添加圖例
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="lime", lw=3, label="上漲趨勢"),
        Line2D([0], [0], color="red", lw=3, label="下跌趨勢"),
        Line2D([0], [0], color="gray", lw=3, label="橫盤整理"),
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
                    markersize=10,
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
                    markersize=10,
                    label=f"V轉 ({len(v_turns)})",
                    linestyle="None",
                )
            )

    ax.legend(handles=legend_elements, loc="upper left")

    ax.set_title(
        "BTC 趨勢線段分析 (綠色=上漲, 紅色=下跌, 灰色=橫盤)",
        fontsize=16,
        color="white",
    )
    ax.set_xlabel("時間", fontsize=12, color="white")
    ax.set_ylabel("價格 (USD)", fontsize=12, color="white")
    ax.grid(True, alpha=0.3)

    # 設置x軸間隔
    ax.tick_params(axis="x", rotation=45)

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

# 使用最近3個月數據
recent_data = df.tail(8640)  # 3個月 * 30天 * 24小時 * 4 (15分鐘)
print(f"分析數據: {len(recent_data)} 行 (最近3個月)")

# 識別趨勢線段
print("🔍 識別趨勢線段...")
segments = identify_trend_segments(recent_data, window=20)

# 統計趨勢分布
uptrend_count = sum(1 for s in segments if s["trend"] == 1)
downtrend_count = sum(1 for s in segments if s["trend"] == -1)
sideways_count = sum(1 for s in segments if s["trend"] == 0)

print(f"\n📊 趨勢線段統計:")
print(f"上漲線段: {uptrend_count} 個 ({uptrend_count/len(segments)*100:.1f}%)")
print(
    f"下跌線段: {downtrend_count} 個 ({downtrend_count/len(segments)*100:.1f}%)"
)
print(
    f"橫盤線段: {sideways_count} 個 ({sideways_count/len(segments)*100:.1f}%)"
)

# 分析最近趨勢
print(f"\n🔍 最近10個趨勢線段:")
for segment in segments[-10:]:
    trend_name = {1: "上漲", -1: "下跌", 0: "橫盤"}[segment["trend"]]
    print(f"  {segment['timestamp']}: {trend_name} @ {segment['price']:.0f}")

# 繪製圖表
print(f"\n📊 繪製趨勢線段圖表...")
fig = plot_trend_lines(recent_data, segments, reversals_df)

# 保存圖表
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"btc_trend_lines_{timestamp}.png"
plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="black")
print(f"📊 趨勢線段圖表已保存: {filename}")

# 保存線段數據
segments_df = pd.DataFrame(segments)
segments_df.to_csv("btc_trend_segments.csv", index=False)
print(f"📄 趨勢線段數據已保存: btc_trend_segments.csv")

plt.show()
