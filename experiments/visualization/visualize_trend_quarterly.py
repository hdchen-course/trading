import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def identify_trend_by_slope(df, window=10):
    """基於移動平均斜率識別趨勢"""
    df = df.copy()
    df["ma_short"] = df["close"].rolling(window=window).mean()
    df["slope"] = df["ma_short"].diff(5) / df["ma_short"].shift(5)

    trend_threshold = 0.001
    df["trend"] = 0
    df.loc[df["slope"] > trend_threshold, "trend"] = 1
    df.loc[df["slope"] < -trend_threshold, "trend"] = -1

    return df


def plot_quarterly_trend(df, reversals_df, quarter_name, quarter_num):
    """繪製單季度趨勢圖"""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 10), height_ratios=[4, 1]
    )

    # 主圖：根據趨勢著色的價格線
    for i in range(1, len(df)):
        prev_time = df.iloc[i - 1]["timestamp"]
        curr_time = df.iloc[i]["timestamp"]
        prev_price = df.iloc[i - 1]["close"]
        curr_price = df.iloc[i]["close"]
        trend = df.iloc[i]["trend"]

        if trend == 1:
            color = "lime"
            alpha = 0.8
        elif trend == -1:
            color = "red"
            alpha = 0.8
        else:
            color = "white"
            alpha = 0.6

        ax1.plot(
            [prev_time, curr_time],
            [prev_price, curr_price],
            color=color,
            linewidth=1.5,
            alpha=alpha,
        )

    # 移動平均線
    ax1.plot(
        df["timestamp"],
        df["ma_short"],
        color="orange",
        linewidth=1,
        alpha=0.7,
        label="MA10",
    )

    # 標記轉折點
    if reversals_df is not None and len(reversals_df) > 0:
        period_reversals = reversals_df[
            (reversals_df["timestamp"] >= df["timestamp"].min())
            & (reversals_df["timestamp"] <= df["timestamp"].max())
        ]

        a_turns = period_reversals[period_reversals["type"] == "A_turn"]
        v_turns = period_reversals[period_reversals["type"] == "V_turn"]

        if len(a_turns) > 0:
            ax1.scatter(
                a_turns["timestamp"],
                a_turns["price"],
                color="orange",
                s=60,
                marker="v",
                label=f"A轉 ({len(a_turns)})",
                zorder=5,
                edgecolors="black",
                linewidth=0.5,
            )

        if len(v_turns) > 0:
            ax1.scatter(
                v_turns["timestamp"],
                v_turns["price"],
                color="cyan",
                s=60,
                marker="^",
                label=f"V轉 ({len(v_turns)})",
                zorder=5,
                edgecolors="black",
                linewidth=0.5,
            )

    # 統計趨勢分布
    uptrend_count = sum(df["trend"] == 1)
    downtrend_count = sum(df["trend"] == -1)
    sideways_count = sum(df["trend"] == 0)
    total_count = len(df)

    # 圖例和標題
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="lime",
            lw=3,
            label=f"上漲 ({uptrend_count/total_count*100:.1f}%)",
        ),
        Line2D(
            [0],
            [0],
            color="red",
            lw=3,
            label=f"下跌 ({downtrend_count/total_count*100:.1f}%)",
        ),
        Line2D(
            [0],
            [0],
            color="white",
            lw=3,
            label=f"橫盤 ({sideways_count/total_count*100:.1f}%)",
        ),
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
                    markersize=6,
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
                    markersize=6,
                    label=f"V轉 ({len(v_turns)})",
                    linestyle="None",
                )
            )

    ax1.legend(handles=legend_elements, loc="upper left")
    ax1.set_title(
        f"BTC {quarter_name} 趨勢分析 (綠=上漲, 紅=下跌, 白=橫盤)",
        fontsize=14,
        color="white",
    )
    ax1.set_ylabel("價格 (USD)", fontsize=10, color="white")
    ax1.grid(True, alpha=0.3)

    # 下圖：趨勢斜率
    colors = [
        "red" if s < -0.001 else "lime" if s > 0.001 else "gray"
        for s in df["slope"]
    ]
    ax2.scatter(df["timestamp"], df["slope"], c=colors, s=0.5, alpha=0.7)
    ax2.axhline(y=0.001, color="lime", linestyle="--", alpha=0.7, linewidth=1)
    ax2.axhline(y=-0.001, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax2.axhline(y=0, color="white", linestyle="-", alpha=0.5, linewidth=1)

    ax2.set_title("趨勢斜率", fontsize=10, color="white")
    ax2.set_xlabel("時間", fontsize=10, color="white")
    ax2.set_ylabel("斜率", fontsize=10, color="white")
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

# 計算每季度的數據點數 (3個月 = 90天 * 24小時 * 4)
quarter_size = 8640

print(f"總數據: {len(df)} 行")
print(f"每季度: {quarter_size} 行")

# 創建四個季度的圖表
quarters = [
    ("2024年11月-2025年2月", 1),
    ("2025年2月-2025年5月", 2),
    ("2025年5月-2025年8月", 3),
    ("2025年8月-2025年11月", 4),
]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for i, (quarter_name, quarter_num) in enumerate(quarters):
    start_idx = i * quarter_size
    end_idx = min((i + 1) * quarter_size, len(df))

    if start_idx >= len(df):
        print(f"⚠️ 數據不足，跳過 {quarter_name}")
        continue

    quarter_data = df.iloc[start_idx:end_idx].copy()

    if len(quarter_data) < 100:
        print(f"⚠️ {quarter_name} 數據太少，跳過")
        continue

    print(f"\n📊 處理 {quarter_name} ({len(quarter_data)} 行數據)")

    # 識別趨勢
    quarter_data = identify_trend_by_slope(quarter_data, window=10)

    # 繪製圖表
    fig = plot_quarterly_trend(
        quarter_data, reversals_df, quarter_name, quarter_num
    )

    # 保存圖表
    filename = f"btc_trend_Q{quarter_num}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="black")
    print(f"✅ {quarter_name} 圖表已保存: {filename}")

    plt.show()
    plt.close()

print(f"\n🎉 四個季度趨勢圖表生成完成！")
