import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def calculate_slope(prices, window=5):
    """計算價格斜率"""
    if len(prices) < window:
        return 0
    x = np.arange(window)
    y = prices[-window:]
    return np.polyfit(x, y, 1)[0]


def detect_improved_reversals(df, lookback=10, min_change=0.008):
    """
    改進的A轉V轉檢測
    - A轉：必須是局部最高點 + 前後趨勢斜率相反
    - V轉：必須是局部最低點 + 前後趨勢斜率相反
    """
    reversals = []

    for i in range(lookback, len(df) - lookback):
        current_price = df.iloc[i]["close"]
        current_time = df.iloc[i]["timestamp"]

        # 前後價格窗口
        before_prices = df.iloc[i - lookback : i]["close"].values
        after_prices = df.iloc[i + 1 : i + lookback + 1]["close"].values

        # 計算前後斜率
        before_slope = calculate_slope(before_prices)
        after_slope = calculate_slope(after_prices)

        # 檢查是否為局部最高點 (A轉候選)
        is_local_high = all(
            current_price >= df.iloc[j]["close"]
            for j in range(i - lookback, i + lookback + 1)
            if j != i
        )

        # 檢查是否為局部最低點 (V轉候選)
        is_local_low = all(
            current_price <= df.iloc[j]["close"]
            for j in range(i - lookback, i + lookback + 1)
            if j != i
        )

        if is_local_high and before_slope > 0 and after_slope < 0:
            # A轉：上升趨勢轉下降趨勢
            change = (after_prices[-1] - current_price) / current_price
            if abs(change) >= min_change:
                reversals.append(
                    {
                        "timestamp": current_time,
                        "price": current_price,
                        "type": "A_turn",
                        "change": change,
                        "before_slope": before_slope,
                        "after_slope": after_slope,
                    }
                )

        elif is_local_low and before_slope < 0 and after_slope > 0:
            # V轉：下降趨勢轉上升趨勢
            change = (after_prices[-1] - current_price) / current_price
            if abs(change) >= min_change:
                reversals.append(
                    {
                        "timestamp": current_time,
                        "price": current_price,
                        "type": "V_turn",
                        "change": change,
                        "before_slope": before_slope,
                        "after_slope": after_slope,
                    }
                )

    return pd.DataFrame(reversals)


# 讀取數據
print("📊 讀取BTC數據...")
df = pd.read_csv("btc_15min_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 改進檢測
print("🔍 執行改進的A轉V轉檢測...")
reversals_df = detect_improved_reversals(df, lookback=20, min_change=0.008)

print(f"\n📈 檢測結果:")
print(f"總共檢測到 {len(reversals_df)} 個轉折點")

if len(reversals_df) > 0:
    a_turns = reversals_df[reversals_df["type"] == "A_turn"]
    v_turns = reversals_df[reversals_df["type"] == "V_turn"]

    print(
        f"A轉 (峰值): {len(a_turns)} 個，平均變化: {a_turns['change'].mean():.2%}"
    )
    print(
        f"V轉 (谷值): {len(v_turns)} 個，平均變化: {v_turns['change'].mean():.2%}"
    )

    print(f"\n最近的轉折點:")
    for _, row in reversals_df.tail(10).iterrows():
        print(
            f"  {row['timestamp']}: {row['type']} @ {row['price']:.2f}, 變化 {row['change']:.2%}, 斜率 {row['before_slope']:.1f}→{row['after_slope']:.1f}"
        )

# 繪圖
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(16, 10))

# 繪製價格線
recent_data = df.tail(2000)
ax.plot(
    recent_data["timestamp"],
    recent_data["close"],
    color="white",
    linewidth=1,
    alpha=0.8,
    label="BTC價格",
)

# 標記轉折點
if len(reversals_df) > 0:
    recent_reversals = reversals_df[
        reversals_df["timestamp"] >= recent_data["timestamp"].min()
    ]

    a_turns = recent_reversals[recent_reversals["type"] == "A_turn"]
    v_turns = recent_reversals[recent_reversals["type"] == "V_turn"]

    if len(a_turns) > 0:
        ax.scatter(
            a_turns["timestamp"],
            a_turns["price"],
            color="red",
            s=100,
            marker="v",
            label=f"A轉 ({len(a_turns)})",
            zorder=5,
        )

    if len(v_turns) > 0:
        ax.scatter(
            v_turns["timestamp"],
            v_turns["price"],
            color="lime",
            s=100,
            marker="^",
            label=f"V轉 ({len(v_turns)})",
            zorder=5,
        )

ax.set_title(
    "BTC 改進版A轉V轉檢測 (20K線回看 + 斜率判斷)", fontsize=16, color="white"
)
ax.set_xlabel("時間", fontsize=12, color="white")
ax.set_ylabel("價格 (USD)", fontsize=12, color="white")
ax.legend()
ax.grid(True, alpha=0.3)

# 設置x軸間隔
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()

# 保存
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"btc_reversals_improved_{timestamp}.png"
plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="black")
print(f"\n📊 改進版圖表已保存: {filename}")

# 保存數據
if len(reversals_df) > 0:
    reversals_df.to_csv("btc_reversals_improved.csv", index=False)
    print(f"📄 改進版數據已保存: btc_reversals_improved.csv")

plt.show()
