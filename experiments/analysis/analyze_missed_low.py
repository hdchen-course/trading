import pandas as pd
import numpy as np

# 讀取數據
df = pd.read_csv("training_data/btc_12months_15m.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 檢查9/25-9/26的低點
target_date = pd.to_datetime("2025-09-25 17:45:00")
target_idx = df[df["timestamp"] == target_date].index[0]

print(f"🔍 分析 {target_date} 低點 (價格: {df.loc[target_idx, 'low']:.2f})")

# 檢查20K線窗口
lookback = 20
window_start = max(0, target_idx - lookback)
window_end = min(len(df), target_idx + lookback)
window_data = df.iloc[window_start : window_end + 1]

current_price = df.loc[target_idx, "close"]
is_local_low = current_price <= window_data["low"].min() * 1.001

print(f"當前價格: {current_price:.2f}")
print(f"窗口最低價: {window_data['low'].min():.2f}")
print(f"是否為局部最低點: {is_local_low}")

# 檢查斜率
before_prices = df.iloc[target_idx - 5 : target_idx]["close"].values
after_prices = df.iloc[target_idx + 1 : target_idx + 6]["close"].values

if len(before_prices) >= 3 and len(after_prices) >= 3:
    before_slope = (before_prices[-1] - before_prices[0]) / len(before_prices)
    after_slope = (after_prices[-1] - after_prices[0]) / len(after_prices)

    print(f"前5根K線價格: {before_prices}")
    print(f"後5根K線價格: {after_prices}")
    print(f"前斜率: {before_slope:.1f} (需要<0)")
    print(f"後斜率: {after_slope:.1f} (需要>0)")

    # 檢查變化幅度
    max_gain = (max(after_prices) - current_price) / current_price
    min_change = 0.012 if current_price > 100000 else 0.015

    print(f"最大漲幅: {max_gain:.2%} (需要>={min_change:.1%})")

    # 總結
    slope_ok = before_slope < 0 and after_slope > 0
    change_ok = max_gain >= min_change

    print(f"\n📊 檢測結果:")
    print(f"局部最低點: {is_local_low}")
    print(f"斜率條件: {slope_ok}")
    print(f"變化條件: {change_ok}")
    print(f"應該被檢測: {is_local_low and slope_ok and change_ok}")

# 檢查這個區域的其他低點
print(f"\n🔍 9/25-9/26區域所有低點:")
area_start = pd.to_datetime("2025-09-25 00:00:00")
area_end = pd.to_datetime("2025-09-27 00:00:00")
area_data = df[(df["timestamp"] >= area_start) & (df["timestamp"] <= area_end)]

low_points = area_data.nsmallest(10, "low")
for _, row in low_points.iterrows():
    print(
        f"  {row['timestamp']}: 低點 {row['low']:.2f}, 收盤 {row['close']:.2f}"
    )

# 檢查當前檢測結果中這個時期的點
reversals_df = pd.read_csv("btc_reversals_final.csv")
reversals_df["timestamp"] = pd.to_datetime(reversals_df["timestamp"])

area_reversals = reversals_df[
    (reversals_df["timestamp"] >= area_start)
    & (reversals_df["timestamp"] <= area_end)
]
print(f"\n✅ 當前檢測到的轉折點:")
for _, row in area_reversals.iterrows():
    print(
        f"  {row['timestamp']}: {row['type']} @ {row['price']:.2f}, 變化 {row['change']:.2%}"
    )

if len(area_reversals) == 0:
    print("  ❌ 這個區域沒有檢測到任何轉折點！")
