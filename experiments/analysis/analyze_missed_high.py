import pandas as pd

# 讀取數據
df = pd.read_csv("training_data/btc_12months_15m.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 檢查9/18的高點
target_date = pd.to_datetime("2025-09-18 18:15:00")
target_idx = df[df["timestamp"] == target_date].index[0]

print(f"🔍 分析 {target_date} 高點 (價格: {df.loc[target_idx, 'high']:.2f})")

# 檢查20K線窗口
lookback = 20
window_start = max(0, target_idx - lookback)
window_end = min(len(df), target_idx + lookback)
window_data = df.iloc[window_start : window_end + 1]

current_price = df.loc[target_idx, "close"]
is_local_high = current_price >= window_data["high"].max() * 0.999

print(f"當前收盤價: {current_price:.2f}")
print(f"當前最高價: {df.loc[target_idx, 'high']:.2f}")
print(f"窗口最高價: {window_data['high'].max():.2f}")
print(f"是否為局部最高點: {is_local_high}")

# 檢查斜率
before_prices = df.iloc[target_idx - 5 : target_idx]["close"].values
after_prices = df.iloc[target_idx + 1 : target_idx + 6]["close"].values

if len(before_prices) >= 3 and len(after_prices) >= 3:
    before_slope = (before_prices[-1] - before_prices[0]) / len(before_prices)
    after_slope = (after_prices[-1] - after_prices[0]) / len(after_prices)

    print(f"前5根K線價格: {before_prices}")
    print(f"後5根K線價格: {after_prices}")
    print(f"前斜率: {before_slope:.1f} (需要>0)")
    print(f"後斜率: {after_slope:.1f} (需要<0)")

    # 檢查變化幅度
    max_decline = (min(after_prices) - current_price) / current_price

    # 檢查是否為重要區域
    extended_window = df.iloc[
        max(0, target_idx - 80) : min(len(df), target_idx + 80)
    ]
    price_range = current_price * 0.02
    similar_prices = extended_window[
        (extended_window["high"] >= current_price - price_range)
        & (extended_window["high"] <= current_price + price_range)
    ]

    base_change = 0.012
    if len(similar_prices) >= 5:
        min_change = base_change * 0.7
        print(f"重要區域檢測: 是 (相似價格點: {len(similar_prices)})")
    else:
        min_change = base_change
        print(f"重要區域檢測: 否 (相似價格點: {len(similar_prices)})")

    print(f"最大跌幅: {max_decline:.2%} (需要<=-{min_change:.1%})")

    # 總結
    slope_ok = before_slope > 0 and after_slope < 0
    change_ok = abs(max_decline) >= min_change

    print(f"\n📊 檢測結果:")
    print(f"局部最高點: {is_local_high}")
    print(f"斜率條件: {slope_ok}")
    print(f"變化條件: {change_ok}")
    print(f"應該被檢測: {is_local_high and slope_ok and change_ok}")

# 檢查這個時期前後的轉折點
print(f"\n🔍 9/15-9/25期間檢測到的轉折點:")
reversals_df = pd.read_csv("btc_reversals_final.csv")
reversals_df["timestamp"] = pd.to_datetime(reversals_df["timestamp"])

area_start = pd.to_datetime("2025-09-15")
area_end = pd.to_datetime("2025-09-25")
area_reversals = reversals_df[
    (reversals_df["timestamp"] >= area_start)
    & (reversals_df["timestamp"] <= area_end)
]

for _, row in area_reversals.iterrows():
    print(
        f"  {row['timestamp']}: {row['type']} @ {row['price']:.2f}, 變化 {row['change']:.2%}"
    )

if len(area_reversals) == 0:
    print("  ❌ 這個區域沒有檢測到任何轉折點！")
