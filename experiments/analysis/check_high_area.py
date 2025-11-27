import pandas as pd

# 讀取數據
df = pd.read_csv("training_data/btc_12months_15m.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
reversals_df = pd.read_csv("btc_reversals_final.csv")
reversals_df["timestamp"] = pd.to_datetime(reversals_df["timestamp"])

# 檢查9/15-9/20期間
start_date = pd.to_datetime("2025-09-15")
end_date = pd.to_datetime("2025-09-20")

period_data = df[
    (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
]
period_reversals = reversals_df[
    (reversals_df["timestamp"] >= start_date)
    & (reversals_df["timestamp"] <= end_date)
]

print(f"📊 9/15-9/20期間分析:")
print(f"檢測到的轉折點: {len(period_reversals)}")

# 找出期間內的最高點
max_price = period_data["high"].max()
max_idx = period_data["high"].idxmax()
max_time = period_data.loc[max_idx, "timestamp"]

print(f"\n📈 期間最高點:")
print(f"時間: {max_time}")
print(f"價格: {max_price:.2f}")

# 檢查期間內的高點
print(f"\n🔍 期間內前10個高點:")
high_points = period_data.nlargest(10, "high")
for _, row in high_points.iterrows():
    is_detected = any(
        abs((rev_time - row["timestamp"]).total_seconds()) < 3600
        for rev_time in period_reversals["timestamp"]
    )
    status = "✅已檢測" if is_detected else "❌未檢測"
    print(f"  {row['timestamp']}: {row['high']:.2f} {status}")

print(f"\n✅ 當前檢測到的轉折點:")
for _, row in period_reversals.iterrows():
    print(
        f"  {row['timestamp']}: {row['type']} @ {row['price']:.2f}, 變化 {row['change']:.2%}"
    )

if len(period_reversals) == 0:
    print("  ❌ 這個區域沒有檢測到任何轉折點！")

# 檢查更大範圍的高點區域
print(f"\n🔍 9/10-9/25期間高點分析:")
extended_start = pd.to_datetime("2025-09-10")
extended_end = pd.to_datetime("2025-09-25")
extended_data = df[
    (df["timestamp"] >= extended_start) & (df["timestamp"] <= extended_end)
]
extended_reversals = reversals_df[
    (reversals_df["timestamp"] >= extended_start)
    & (reversals_df["timestamp"] <= extended_end)
]

extended_highs = extended_data.nlargest(5, "high")
print("前5個最高點:")
for _, row in extended_highs.iterrows():
    is_detected = any(
        abs((rev_time - row["timestamp"]).total_seconds()) < 3600
        for rev_time in extended_reversals["timestamp"]
    )
    status = "✅已檢測" if is_detected else "❌未檢測"
    print(f"  {row['timestamp']}: {row['high']:.2f} {status}")
