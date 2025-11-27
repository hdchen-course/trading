import pandas as pd
import numpy as np

# 讀取原始數據和檢測結果
df = pd.read_csv("training_data/btc_12months_15m.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

reversals_df = pd.read_csv("btc_reversals_final.csv")
reversals_df["timestamp"] = pd.to_datetime(reversals_df["timestamp"])

# 檢查9/20到10/11期間
start_date = pd.to_datetime("2025-09-20")
end_date = pd.to_datetime("2025-10-11")

period_data = df[
    (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
]
period_reversals = reversals_df[
    (reversals_df["timestamp"] >= start_date)
    & (reversals_df["timestamp"] <= end_date)
]

print(f"📊 9/20-10/11期間分析:")
print(f"數據點數: {len(period_data)}")
print(f"檢測到的轉折點: {len(period_reversals)}")

# 找出期間內的最低點
min_price = period_data["low"].min()
min_idx = period_data["low"].idxmin()
min_time = period_data.loc[min_idx, "timestamp"]

print(f"\n📉 期間最低點:")
print(f"時間: {min_time}")
print(f"價格: {min_price:.2f}")

# 檢查這個最低點前後的數據
min_row_idx = period_data.index.get_loc(min_idx)
context_start = max(0, min_row_idx - 20)
context_end = min(len(period_data), min_row_idx + 20)
context_data = period_data.iloc[context_start:context_end]

print(f"\n🔍 最低點前後20根K線:")
for i, row in context_data.iterrows():
    marker = " ⭐" if i == min_idx else ""
    print(
        f"  {row['timestamp']}: {row['low']:.2f} - {row['high']:.2f}{marker}"
    )

# 檢查為什麼沒被檢測到
print(f"\n❓ 檢測邏輯分析:")
print("當前檢測條件:")
print("1. 必須是20K線窗口內的局部最低點")
print("2. 前5根K線斜率為負，後5根K線斜率為正")
print("3. 變化幅度 >= 1.2% (>100K價格)")
print("4. 與前一個轉折點間隔 >= 2小時")

# 模擬檢測邏輯
lookback = 20
target_idx = period_data.index.get_loc(min_idx)

if target_idx >= lookback and target_idx < len(period_data) - lookback:
    current_price = period_data.loc[min_idx, "close"]

    # 檢查局部最低點
    window_start = target_idx - lookback
    window_end = target_idx + lookback
    window_data = period_data.iloc[window_start : window_end + 1]
    is_local_low = current_price <= window_data["low"].min() * 1.001

    # 檢查斜率
    before_prices = period_data.iloc[target_idx - 5 : target_idx][
        "close"
    ].values
    after_prices = period_data.iloc[target_idx + 1 : target_idx + 6][
        "close"
    ].values

    if len(before_prices) >= 3 and len(after_prices) >= 3:
        before_slope = (before_prices[-1] - before_prices[0]) / len(
            before_prices
        )
        after_slope = (after_prices[-1] - after_prices[0]) / len(after_prices)

        # 檢查變化幅度
        max_gain = (max(after_prices) - current_price) / current_price
        min_change = 0.012 if current_price > 100000 else 0.015

        print(f"\n🔍 最低點檢測結果:")
        print(f"是否為局部最低點: {is_local_low}")
        print(f"前斜率: {before_slope:.1f} (需要<0)")
        print(f"後斜率: {after_slope:.1f} (需要>0)")
        print(f"最大漲幅: {max_gain:.2%} (需要>={min_change:.1%})")
        print(f"斜率條件: {before_slope < 0 and after_slope > 0}")
        print(f"變化條件: {max_gain >= min_change}")

# 檢查其他可能的低點
print(f"\n🔍 期間內其他低點:")
period_sorted = period_data.nsmallest(5, "low")
for _, row in period_sorted.iterrows():
    is_detected = any(
        abs((rev_time - row["timestamp"]).total_seconds()) < 3600
        for rev_time in period_reversals["timestamp"]
    )
    status = "✅已檢測" if is_detected else "❌未檢測"
    print(f"  {row['timestamp']}: {row['low']:.2f} {status}")
