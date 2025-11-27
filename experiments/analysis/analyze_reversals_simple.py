import pandas as pd
import numpy as np

# 讀取檢測結果
reversals_df = pd.read_csv("btc_reversals_final.csv")
reversals_df["timestamp"] = pd.to_datetime(reversals_df["timestamp"])

print(f"📊 轉折點分析 (總共 {len(reversals_df)} 個)")

# 分析斜率數據
print(f"\n🔍 斜率分析:")
for _, row in reversals_df.iterrows():
    if pd.notna(row.get("slopes")):
        slopes = row["slopes"].split("→")
        if len(slopes) == 2:
            before = float(slopes[0])
            after = float(slopes[1])

            # 檢查斜率邏輯
            if row["type"] == "A_turn":
                if before <= 0 or after >= 0:
                    print(
                        f"❌ A轉斜率異常: {row['timestamp']} @ {row['price']:.0f}, 斜率 {row['slopes']}"
                    )
            else:  # V_turn
                if before >= 0 or after <= 0:
                    print(
                        f"❌ V轉斜率異常: {row['timestamp']} @ {row['price']:.0f}, 斜率 {row['slopes']}"
                    )

# 分析變化幅度
print(f"\n📈 變化幅度分析:")
a_turns = reversals_df[reversals_df["type"] == "A_turn"]
v_turns = reversals_df[reversals_df["type"] == "V_turn"]

print(f"A轉: {len(a_turns)} 個, 平均變化 {a_turns['change'].mean():.2%}")
print(f"V轉: {len(v_turns)} 個, 平均變化 {v_turns['change'].mean():.2%}")

# 找出變化幅度小的點
small_changes = reversals_df[abs(reversals_df["change"]) < 0.01]
print(f"\n⚠️  變化幅度<1%的轉折點: {len(small_changes)} 個")
for _, row in small_changes.tail(5).iterrows():
    print(
        f"  {row['timestamp']}: {row['type']} @ {row['price']:.0f}, 變化 {row['change']:.2%}"
    )

# 分析時間間隔
reversals_df = reversals_df.sort_values("timestamp")
time_diffs = reversals_df["timestamp"].diff().dt.total_seconds() / 3600  # 小時

print(f"\n⏰ 時間間隔分析:")
print(f"平均間隔: {time_diffs.mean():.1f} 小時")
print(f"最短間隔: {time_diffs.min():.1f} 小時")

# 找出間隔太短的點
short_intervals = reversals_df[time_diffs < 2]  # 2小時內
print(f"\n⚠️  間隔<2小時的轉折點: {len(short_intervals)} 個")
for i, row in short_intervals.tail(5).iterrows():
    if i > 0:
        prev_row = reversals_df.iloc[i - 1]
        interval = (
            row["timestamp"] - prev_row["timestamp"]
        ).total_seconds() / 3600
        print(
            f"  {row['timestamp']}: {row['type']} @ {row['price']:.0f}, 距離上個點 {interval:.1f}h"
        )

# 最近的轉折點詳細分析
print(f"\n🔍 最近10個轉折點詳細分析:")
for _, row in reversals_df.tail(10).iterrows():
    slopes_info = (
        f", 斜率 {row.get('slopes', 'N/A')}"
        if pd.notna(row.get("slopes"))
        else ""
    )
    print(
        f"  {row['timestamp']}: {row['type']} @ {row['price']:.0f}, 變化 {row['change']:.2%}{slopes_info}"
    )
