import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def analyze_reversal_quality(df, reversals):
    """分析轉折點質量"""
    analysis = []

    for reversal in reversals:
        idx = reversal["index"]
        price = reversal["price"]
        rev_type = reversal["type"]

        # 檢查前後更大的窗口 (40K線)
        start_idx = max(0, idx - 40)
        end_idx = min(len(df), idx + 40)

        window_data = df.iloc[start_idx:end_idx]

        # 檢查是否真的是極值
        if rev_type == "A_turn":
            is_true_peak = price >= window_data["high"].max() * 0.995
            # 檢查後續是否有更高點
            future_max = (
                df.iloc[idx + 1 : idx + 80]["high"].max()
                if idx + 80 < len(df)
                else 0
            )
            has_higher_later = future_max > price * 1.01
        else:  # V_turn
            is_true_valley = price <= window_data["low"].min() * 1.005
            # 檢查後續是否有更低點
            future_min = (
                df.iloc[idx + 1 : idx + 80]["low"].min()
                if idx + 80 < len(df)
                else float("inf")
            )
            has_lower_later = future_min < price * 0.99

        # 計算前後趨勢強度
        before_trend = (
            df.iloc[idx]["close"] - df.iloc[max(0, idx - 20)]["close"]
        ) / df.iloc[max(0, idx - 20)]["close"]
        after_trend = (
            df.iloc[min(len(df) - 1, idx + 20)]["close"]
            - df.iloc[idx]["close"]
        ) / df.iloc[idx]["close"]

        quality_score = 0
        issues = []

        if rev_type == "A_turn":
            if not is_true_peak:
                issues.append("非40K線最高點")
            if has_higher_later:
                issues.append("後續有更高點")
                quality_score -= 2
            if before_trend < 0.005:  # 前面沒明顯上升
                issues.append("前趨勢不明顯")
                quality_score -= 1
            if after_trend > -0.005:  # 後面沒明顯下降
                issues.append("後趨勢不明顯")
                quality_score -= 1
        else:  # V_turn
            if not is_true_valley:
                issues.append("非40K線最低點")
            if has_lower_later:
                issues.append("後續有更低點")
                quality_score -= 2
            if before_trend > -0.005:  # 前面沒明顯下降
                issues.append("前趨勢不明顯")
                quality_score -= 1
            if after_trend < 0.005:  # 後面沒明顯上升
                issues.append("後趨勢不明顯")
                quality_score -= 1

        analysis.append(
            {
                "timestamp": reversal["timestamp"],
                "type": rev_type,
                "price": price,
                "quality_score": quality_score,
                "issues": issues,
                "before_trend": before_trend,
                "after_trend": after_trend,
            }
        )

    return analysis


def find_missed_reversals(df):
    """找出可能遺漏的重要轉折點"""
    missed = []

    # 找出價格的重要高低點
    for i in range(50, len(df) - 50):
        current_price = df.iloc[i]["close"]

        # 檢查是否為50K線窗口的極值
        window = df.iloc[i - 50 : i + 50]
        is_highest = current_price >= window["high"].max() * 0.998
        is_lowest = current_price <= window["low"].min() * 1.002

        if is_highest or is_lowest:
            # 檢查是否已被檢測到
            timestamp = df.iloc[i]["timestamp"]
            found = False
            for rev in reversals:
                if (
                    abs(
                        (
                            pd.to_datetime(rev["timestamp"])
                            - pd.to_datetime(timestamp)
                        ).total_seconds()
                    )
                    < 3600
                ):  # 1小時內
                    found = True
                    break

            if not found:
                rev_type = "A_turn" if is_highest else "V_turn"
                # 計算變化幅度
                if is_highest:
                    future_min = df.iloc[i + 1 : i + 50]["low"].min()
                    change = (future_min - current_price) / current_price
                else:
                    future_max = df.iloc[i + 1 : i + 50]["high"].max()
                    change = (future_max - current_price) / current_price

                if abs(change) > 0.01:  # 1%以上變化
                    missed.append(
                        {
                            "timestamp": timestamp,
                            "type": rev_type,
                            "price": current_price,
                            "potential_change": change,
                        }
                    )

    return missed


# 讀取數據
df = pd.read_csv("btc_15min_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 讀取檢測結果
reversals_df = pd.read_csv("btc_reversals_final.csv")
reversals = reversals_df.to_dict("records")

print("🔍 分析轉折點質量...")
analysis = analyze_reversal_quality(df, reversals)

# 找出問題轉折點
problem_reversals = [
    a for a in analysis if a["quality_score"] < 0 or len(a["issues"]) > 0
]
print(f"\n❌ 有問題的轉折點: {len(problem_reversals)}/{len(analysis)}")

for rev in problem_reversals[-10:]:  # 顯示最近10個問題點
    print(f"  {rev['timestamp']}: {rev['type']} @ {rev['price']:.2f}")
    print(f"    問題: {', '.join(rev['issues'])}")
    print(f"    趨勢: {rev['before_trend']:.2%} → {rev['after_trend']:.2%}")

print(f"\n🔍 尋找遺漏的轉折點...")
missed = find_missed_reversals(df)
print(f"可能遺漏的重要轉折點: {len(missed)}")

for rev in missed[-5:]:  # 顯示最近5個遺漏點
    print(
        f"  {rev['timestamp']}: {rev['type']} @ {rev['price']:.2f}, 潛在變化 {rev['potential_change']:.2%}"
    )

# 統計分析
good_reversals = [
    a for a in analysis if a["quality_score"] >= 0 and len(a["issues"]) == 0
]
print(f"\n📊 質量統計:")
print(
    f"高質量轉折點: {len(good_reversals)}/{len(analysis)} ({len(good_reversals)/len(analysis)*100:.1f}%)"
)
print(f"平均前趨勢強度: {np.mean([a['before_trend'] for a in analysis]):.2%}")
print(f"平均後趨勢強度: {np.mean([a['after_trend'] for a in analysis]):.2%}")
