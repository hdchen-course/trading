#!/usr/bin/env python3
"""
分析布林通道和EMA的支撐壓力效果
"""

import pandas as pd
import numpy as np
import talib


def calculate_indicators(df):
    """計算技術指標"""
    df = df.copy()

    # 布林通道 (48, 2.4) - 與策略一致
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(
        df["close"], timeperiod=48, nbdevup=2.4, nbdevdn=2.4
    )

    # EMA
    df["ema_50"] = talib.EMA(df["close"], timeperiod=50)
    df["ema_100"] = talib.EMA(df["close"], timeperiod=100)

    # StochRSI (14, 3, 3) - 與策略一致
    df["stochrsi_k"], df["stochrsi_d"] = talib.STOCHRSI(
        df["close"], timeperiod=14, fastk_period=3, fastd_period=3
    )

    return df


def detect_touch_events(df, tolerance=0.002):  # 0.2%容忍度
    """檢測價格觸碰技術指標的事件"""
    events = []

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i - 1]

        # 檢測布林上軌觸碰
        if current["high"] >= current["bb_upper"] * (1 - tolerance) and prev[
            "high"
        ] < prev["bb_upper"] * (1 - tolerance):
            events.append(
                {
                    "timestamp": current["timestamp"],
                    "type": "bb_upper_touch",
                    "price": current["high"],
                    "level": current["bb_upper"],
                    "index": i,
                }
            )

        # 檢測布林下軌觸碰
        if current["low"] <= current["bb_lower"] * (1 + tolerance) and prev[
            "low"
        ] > prev["bb_lower"] * (1 + tolerance):
            events.append(
                {
                    "timestamp": current["timestamp"],
                    "type": "bb_lower_touch",
                    "price": current["low"],
                    "level": current["bb_lower"],
                    "index": i,
                }
            )

        # 檢測布林中軸觸碰
        bb_mid_upper = current["bb_middle"] * (1 + tolerance)
        bb_mid_lower = current["bb_middle"] * (1 - tolerance)
        if (
            bb_mid_lower <= current["low"] <= bb_mid_upper
            or bb_mid_lower <= current["high"] <= bb_mid_upper
        ):
            events.append(
                {
                    "timestamp": current["timestamp"],
                    "type": "bb_middle_touch",
                    "price": current["close"],
                    "level": current["bb_middle"],
                    "index": i,
                }
            )

        # 檢測EMA50觸碰
        if pd.notna(current["ema_50"]):
            ema50_upper = current["ema_50"] * (1 + tolerance)
            ema50_lower = current["ema_50"] * (1 - tolerance)
            if (
                ema50_lower <= current["low"] <= ema50_upper
                or ema50_lower <= current["high"] <= ema50_upper
            ):
                events.append(
                    {
                        "timestamp": current["timestamp"],
                        "type": "ema_50_touch",
                        "price": current["close"],
                        "level": current["ema_50"],
                        "index": i,
                    }
                )

        # 檢測EMA100觸碰
        if pd.notna(current["ema_100"]):
            ema100_upper = current["ema_100"] * (1 + tolerance)
            ema100_lower = current["ema_100"] * (1 - tolerance)
            if (
                ema100_lower <= current["low"] <= ema100_upper
                or ema100_lower <= current["high"] <= ema100_upper
            ):
                events.append(
                    {
                        "timestamp": current["timestamp"],
                        "type": "ema_100_touch",
                        "price": current["close"],
                        "level": current["ema_100"],
                        "index": i,
                    }
                )

    return pd.DataFrame(events)


def analyze_support_resistance(df, events_df, lookback=5):
    """分析支撐壓力效果"""
    results = []

    for _, event in events_df.iterrows():
        idx = event["index"]
        event_type = event["type"]

        # 確保有足夠的後續數據
        if idx + lookback >= len(df):
            continue

        # 獲取觸碰後的價格數據
        future_data = df.iloc[idx + 1 : idx + 1 + lookback]

        # 判斷是支撐還是壓力
        if "upper" in event_type or event_type in [
            "bb_middle_touch",
            "ema_50_touch",
            "ema_100_touch",
        ]:
            # 上軌或中軸/EMA - 檢查是否形成壓力
            resistance_count = sum(
                future_data["high"] < event["level"] * 0.998
            )  # 未能突破
            support_count = sum(
                future_data["low"] > event["level"] * 1.002
            )  # 形成支撐
        else:
            # 下軌 - 檢查是否形成支撐
            support_count = sum(
                future_data["low"] > event["level"] * 1.002
            )  # 形成支撐
            resistance_count = sum(
                future_data["high"] < event["level"] * 0.998
            )  # 形成壓力

        # 計算後續價格變化
        future_returns = []
        for i in range(1, min(lookback + 1, len(future_data) + 1)):
            if idx + i < len(df):
                ret = (df.iloc[idx + i]["close"] - event["price"]) / event[
                    "price"
                ]
                future_returns.append(ret)

        # 計算其他技術指標
        current_data = df.iloc[idx]

        # StochRSI
        stochrsi_k = (
            current_data["stochrsi_k"]
            if pd.notna(current_data["stochrsi_k"])
            else None
        )
        stochrsi_d = (
            current_data["stochrsi_d"]
            if pd.notna(current_data["stochrsi_d"])
            else None
        )

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df["close"].values)

        # 成交量比率
        volume_sma = df["volume"].rolling(20).mean()
        volume_ratio = df["volume"] / volume_sma

        results.append(
            {
                "timestamp": event["timestamp"],
                "type": event_type,
                "price": event["price"],
                "level": event["level"],
                "support_count": support_count,
                "resistance_count": resistance_count,
                "max_return_1": (
                    max(future_returns[:1]) if future_returns else 0
                ),
                "min_return_1": (
                    min(future_returns[:1]) if future_returns else 0
                ),
                "max_return_3": (
                    max(future_returns[:3]) if len(future_returns) >= 3 else 0
                ),
                "min_return_3": (
                    min(future_returns[:3]) if len(future_returns) >= 3 else 0
                ),
                "max_return_5": max(future_returns) if future_returns else 0,
                "min_return_5": min(future_returns) if future_returns else 0,
                "stochrsi_k": stochrsi_k,
                "stochrsi_d": stochrsi_d,
                "macd": (
                    macd[idx]
                    if idx < len(macd) and pd.notna(macd[idx])
                    else None
                ),
                "macd_signal": (
                    macd_signal[idx]
                    if idx < len(macd_signal) and pd.notna(macd_signal[idx])
                    else None
                ),
                "macd_hist": (
                    macd_hist[idx]
                    if idx < len(macd_hist) and pd.notna(macd_hist[idx])
                    else None
                ),
                "volume_ratio": (
                    volume_ratio.iloc[idx]
                    if pd.notna(volume_ratio.iloc[idx])
                    else None
                ),
                "bb_position": (
                    current_data["close"] - current_data["bb_lower"]
                )
                / (current_data["bb_upper"] - current_data["bb_lower"]),
            }
        )

    return pd.DataFrame(results)


def generate_statistics(analysis_df):
    """生成統計報告"""
    print("=== 支撐壓力分析報告 ===\n")

    # 按類型統計
    for event_type in analysis_df["type"].unique():
        type_data = analysis_df[analysis_df["type"] == event_type]
        print(f"📊 {event_type.upper()} 統計 (共{len(type_data)}次):")

        # 支撐/壓力效果
        avg_support = type_data["support_count"].mean()
        avg_resistance = type_data["resistance_count"].mean()
        print(f"   平均支撐效果: {avg_support:.2f}/5")
        print(f"   平均壓力效果: {avg_resistance:.2f}/5")

        # 價格反應
        print(f"   1期後最大漲幅: {type_data['max_return_1'].mean()*100:.2f}%")
        print(f"   1期後最大跌幅: {type_data['min_return_1'].mean()*100:.2f}%")
        print(f"   5期後最大漲幅: {type_data['max_return_5'].mean()*100:.2f}%")
        print(f"   5期後最大跌幅: {type_data['min_return_5'].mean()*100:.2f}%")

        # 技術指標統計
        if type_data["stochrsi_k"].notna().any():
            print(f"   平均StochRSI K: {type_data['stochrsi_k'].mean():.1f}")
        if type_data["stochrsi_d"].notna().any():
            print(f"   平均StochRSI D: {type_data['stochrsi_d'].mean():.1f}")
        if type_data["volume_ratio"].notna().any():
            print(f"   平均成交量比率: {type_data['volume_ratio'].mean():.2f}")
        if type_data["bb_position"].notna().any():
            print(f"   平均布林位置: {type_data['bb_position'].mean():.2f}")

        print()


def main():
    # 載入數據
    print("載入數據...")
    df = pd.read_csv("training_data/btc_12months_15m.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 計算技術指標
    print("計算技術指標...")
    df = calculate_indicators(df)

    # 檢測觸碰事件
    print("檢測觸碰事件...")
    events_df = detect_touch_events(df)
    print(f"發現 {len(events_df)} 個觸碰事件")

    # 分析支撐壓力效果
    print("分析支撐壓力效果...")
    analysis_df = analyze_support_resistance(df, events_df)

    # 生成統計報告
    generate_statistics(analysis_df)

    # 保存詳細結果
    analysis_df.to_csv("support_resistance_analysis.csv", index=False)
    print("詳細分析結果已保存至: support_resistance_analysis.csv")


if __name__ == "__main__":
    main()
