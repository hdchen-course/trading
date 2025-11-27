#!/usr/bin/env python3
"""
分析A轉(尖頂)和V轉(尖底)的技術指標特徵
"""

import pandas as pd
import numpy as np
import talib


def calculate_indicators(df):
    """計算所有技術指標"""
    df = df.copy()

    # 布林通道
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(
        df["close"], timeperiod=48, nbdevup=2.4, nbdevdn=2.4
    )

    # StochRSI
    df["stochrsi_k"], df["stochrsi_d"] = talib.STOCHRSI(
        df["close"], timeperiod=14, fastk_period=3, fastd_period=3
    )

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(df["close"])

    # 成交量比率
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    # 布林位置
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )

    return df


def detect_reversal_patterns(df, lookback=5, min_change=0.02):
    """檢測A轉和V轉模式"""
    reversals = []

    for i in range(lookback, len(df) - lookback):
        current_idx = i

        # 獲取前後價格
        before_prices = df["close"].iloc[i - lookback : i].values
        after_prices = df["close"].iloc[i + 1 : i + 1 + lookback].values
        current_price = df["close"].iloc[i]

        # A轉檢測 (尖頂反轉)
        if (
            current_price
            == max(
                df["close"].iloc[i - lookback : i + 1 + lookback]
            )  # 是區間最高點
            and all(before_prices[-3:] < current_price)  # 前面上漲
            and all(after_prices[:3] < current_price)  # 後面下跌
            and (current_price - min(after_prices)) / current_price
            >= min_change
        ):  # 跌幅足夠

            reversals.append(
                {
                    "timestamp": df["timestamp"].iloc[i],
                    "type": "A_turn",
                    "price": current_price,
                    "index": i,
                    "change_5": (min(after_prices) - current_price)
                    / current_price,
                }
            )

        # V轉檢測 (尖底反轉)
        if (
            current_price
            == min(
                df["close"].iloc[i - lookback : i + 1 + lookback]
            )  # 是區間最低點
            and all(before_prices[-3:] > current_price)  # 前面下跌
            and all(after_prices[:3] > current_price)  # 後面上漲
            and (max(after_prices) - current_price) / current_price
            >= min_change
        ):  # 漲幅足夠

            reversals.append(
                {
                    "timestamp": df["timestamp"].iloc[i],
                    "type": "V_turn",
                    "price": current_price,
                    "index": i,
                    "change_5": (max(after_prices) - current_price)
                    / current_price,
                }
            )

    return pd.DataFrame(reversals)


def analyze_reversal_indicators(df, reversals_df):
    """分析反轉點的技術指標特徵"""
    results = []

    for _, reversal in reversals_df.iterrows():
        idx = reversal["index"]
        current_data = df.iloc[idx]

        results.append(
            {
                "timestamp": reversal["timestamp"],
                "type": reversal["type"],
                "price": reversal["price"],
                "change_5": reversal["change_5"],
                "stochrsi_k": current_data["stochrsi_k"],
                "stochrsi_d": current_data["stochrsi_d"],
                "macd": current_data["macd"],
                "macd_signal": current_data["macd_signal"],
                "macd_hist": current_data["macd_hist"],
                "volume_ratio": current_data["volume_ratio"],
                "bb_position": current_data["bb_position"],
                "bb_upper_dist": (
                    current_data["bb_upper"] - current_data["close"]
                )
                / current_data["close"],
                "bb_lower_dist": (
                    current_data["close"] - current_data["bb_lower"]
                )
                / current_data["close"],
                "bb_middle_dist": abs(
                    current_data["close"] - current_data["bb_middle"]
                )
                / current_data["close"],
            }
        )

    return pd.DataFrame(results)


def generate_reversal_statistics(analysis_df):
    """生成反轉模式統計"""
    print("=== A轉/V轉模式分析 ===\n")

    for pattern_type in ["A_turn", "V_turn"]:
        pattern_data = analysis_df[
            analysis_df["type"] == pattern_type
        ].dropna()
        if len(pattern_data) == 0:
            continue

        print(f"📊 {pattern_type.upper()} 統計 (共{len(pattern_data)}次):")
        print(f"   平均變化幅度: {pattern_data['change_5'].mean()*100:.2f}%")
        print(f"   最大變化幅度: {pattern_data['change_5'].max()*100:.2f}%")

        # StochRSI特徵
        print(f"   平均StochRSI K: {pattern_data['stochrsi_k'].mean():.1f}")
        print(f"   平均StochRSI D: {pattern_data['stochrsi_d'].mean():.1f}")

        # MACD特徵
        print(f"   平均MACD柱: {pattern_data['macd_hist'].mean():.2f}")
        macd_positive = (pattern_data["macd_hist"] > 0).mean() * 100
        print(f"   MACD柱正值比例: {macd_positive:.1f}%")

        # 成交量特徵
        print(f"   平均成交量比率: {pattern_data['volume_ratio'].mean():.2f}")
        high_volume = (pattern_data["volume_ratio"] > 1.5).mean() * 100
        print(f"   大成交量比例: {high_volume:.1f}%")

        # 布林通道特徵
        print(f"   平均布林位置: {pattern_data['bb_position'].mean():.2f}")
        print(f"   距離上軌: {pattern_data['bb_upper_dist'].mean()*100:.2f}%")
        print(f"   距離下軌: {pattern_data['bb_lower_dist'].mean()*100:.2f}%")
        print(f"   距離中軸: {pattern_data['bb_middle_dist'].mean()*100:.2f}%")

        # 極端值分析
        if pattern_type == "A_turn":
            extreme = pattern_data[pattern_data["stochrsi_k"] > 80]
            print(
                f"   StochRSI>80時 ({len(extreme)}次): 平均跌幅 {extreme['change_5'].mean()*100:.2f}%"
            )
        else:
            extreme = pattern_data[pattern_data["stochrsi_k"] < 20]
            print(
                f"   StochRSI<20時 ({len(extreme)}次): 平均漲幅 {extreme['change_5'].mean()*100:.2f}%"
            )

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

    # 檢測反轉模式
    print("檢測反轉模式...")
    reversals_df = detect_reversal_patterns(
        df, lookback=5, min_change=0.015
    )  # 1.5%最小變化
    print(f"發現 {len(reversals_df)} 個反轉模式")

    # 分析指標特徵
    print("分析指標特徵...")
    analysis_df = analyze_reversal_indicators(df, reversals_df)

    # 生成統計報告
    generate_reversal_statistics(analysis_df)

    # 保存結果
    analysis_df.to_csv("reversal_patterns_analysis.csv", index=False)
    print("詳細分析結果已保存至: reversal_patterns_analysis.csv")


if __name__ == "__main__":
    main()
