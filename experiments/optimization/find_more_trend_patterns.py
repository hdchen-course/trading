#!/usr/bin/env python3
"""
尋找更多趨勢模式 - 放寬條件找到足夠的訓練樣本
"""

import pandas as pd
import numpy as np
import talib


def load_and_prepare_data():
    """載入並準備數據"""
    df = pd.read_csv("training_data/btc_12months_15m.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 基本特徵
    df["price_change"] = df["close"].pct_change()

    # 布林帶
    bb_upper, bb_middle, bb_lower = talib.BBANDS(
        df["close"], timeperiod=48, nbdevup=2.4, nbdevdn=2.4
    )
    df["bb_middle"] = bb_middle
    df["distance_to_middle"] = (df["close"] - bb_middle) / bb_middle

    # EMA
    df["ema_50"] = talib.EMA(df["close"], 50)
    df["ema_100"] = talib.EMA(df["close"], 100)
    df["ema_200"] = talib.EMA(df["close"], 200)

    # MACD
    macd, macd_signal, macd_hist = talib.MACD(df["close"])
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # ADX
    df["adx"] = talib.ADX(df["high"], df["low"], df["close"])

    # 成交量
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

    # OBV
    df["obv"] = talib.OBV(df["close"], df["volume"])
    df["obv_sma_20"] = df["obv"].rolling(20).mean()
    df["obv_trend"] = df["obv"] / df["obv_sma_20"]

    # SAR
    df["sar"] = talib.SAR(df["high"], df["low"])

    # 未來報酬
    df["future_return_3"] = df["close"].shift(-3) / df["close"] - 1
    df["future_return_5"] = df["close"].shift(-5) / df["close"] - 1

    return df


def test_different_conditions(df):
    """測試不同的條件組合來找到更多樣本"""

    print("=== 測試不同條件組合 ===\n")

    # 條件1: 只看中軸附近 + 基本趨勢
    condition1_long = (
        (abs(df["distance_to_middle"]) < 0.02)  # 放寬到2%
        & (df["ema_50"] > df["ema_100"])  # 簡單多頭排列
        & (df["future_return_5"] > 0.015)  # 1.5%報酬
    )

    condition1_short = (
        (abs(df["distance_to_middle"]) < 0.02)
        & (df["ema_50"] < df["ema_100"])  # 簡單空頭排列
        & (df["future_return_5"] < -0.015)
    )

    print(f"條件1 - 中軸2% + 基本趨勢:")
    print(f"  做多: {condition1_long.sum()} 次")
    print(f"  做空: {condition1_short.sum()} 次")

    # 條件2: 放寬中軸範圍 + MACD
    condition2_long = (
        (abs(df["distance_to_middle"]) < 0.03)  # 3%範圍
        & (df["macd"] > df["macd_signal"])  # MACD多頭
        & (df["future_return_5"] > 0.015)
    )

    condition2_short = (
        (abs(df["distance_to_middle"]) < 0.03)
        & (df["macd"] < df["macd_signal"])  # MACD空頭
        & (df["future_return_5"] < -0.015)
    )

    print(f"條件2 - 中軸3% + MACD:")
    print(f"  做多: {condition2_long.sum()} 次")
    print(f"  做空: {condition2_short.sum()} 次")

    # 條件3: 只看趨勢 + 成交量
    condition3_long = (
        (df["ema_50"] > df["ema_100"])
        & (df["volume_ratio"] > 1.2)  # 成交量放大
        & (df["future_return_5"] > 0.015)
    )

    condition3_short = (
        (df["ema_50"] < df["ema_100"])
        & (df["volume_ratio"] > 1.2)
        & (df["future_return_5"] < -0.015)
    )

    print(f"條件3 - 趨勢 + 成交量:")
    print(f"  做多: {condition3_long.sum()} 次")
    print(f"  做空: {condition3_short.sum()} 次")

    # 條件4: 降低報酬要求
    condition4_long = (
        (abs(df["distance_to_middle"]) < 0.025)
        & (df["ema_50"] > df["ema_100"])
        & (df["macd"] > df["macd_signal"])
        & (df["future_return_5"] > 0.01)  # 降低到1%
    )

    condition4_short = (
        (abs(df["distance_to_middle"]) < 0.025)
        & (df["ema_50"] < df["ema_100"])
        & (df["macd"] < df["macd_signal"])
        & (df["future_return_5"] < -0.01)
    )

    print(f"條件4 - 綜合條件 + 1%報酬:")
    print(f"  做多: {condition4_long.sum()} 次")
    print(f"  做空: {condition4_short.sum()} 次")

    # 條件5: SAR + 趨勢
    condition5_long = (
        (df["close"] > df["sar"])  # SAR多頭
        & (df["ema_50"] > df["ema_100"])
        & (df["future_return_3"] > 0.01)  # 3期1%報酬
    )

    condition5_short = (
        (df["close"] < df["sar"])  # SAR空頭
        & (df["ema_50"] < df["ema_100"])
        & (df["future_return_3"] < -0.01)
    )

    print(f"條件5 - SAR + 趨勢 (3期):")
    print(f"  做多: {condition5_long.sum()} 次")
    print(f"  做空: {condition5_short.sum()} 次")

    # 條件6: 只看MACD交叉
    macd_cross_up = (df["macd"] > df["macd_signal"]) & (
        df["macd"].shift(1) <= df["macd_signal"].shift(1)
    )
    macd_cross_down = (df["macd"] < df["macd_signal"]) & (
        df["macd"].shift(1) >= df["macd_signal"].shift(1)
    )

    condition6_long = macd_cross_up & (df["future_return_5"] > 0.01)
    condition6_short = macd_cross_down & (df["future_return_5"] < -0.01)

    print(f"條件6 - MACD交叉:")
    print(f"  做多: {condition6_long.sum()} 次")
    print(f"  做空: {condition6_short.sum()} 次")

    return {
        "condition4_long": condition4_long,
        "condition4_short": condition4_short,
        "condition5_long": condition5_long,
        "condition5_short": condition5_short,
        "condition6_long": condition6_long,
        "condition6_short": condition6_short,
    }


def analyze_best_conditions(df, conditions):
    """分析最佳條件的特徵"""
    print(f"\n=== 分析最有潛力的條件 ===\n")

    # 選擇樣本數量合適的條件 (100-1000次)
    for name, condition in conditions.items():
        count = condition.sum()
        if 100 <= count <= 1000:
            print(f"{name}: {count} 次樣本")

            # 分析這個條件的特徵
            subset = df[condition]
            if len(subset) > 0:
                avg_return = (
                    subset["future_return_5"].mean()
                    if "future_return_5" in subset.columns
                    else subset["future_return_3"].mean()
                )
                print(f"  平均報酬: {avg_return*100:.2f}%")
                print(
                    f"  成功率: {(avg_return > 0 if 'long' in name else avg_return < 0)}"
                )

                if "macd" in subset.columns:
                    print(f"  平均MACD: {subset['macd'].mean():.4f}")
                if "volume_ratio" in subset.columns:
                    print(
                        f"  平均成交量比: {subset['volume_ratio'].mean():.2f}"
                    )
                print()


def create_final_features(df):
    """創建最終的特徵集合"""
    print("=== 創建最終特徵集合 ===\n")

    # 基於分析結果，選擇條件4作為基礎 (綜合條件 + 1%報酬)
    df["trend_long_v2"] = (
        (abs(df["distance_to_middle"]) < 0.025)
        & (df["ema_50"] > df["ema_100"])
        & (df["macd"] > df["macd_signal"])
        & (df["future_return_5"] > 0.01)
    ).astype(int)

    df["trend_short_v2"] = (
        (abs(df["distance_to_middle"]) < 0.025)
        & (df["ema_50"] < df["ema_100"])
        & (df["macd"] < df["macd_signal"])
        & (df["future_return_5"] < -0.01)
    ).astype(int)

    # 添加SAR條件作為補充
    df["sar_long_v2"] = (
        (df["close"] > df["sar"])
        & (df["ema_50"] > df["ema_100"])
        & (df["future_return_3"] > 0.01)
    ).astype(int)

    df["sar_short_v2"] = (
        (df["close"] < df["sar"])
        & (df["ema_50"] < df["ema_100"])
        & (df["future_return_3"] < -0.01)
    ).astype(int)

    print(f"最終特徵統計:")
    print(f"  trend_long_v2: {df['trend_long_v2'].sum()} 次")
    print(f"  trend_short_v2: {df['trend_short_v2'].sum()} 次")
    print(f"  sar_long_v2: {df['sar_long_v2'].sum()} 次")
    print(f"  sar_short_v2: {df['sar_short_v2'].sum()} 次")

    return df


def main():
    """主函數"""
    print("載入數據並準備特徵...")
    df = load_and_prepare_data()

    print("測試不同條件組合...")
    conditions = test_different_conditions(df)

    print("分析最佳條件...")
    analyze_best_conditions(df, conditions)

    print("創建最終特徵...")
    df_final = create_final_features(df)

    # 保存結果
    df_final.to_csv("training_data/btc_trend_patterns_v2.csv", index=False)
    print(f"\n數據已保存到 training_data/btc_trend_patterns_v2.csv")

    return df_final


if __name__ == "__main__":
    result = main()
