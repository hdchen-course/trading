#!/usr/bin/env python3
"""
比較RSI vs StochRSI在布林帶策略中的效果
"""

import pandas as pd
import numpy as np
import talib


def add_stochrsi_features(df):
    """添加StochRSI指標"""
    data = df.copy()

    # StochRSI (14, 14, 3, 3) - 標準參數
    fastk, fastd = talib.STOCHRSI(
        data["close"], timeperiod=14, fastk_period=3, fastd_period=3
    )
    data["stochrsi_k"] = fastk
    data["stochrsi_d"] = fastd

    # StochRSI (48, 48, 3, 3) - 與布林帶同週期
    fastk_48, fastd_48 = talib.STOCHRSI(
        data["close"], timeperiod=48, fastk_period=3, fastd_period=3
    )
    data["stochrsi_k_48"] = fastk_48
    data["stochrsi_d_48"] = fastd_48

    return data


def create_stochrsi_signals(df, profit_threshold=0.015):
    """創建基於StochRSI的信號"""
    data = df.copy()

    # StochRSI做多信號 (超賣 + 布林帶下軌)
    data["stochrsi_long_signal"] = (
        (data["bb_position"] < 0.2)  # 布林帶下軌附近
        & (data["stochrsi_k"] < 20)  # StochRSI超賣
        & (data["long_profit"] > profit_threshold)
    ).astype(int)

    # StochRSI做空信號 (超買 + 布林帶上軌)
    data["stochrsi_short_signal"] = (
        (data["bb_position"] > 0.8)  # 布林帶上軌附近
        & (data["stochrsi_k"] > 80)  # StochRSI超買
        & (data["short_profit"] > profit_threshold)
    ).astype(int)

    # 48週期StochRSI信號
    data["stochrsi48_long_signal"] = (
        (data["bb_position"] < 0.2)
        & (data["stochrsi_k_48"] < 20)
        & (data["long_profit"] > profit_threshold)
    ).astype(int)

    data["stochrsi48_short_signal"] = (
        (data["bb_position"] > 0.8)
        & (data["stochrsi_k_48"] > 80)
        & (data["short_profit"] > profit_threshold)
    ).astype(int)

    return data


def compare_indicators(df):
    """比較RSI vs StochRSI效果"""
    print("=== RSI vs StochRSI 比較分析 ===\n")

    # 原始RSI信號
    rsi_long = df["bb_long_signal"].sum()
    rsi_short = df["bb_short_signal"].sum()

    # StochRSI信號
    stochrsi_long = df["stochrsi_long_signal"].sum()
    stochrsi_short = df["stochrsi_short_signal"].sum()

    # 48週期StochRSI信號
    stochrsi48_long = df["stochrsi48_long_signal"].sum()
    stochrsi48_short = df["stochrsi48_short_signal"].sum()

    total_samples = len(df.dropna())

    print("信號數量比較:")
    print(f"  RSI(14) + 布林帶:")
    print(f"    做多: {rsi_long} ({rsi_long/total_samples*100:.2f}%)")
    print(f"    做空: {rsi_short} ({rsi_short/total_samples*100:.2f}%)")

    print(f"  StochRSI(14) + 布林帶:")
    print(
        f"    做多: {stochrsi_long} ({stochrsi_long/total_samples*100:.2f}%)"
    )
    print(
        f"    做空: {stochrsi_short} ({stochrsi_short/total_samples*100:.2f}%)"
    )

    print(f"  StochRSI(48) + 布林帶:")
    print(
        f"    做多: {stochrsi48_long} ({stochrsi48_long/total_samples*100:.2f}%)"
    )
    print(
        f"    做空: {stochrsi48_short} ({stochrsi48_short/total_samples*100:.2f}%)"
    )

    # 分析信號質量
    if stochrsi_long > 0:
        stochrsi_long_data = df[df["stochrsi_long_signal"] == 1]
        avg_profit = stochrsi_long_data["long_profit"].mean()
        print(
            f"\nStochRSI做多信號平均獲利: {avg_profit:.4f} ({avg_profit*100:.2f}%)"
        )
        print(
            f"StochRSI值範圍: {stochrsi_long_data['stochrsi_k'].min():.1f} - {stochrsi_long_data['stochrsi_k'].max():.1f}"
        )

    if stochrsi_short > 0:
        stochrsi_short_data = df[df["stochrsi_short_signal"] == 1]
        avg_profit = stochrsi_short_data["short_profit"].mean()
        print(
            f"StochRSI做空信號平均獲利: {avg_profit:.4f} ({avg_profit*100:.2f}%)"
        )
        print(
            f"StochRSI值範圍: {stochrsi_short_data['stochrsi_k'].min():.1f} - {stochrsi_short_data['stochrsi_k'].max():.1f}"
        )

    # RSI vs StochRSI極值比較
    print(f"\n極值比較:")
    rsi_extreme_low = (df["rsi_14"] < 30).sum()
    rsi_extreme_high = (df["rsi_14"] > 70).sum()
    stochrsi_extreme_low = (df["stochrsi_k"] < 20).sum()
    stochrsi_extreme_high = (df["stochrsi_k"] > 80).sum()

    print(
        f"  RSI < 30: {rsi_extreme_low} ({rsi_extreme_low/total_samples*100:.2f}%)"
    )
    print(
        f"  RSI > 70: {rsi_extreme_high} ({rsi_extreme_high/total_samples*100:.2f}%)"
    )
    print(
        f"  StochRSI < 20: {stochrsi_extreme_low} ({stochrsi_extreme_low/total_samples*100:.2f}%)"
    )
    print(
        f"  StochRSI > 80: {stochrsi_extreme_high} ({stochrsi_extreme_high/total_samples*100:.2f}%)"
    )


def main():
    # 載入之前的數據
    df = pd.read_csv("training_data/btc_binance_features.csv")

    # 添加StochRSI
    df_with_stochrsi = add_stochrsi_features(df)

    # 創建StochRSI信號
    df_signals = create_stochrsi_signals(df_with_stochrsi)

    # 比較分析
    compare_indicators(df_signals)

    # 保存結果
    df_signals.to_csv(
        "training_data/btc_rsi_stochrsi_comparison.csv", index=False
    )
    print(f"\n比較結果已保存到 training_data/btc_rsi_stochrsi_comparison.csv")

    return df_signals


if __name__ == "__main__":
    result = main()
