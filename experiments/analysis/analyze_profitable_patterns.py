#!/usr/bin/env python3
"""
分析獲利模式的具體特徵區間
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_patterns():
    # 載入特徵數據
    df = pd.read_csv("training_data/btc_features_labeled.csv")

    # 關鍵特徵列表
    key_features = [
        "high_low_ratio",
        "close_sma5_ratio",
        "volatility_5",
        "close_sma10_ratio",
        "volatility_10",
        "close_sma20_ratio",
        "volatility_20",
        "rsi_14",
        "bb_position",
    ]

    print("=== 獲利模式特徵區間分析 ===\n")

    # 分析做多信號的特徵分布
    long_signals = df[df["long_signal"] == 1]
    print(f"做多信號樣本數: {len(long_signals)}")
    print("做多信號的特徵區間:")

    for feature in key_features:
        if feature in df.columns and not long_signals[feature].isna().all():
            stats = {
                "mean": long_signals[feature].mean(),
                "std": long_signals[feature].std(),
                "min": long_signals[feature].quantile(0.1),
                "max": long_signals[feature].quantile(0.9),
                "median": long_signals[feature].median(),
            }
            print(f"  {feature}:")
            print(f"    範圍: {stats['min']:.6f} ~ {stats['max']:.6f}")
            print(f"    平均: {stats['mean']:.6f} ± {stats['std']:.6f}")
            print(f"    中位數: {stats['median']:.6f}")

    print("\n" + "=" * 50 + "\n")

    # 分析做空信號的特徵分布
    short_signals = df[df["short_signal"] == 1]
    print(f"做空信號樣本數: {len(short_signals)}")
    print("做空信號的特徵區間:")

    for feature in key_features:
        if feature in df.columns and not short_signals[feature].isna().all():
            stats = {
                "mean": short_signals[feature].mean(),
                "std": short_signals[feature].std(),
                "min": short_signals[feature].quantile(0.1),
                "max": short_signals[feature].quantile(0.9),
                "median": short_signals[feature].median(),
            }
            print(f"  {feature}:")
            print(f"    範圍: {stats['min']:.6f} ~ {stats['max']:.6f}")
            print(f"    平均: {stats['mean']:.6f} ± {stats['std']:.6f}")
            print(f"    中位數: {stats['median']:.6f}")

    print("\n" + "=" * 50 + "\n")

    # 對比分析
    print("做多 vs 做空 關鍵差異:")
    for feature in key_features[:5]:  # 只看前5個最重要的
        if feature in df.columns:
            long_mean = long_signals[feature].mean()
            short_mean = short_signals[feature].mean()
            overall_mean = df[feature].mean()

            print(f"  {feature}:")
            print(f"    整體平均: {overall_mean:.6f}")
            print(
                f"    做多平均: {long_mean:.6f} ({(long_mean/overall_mean-1)*100:+.2f}%)"
            )
            print(
                f"    做空平均: {short_mean:.6f} ({(short_mean/overall_mean-1)*100:+.2f}%)"
            )
            print(f"    差異: {abs(long_mean-short_mean):.6f}")

    return df, long_signals, short_signals


def create_trading_rules(long_signals, short_signals):
    """根據分析結果創建交易規則"""

    key_features = [
        "high_low_ratio",
        "close_sma5_ratio",
        "volatility_5",
        "close_sma10_ratio",
        "volatility_10",
    ]

    print("\n=== 基於分析的交易規則 ===\n")

    # 做多規則
    print("做多信號條件 (所有條件需同時滿足):")
    long_rules = {}
    for feature in key_features:
        if not long_signals[feature].isna().all():
            min_val = long_signals[feature].quantile(0.2)  # 20%分位數
            max_val = long_signals[feature].quantile(0.8)  # 80%分位數
            long_rules[feature] = (min_val, max_val)
            print(f"  {feature}: {min_val:.6f} <= x <= {max_val:.6f}")

    print("\n做空信號條件 (所有條件需同時滿足):")
    short_rules = {}
    for feature in key_features:
        if not short_signals[feature].isna().all():
            min_val = short_signals[feature].quantile(0.2)
            max_val = short_signals[feature].quantile(0.8)
            short_rules[feature] = (min_val, max_val)
            print(f"  {feature}: {min_val:.6f} <= x <= {max_val:.6f}")

    return long_rules, short_rules


if __name__ == "__main__":
    df, long_signals, short_signals = analyze_patterns()
    long_rules, short_rules = create_trading_rules(long_signals, short_signals)
