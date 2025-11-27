#!/usr/bin/env python3
"""
新的特徵工程方法 - 專注於找出有效的特徵區間
"""

import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureEngineerV2:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None

    def load_data(self, filepath):
        """載入數據"""
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def create_advanced_features(self, df):
        """創建進階技術指標特徵"""
        data = df.copy()

        # 基本價格特徵
        data["price_change"] = data["close"].pct_change()
        data["high_low_ratio"] = data["high"] / data["low"]
        data["close_open_ratio"] = data["close"] / data["open"]

        # 波動率特徵
        data["volatility_5"] = data["price_change"].rolling(5).std()
        data["volatility_10"] = data["price_change"].rolling(10).std()
        data["volatility_20"] = data["price_change"].rolling(20).std()

        # 趨勢特徵
        data["sma_5"] = talib.SMA(data["close"], 5)
        data["sma_10"] = talib.SMA(data["close"], 10)
        data["sma_20"] = talib.SMA(data["close"], 20)
        data["sma_50"] = talib.SMA(data["close"], 50)

        # 價格相對於移動平均線的位置
        data["close_sma5_ratio"] = data["close"] / data["sma_5"]
        data["close_sma10_ratio"] = data["close"] / data["sma_10"]
        data["close_sma20_ratio"] = data["close"] / data["sma_20"]

        # RSI 相關
        data["rsi_14"] = talib.RSI(data["close"], 14)
        data["rsi_7"] = talib.RSI(data["close"], 7)
        data["rsi_21"] = talib.RSI(data["close"], 21)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(data["close"])
        data["macd"] = macd
        data["macd_signal"] = macd_signal
        data["macd_hist"] = macd_hist

        # 布林帶
        bb_upper, bb_middle, bb_lower = talib.BBANDS(data["close"])
        data["bb_upper"] = bb_upper
        data["bb_middle"] = bb_middle
        data["bb_lower"] = bb_lower
        data["bb_width"] = (bb_upper - bb_lower) / bb_middle
        data["bb_position"] = (data["close"] - bb_lower) / (
            bb_upper - bb_lower
        )

        # 成交量特徵
        data["volume_sma_10"] = data["volume"].rolling(10).mean()
        data["volume_ratio"] = data["volume"] / data["volume_sma_10"]

        # 動量指標
        data["momentum_5"] = data["close"] / data["close"].shift(5) - 1
        data["momentum_10"] = data["close"] / data["close"].shift(10) - 1

        # 支撐阻力特徵
        data["high_max_10"] = data["high"].rolling(10).max()
        data["low_min_10"] = data["low"].rolling(10).min()
        data["price_position"] = (data["close"] - data["low_min_10"]) / (
            data["high_max_10"] - data["low_min_10"]
        )

        return data

    def create_target_labels(
        self, df, future_periods=4, profit_threshold=0.015
    ):
        """創建目標標籤 - 未來N個週期內是否有足夠獲利機會"""
        data = df.copy()

        # 計算未來最高價和最低價
        data["future_high"] = (
            data["high"].shift(-1).rolling(future_periods).max()
        )
        data["future_low"] = (
            data["low"].shift(-1).rolling(future_periods).min()
        )

        # 計算潛在獲利
        data["long_profit"] = (data["future_high"] - data["close"]) / data[
            "close"
        ]
        data["short_profit"] = (data["close"] - data["future_low"]) / data[
            "close"
        ]

        # 創建標籤
        data["long_signal"] = (data["long_profit"] > profit_threshold).astype(
            int
        )
        data["short_signal"] = (
            data["short_profit"] > profit_threshold
        ).astype(int)
        data["no_signal"] = (
            (data["long_profit"] <= profit_threshold)
            & (data["short_profit"] <= profit_threshold)
        ).astype(int)

        return data

    def find_profitable_patterns(self, df, min_samples=50):
        """找出獲利模式的特徵區間"""
        feature_cols = [
            col
            for col in df.columns
            if col
            not in [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "future_high",
                "future_low",
                "long_profit",
                "short_profit",
                "long_signal",
                "short_signal",
                "no_signal",
            ]
        ]

        profitable_patterns = {}

        # 分析做多信號的特徵分布
        long_data = df[df["long_signal"] == 1]
        if len(long_data) >= min_samples:
            profitable_patterns["long"] = {}
            for feature in feature_cols:
                if not long_data[feature].isna().all():
                    profitable_patterns["long"][feature] = {
                        "mean": long_data[feature].mean(),
                        "std": long_data[feature].std(),
                        "min": long_data[feature].quantile(0.1),
                        "max": long_data[feature].quantile(0.9),
                        "median": long_data[feature].median(),
                    }

        # 分析做空信號的特徵分布
        short_data = df[df["short_signal"] == 1]
        if len(short_data) >= min_samples:
            profitable_patterns["short"] = {}
            for feature in feature_cols:
                if not short_data[feature].isna().all():
                    profitable_patterns["short"][feature] = {
                        "mean": short_data[feature].mean(),
                        "std": short_data[feature].std(),
                        "min": short_data[feature].quantile(0.1),
                        "max": short_data[feature].quantile(0.9),
                        "median": short_data[feature].median(),
                    }

        return profitable_patterns

    def select_best_features(self, X, y, k=20):
        """選擇最佳特徵"""
        # 移除 NaN 值
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) == 0:
            return []

        # 特徵選擇
        selector = SelectKBest(
            score_func=f_regression, k=min(k, X_clean.shape[1])
        )
        selector.fit(X_clean, y_clean)

        # 獲取選中的特徵
        selected_features = X.columns[selector.get_support()].tolist()
        feature_scores = selector.scores_[selector.get_support()]

        return list(zip(selected_features, feature_scores))

    def analyze_feature_importance(self, df):
        """分析特徵重要性"""
        feature_cols = [
            col
            for col in df.columns
            if col
            not in [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "future_high",
                "future_low",
                "long_profit",
                "short_profit",
                "long_signal",
                "short_signal",
                "no_signal",
            ]
        ]

        X = df[feature_cols]

        # 分析做多信號的特徵重要性
        long_features = self.select_best_features(X, df["long_signal"])

        # 分析做空信號的特徵重要性
        short_features = self.select_best_features(X, df["short_signal"])

        return {
            "long_features": long_features,
            "short_features": short_features,
        }

    def create_training_dataset(self, df, patterns, signal_type="long"):
        """根據獲利模式創建訓練數據集"""
        if signal_type not in patterns:
            return None, None

        pattern = patterns[signal_type]

        # 根據獲利模式篩選數據
        mask = pd.Series([True] * len(df))

        for feature, stats in pattern.items():
            if feature in df.columns:
                # 使用統計範圍篩選
                feature_mask = (
                    (df[feature] >= stats["min"])
                    & (df[feature] <= stats["max"])
                    & (~df[feature].isna())
                )
                mask = mask & feature_mask

        filtered_df = df[mask].copy()

        if len(filtered_df) < 100:  # 確保有足夠的數據
            return None, None

        # 準備特徵和標籤
        feature_cols = list(pattern.keys())
        X = filtered_df[feature_cols]
        y = filtered_df[f"{signal_type}_signal"]

        return X, y


def main():
    """主函數 - 執行特徵工程分析"""
    engineer = FeatureEngineerV2()

    # 載入數據
    print("載入數據...")
    df = engineer.load_data("training_data/btc_12months_15m.csv")
    print(f"數據形狀: {df.shape}")

    # 創建特徵
    print("創建技術指標特徵...")
    df_features = engineer.create_advanced_features(df)

    # 創建目標標籤
    print("創建目標標籤...")
    df_labeled = engineer.create_target_labels(df_features)

    # 分析信號分布
    long_signals = df_labeled["long_signal"].sum()
    short_signals = df_labeled["short_signal"].sum()
    no_signals = df_labeled["no_signal"].sum()

    print(f"\n信號分布:")
    print(
        f"做多信號: {long_signals} ({long_signals/len(df_labeled)*100:.2f}%)"
    )
    print(
        f"做空信號: {short_signals} ({short_signals/len(df_labeled)*100:.2f}%)"
    )
    print(f"無信號: {no_signals} ({no_signals/len(df_labeled)*100:.2f}%)")

    # 找出獲利模式
    print("\n分析獲利模式...")
    patterns = engineer.find_profitable_patterns(df_labeled)

    # 分析特徵重要性
    print("分析特徵重要性...")
    feature_importance = engineer.analyze_feature_importance(df_labeled)

    print("\n做多信號最重要特徵:")
    for feature, score in feature_importance["long_features"][:10]:
        print(f"  {feature}: {score:.4f}")

    print("\n做空信號最重要特徵:")
    for feature, score in feature_importance["short_features"][:10]:
        print(f"  {feature}: {score:.4f}")

    # 保存結果
    df_labeled.to_csv("training_data/btc_features_labeled.csv", index=False)
    print("\n特徵數據已保存到 training_data/btc_features_labeled.csv")

    return df_labeled, patterns, feature_importance


if __name__ == "__main__":
    df_labeled, patterns, feature_importance = main()
