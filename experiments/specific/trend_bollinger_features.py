#!/usr/bin/env python3
"""
趨勢中布林帶中軸反彈模式特徵工程
分析價格在趨勢中碰到布林帶中軸的反彈模式
"""

import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression


class TrendBollingerFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self, filepath):
        """載入數據"""
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def create_trend_features(self, df):
        """創建趨勢相關特徵"""
        data = df.copy()

        # 基本價格特徵
        data["price_change"] = data["close"].pct_change()

        # 布林帶 (48, 2.4) - 與現有策略一致
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            data["close"], timeperiod=48, nbdevup=2.4, nbdevdn=2.4
        )
        data["bb_upper"] = bb_upper
        data["bb_middle"] = bb_middle
        data["bb_lower"] = bb_lower
        data["bb_position"] = (data["close"] - bb_lower) / (
            bb_upper - bb_lower
        )

        # 價格相對於中軸的位置
        data["close_bb_middle_ratio"] = data["close"] / data["bb_middle"]
        data["distance_to_middle"] = (
            data["close"] - data["bb_middle"]
        ) / data["bb_middle"]

        # EMA趨勢指標
        data["ema_50"] = talib.EMA(data["close"], 50)
        data["ema_100"] = talib.EMA(data["close"], 100)
        data["ema_200"] = talib.EMA(data["close"], 200)

        # EMA相對位置
        data["close_ema50_ratio"] = data["close"] / data["ema_50"]
        data["close_ema100_ratio"] = data["close"] / data["ema_100"]
        data["close_ema200_ratio"] = data["close"] / data["ema_200"]

        # EMA趨勢方向
        data["ema50_slope"] = data["ema_50"].pct_change(5)  # 5期斜率
        data["ema100_slope"] = data["ema_100"].pct_change(5)
        data["ema200_slope"] = data["ema_200"].pct_change(5)

        # EMA排列 (多頭/空頭排列)
        data["ema_bullish"] = (data["ema_50"] > data["ema_100"]) & (
            data["ema_100"] > data["ema_200"]
        )
        data["ema_bearish"] = (data["ema_50"] < data["ema_100"]) & (
            data["ema_100"] < data["ema_200"]
        )

        # MACD指標
        macd, macd_signal, macd_hist = talib.MACD(data["close"])
        data["macd"] = macd
        data["macd_signal"] = macd_signal
        data["macd_hist"] = macd_hist

        # MACD交叉信號
        data["macd_cross_up"] = (data["macd"] > data["macd_signal"]) & (
            data["macd"].shift(1) <= data["macd_signal"].shift(1)
        )
        data["macd_cross_down"] = (data["macd"] < data["macd_signal"]) & (
            data["macd"].shift(1) >= data["macd_signal"].shift(1)
        )

        # MACD零軸位置
        data["macd_above_zero"] = data["macd"] > 0
        data["macd_signal_above_zero"] = data["macd_signal"] > 0

        # 成交量指標
        data["volume_sma_20"] = data["volume"].rolling(20).mean()
        data["volume_ratio"] = data["volume"] / data["volume_sma_20"]

        # OBV (On Balance Volume)
        data["obv"] = talib.OBV(data["close"], data["volume"])
        data["obv_sma_20"] = data["obv"].rolling(20).mean()
        data["obv_trend"] = data["obv"] / data["obv_sma_20"]

        # SAR (Parabolic SAR)
        data["sar"] = talib.SAR(data["high"], data["low"])
        data["sar_bullish"] = (
            data["close"] > data["sar"]
        )  # 價格在SAR上方為多頭
        data["sar_bearish"] = (
            data["close"] < data["sar"]
        )  # 價格在SAR下方為空頭

        # ADX (趨勢強度)
        data["adx"] = talib.ADX(data["high"], data["low"], data["close"])
        data["plus_di"] = talib.PLUS_DI(
            data["high"], data["low"], data["close"]
        )
        data["minus_di"] = talib.MINUS_DI(
            data["high"], data["low"], data["close"]
        )

        return data

    def detect_middle_band_bounces(self, df):
        """檢測布林帶中軸反彈模式"""
        data = df.copy()

        # 檢測價格接近中軸的情況 (±1%範圍內)
        middle_touch_threshold = 0.01
        data["near_middle"] = (
            abs(data["distance_to_middle"]) < middle_touch_threshold
        )

        # 檢測反彈 - 價格從中軸附近反彈
        data["bounce_up"] = (
            data["near_middle"]
            & (data["close"].shift(-1) > data["close"])  # 下一根K線上漲
            & (data["close"].shift(-2) > data["close"].shift(-1))  # 持續上漲
        )

        data["bounce_down"] = (
            data["near_middle"]
            & (data["close"].shift(-1) < data["close"])  # 下一根K線下跌
            & (data["close"].shift(-2) < data["close"].shift(-1))  # 持續下跌
        )

        # 計算反彈後的價格變化 (未來5根K線)
        data["future_return_5"] = data["close"].shift(-5) / data["close"] - 1
        data["future_return_10"] = data["close"].shift(-10) / data["close"] - 1

        return data

    def create_trend_labels(self, df, min_return=0.02):
        """創建趨勢交易標籤"""
        data = df.copy()

        # 趨勢做多信號：
        # 1. 多頭趨勢中 (EMA排列 + ADX > 25)
        # 2. 價格回調至中軸附近
        # 3. 有反彈跡象
        # 4. MACD支持
        data["trend_long_signal"] = (
            data["ema_bullish"]  # 多頭EMA排列
            & (data["adx"] > 25)  # 趨勢強度足夠
            & data["near_middle"]  # 接近中軸
            & (data["macd"] > data["macd_signal"])  # MACD多頭
            & (data["sar_bullish"])  # SAR多頭
            & (data["obv_trend"] > 1.0)  # OBV趨勢向上
            & (data["future_return_5"] > min_return)  # 未來確實上漲
        ).astype(int)

        # 趨勢做空信號：
        # 1. 空頭趨勢中
        # 2. 價格反彈至中軸附近
        # 3. 有回落跡象
        # 4. MACD支持
        data["trend_short_signal"] = (
            data["ema_bearish"]  # 空頭EMA排列
            & (data["adx"] > 25)  # 趨勢強度足夠
            & data["near_middle"]  # 接近中軸
            & (data["macd"] < data["macd_signal"])  # MACD空頭
            & (data["sar_bearish"])  # SAR空頭
            & (data["obv_trend"] < 1.0)  # OBV趨勢向下
            & (data["future_return_5"] < -min_return)  # 未來確實下跌
        ).astype(int)

        return data

    def analyze_trend_patterns(self, df):
        """分析趨勢模式"""
        print("=== 趨勢中布林帶中軸反彈模式分析 ===\n")

        total_samples = len(df.dropna())
        trend_long_signals = df["trend_long_signal"].sum()
        trend_short_signals = df["trend_short_signal"].sum()

        print(f"總樣本數: {total_samples}")
        print(
            f"趨勢做多信號: {trend_long_signals} ({trend_long_signals/total_samples*100:.2f}%)"
        )
        print(
            f"趨勢做空信號: {trend_short_signals} ({trend_short_signals/total_samples*100:.2f}%)"
        )

        # 分析信號特徵
        if trend_long_signals > 0:
            long_data = df[df["trend_long_signal"] == 1]
            print(f"\n趨勢做多信號特徵分析:")
            print(f"  平均ADX: {long_data['adx'].mean():.1f}")
            print(f"  平均MACD: {long_data['macd'].mean():.6f}")
            print(f"  平均成交量比: {long_data['volume_ratio'].mean():.2f}")
            print(
                f"  平均未來5期報酬: {long_data['future_return_5'].mean()*100:.2f}%"
            )
            print(f"  EMA50斜率: {long_data['ema50_slope'].mean()*100:.3f}%")

        if trend_short_signals > 0:
            short_data = df[df["trend_short_signal"] == 1]
            print(f"\n趨勢做空信號特徵分析:")
            print(f"  平均ADX: {short_data['adx'].mean():.1f}")
            print(f"  平均MACD: {short_data['macd'].mean():.6f}")
            print(f"  平均成交量比: {short_data['volume_ratio'].mean():.2f}")
            print(
                f"  平均未來5期報酬: {short_data['future_return_5'].mean()*100:.2f}%"
            )
            print(f"  EMA50斜率: {short_data['ema50_slope'].mean()*100:.3f}%")

        return df

    def select_best_trend_features(self, df):
        """選擇最佳趨勢特徵"""
        feature_cols = [
            "close_bb_middle_ratio",
            "distance_to_middle",
            "close_ema50_ratio",
            "close_ema100_ratio",
            "close_ema200_ratio",
            "ema50_slope",
            "ema100_slope",
            "ema200_slope",
            "ema_bullish",
            "ema_bearish",
            "macd",
            "macd_signal",
            "macd_hist",
            "macd_cross_up",
            "macd_cross_down",
            "macd_above_zero",
            "volume_ratio",
            "obv_trend",
            "sar_bullish",
            "sar_bearish",
            "adx",
            "plus_di",
            "minus_di",
        ]

        # 移除缺失值
        df_clean = df.dropna(
            subset=feature_cols + ["trend_long_signal", "trend_short_signal"]
        )

        if len(df_clean) == 0:
            return {}, {}

        X = df_clean[feature_cols]

        # 分析做多信號的特徵重要性
        selector_long = SelectKBest(
            score_func=f_regression, k=min(15, len(feature_cols))
        )
        selector_long.fit(X, df_clean["trend_long_signal"])
        long_features = list(
            zip(
                X.columns[selector_long.get_support()],
                selector_long.scores_[selector_long.get_support()],
            )
        )
        long_features.sort(key=lambda x: x[1], reverse=True)

        # 分析做空信號的特徵重要性
        selector_short = SelectKBest(
            score_func=f_regression, k=min(15, len(feature_cols))
        )
        selector_short.fit(X, df_clean["trend_short_signal"])
        short_features = list(
            zip(
                X.columns[selector_short.get_support()],
                selector_short.scores_[selector_short.get_support()],
            )
        )
        short_features.sort(key=lambda x: x[1], reverse=True)

        print(f"\n趨勢做多最重要特徵 (前10名):")
        for feature, score in long_features[:10]:
            print(f"  {feature}: {score:.4f}")

        print(f"\n趨勢做空最重要特徵 (前10名):")
        for feature, score in short_features[:10]:
            print(f"  {feature}: {score:.4f}")

        return dict(long_features), dict(short_features)


def main():
    """主函數"""
    engineer = TrendBollingerFeatureEngineer()

    # 載入數據
    print("載入BTC 15分鐘數據...")
    df = engineer.load_data("training_data/btc_12months_15m.csv")
    print(f"數據形狀: {df.shape}")

    # 創建趨勢特徵
    print("創建趨勢相關特徵...")
    df_features = engineer.create_trend_features(df)

    # 檢測中軸反彈
    print("檢測布林帶中軸反彈模式...")
    df_bounces = engineer.detect_middle_band_bounces(df_features)

    # 創建趨勢標籤
    print("創建趨勢交易標籤...")
    df_labeled = engineer.create_trend_labels(df_bounces)

    # 分析模式
    df_analyzed = engineer.analyze_trend_patterns(df_labeled)

    # 特徵選擇
    print("分析特徵重要性...")
    long_features, short_features = engineer.select_best_trend_features(
        df_analyzed
    )

    # 保存結果
    df_analyzed.to_csv(
        "training_data/btc_trend_bollinger_features.csv", index=False
    )
    print(f"\n特徵數據已保存到 training_data/btc_trend_bollinger_features.csv")

    return df_analyzed, long_features, short_features


if __name__ == "__main__":
    df_result, long_features, short_features = main()
