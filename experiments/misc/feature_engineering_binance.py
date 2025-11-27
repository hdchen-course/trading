#!/usr/bin/env python3
"""
使用幣安app相同參數的特徵工程 - 布林帶(48, 2.4)
"""

import pandas as pd
import numpy as np
import talib


class BinanceFeatureEngineer:
    def __init__(self):
        self.bb_period = 48
        self.bb_std = 2.4

    def load_data(self, filepath):
        """載入數據"""
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def create_binance_features(self, df):
        """創建與幣安app一致的技術指標"""
        data = df.copy()

        # 基本價格特徵
        data["price_change"] = data["close"].pct_change()
        data["high_low_ratio"] = data["high"] / data["low"]

        # 幣安布林帶參數 (48, 2.4)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            data["close"],
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
        )
        data["bb_upper"] = bb_upper
        data["bb_middle"] = bb_middle  # 48期SMA
        data["bb_lower"] = bb_lower
        data["bb_width"] = (bb_upper - bb_lower) / bb_middle
        data["bb_position"] = (data["close"] - bb_lower) / (
            bb_upper - bb_lower
        )

        # 價格相對於布林帶中軌的位置
        data["close_bb_middle_ratio"] = data["close"] / data["bb_middle"]

        # 布林帶突破信號
        data["bb_upper_break"] = data["close"] > data["bb_upper"]
        data["bb_lower_break"] = data["close"] < data["bb_lower"]

        # 短期移動平均線
        data["sma_5"] = talib.SMA(data["close"], 5)
        data["sma_10"] = talib.SMA(data["close"], 10)
        data["sma_20"] = talib.SMA(data["close"], 20)

        # 價格相對位置
        data["close_sma5_ratio"] = data["close"] / data["sma_5"]
        data["close_sma10_ratio"] = data["close"] / data["sma_10"]
        data["close_sma20_ratio"] = data["close"] / data["sma_20"]

        # 波動率 (使用與布林帶相同的週期)
        data["volatility_48"] = data["price_change"].rolling(48).std()
        data["volatility_24"] = data["price_change"].rolling(24).std()
        data["volatility_12"] = data["price_change"].rolling(12).std()

        # RSI
        data["rsi_14"] = talib.RSI(data["close"], 14)
        data["rsi_48"] = talib.RSI(data["close"], 48)  # 與布林帶同週期

        # 成交量
        data["volume_sma_48"] = data["volume"].rolling(48).mean()
        data["volume_ratio"] = data["volume"] / data["volume_sma_48"]

        return data

    def create_target_labels(
        self, df, future_periods=4, profit_threshold=0.015
    ):
        """創建目標標籤"""
        data = df.copy()

        # 計算未來獲利潛力
        data["future_high"] = (
            data["high"].shift(-1).rolling(future_periods).max()
        )
        data["future_low"] = (
            data["low"].shift(-1).rolling(future_periods).min()
        )

        data["long_profit"] = (data["future_high"] - data["close"]) / data[
            "close"
        ]
        data["short_profit"] = (data["close"] - data["future_low"]) / data[
            "close"
        ]

        # 布林帶策略標籤
        data["bb_long_signal"] = (
            (data["bb_position"] < 0.2)  # 接近下軌
            & (data["long_profit"] > profit_threshold)
        ).astype(int)

        data["bb_short_signal"] = (
            (data["bb_position"] > 0.8)  # 接近上軌
            & (data["short_profit"] > profit_threshold)
        ).astype(int)

        return data

    def analyze_bb_patterns(self, df):
        """分析布林帶模式"""
        print("=== 幣安布林帶(48, 2.4)模式分析 ===\n")

        # 基本統計
        total_samples = len(df.dropna())
        bb_long_signals = df["bb_long_signal"].sum()
        bb_short_signals = df["bb_short_signal"].sum()

        print(f"總樣本數: {total_samples}")
        print(
            f"布林帶做多信號: {bb_long_signals} ({bb_long_signals/total_samples*100:.2f}%)"
        )
        print(
            f"布林帶做空信號: {bb_short_signals} ({bb_short_signals/total_samples*100:.2f}%)"
        )

        # 布林帶位置分布
        bb_positions = df["bb_position"].dropna()
        print(f"\n布林帶位置分布:")
        print(
            f"  下軌突破 (<0): {(bb_positions < 0).sum()} ({(bb_positions < 0).mean()*100:.2f}%)"
        )
        print(
            f"  下軌附近 (0-0.2): {((bb_positions >= 0) & (bb_positions <= 0.2)).sum()} ({((bb_positions >= 0) & (bb_positions <= 0.2)).mean()*100:.2f}%)"
        )
        print(
            f"  中軌附近 (0.4-0.6): {((bb_positions >= 0.4) & (bb_positions <= 0.6)).sum()} ({((bb_positions >= 0.4) & (bb_positions <= 0.6)).mean()*100:.2f}%)"
        )
        print(
            f"  上軌附近 (0.8-1.0): {((bb_positions >= 0.8) & (bb_positions <= 1.0)).sum()} ({((bb_positions >= 0.8) & (bb_positions <= 1.0)).mean()*100:.2f}%)"
        )
        print(
            f"  上軌突破 (>1): {(bb_positions > 1).sum()} ({(bb_positions > 1).mean()*100:.2f}%)"
        )

        # 分析做多信號特徵
        if bb_long_signals > 0:
            long_data = df[df["bb_long_signal"] == 1]
            print(f"\n做多信號特徵 (樣本數: {len(long_data)}):")
            print(
                f"  布林帶位置: {long_data['bb_position'].mean():.3f} ± {long_data['bb_position'].std():.3f}"
            )
            print(
                f"  RSI_14: {long_data['rsi_14'].mean():.1f} ± {long_data['rsi_14'].std():.1f}"
            )
            print(
                f"  波動率_48: {long_data['volatility_48'].mean():.6f} ± {long_data['volatility_48'].std():.6f}"
            )
            print(
                f"  相對中軌: {long_data['close_bb_middle_ratio'].mean():.6f} ± {long_data['close_bb_middle_ratio'].std():.6f}"
            )

        # 分析做空信號特徵
        if bb_short_signals > 0:
            short_data = df[df["bb_short_signal"] == 1]
            print(f"\n做空信號特徵 (樣本數: {len(short_data)}):")
            print(
                f"  布林帶位置: {short_data['bb_position'].mean():.3f} ± {short_data['bb_position'].std():.3f}"
            )
            print(
                f"  RSI_14: {short_data['rsi_14'].mean():.1f} ± {short_data['rsi_14'].std():.1f}"
            )
            print(
                f"  波動率_48: {short_data['volatility_48'].mean():.6f} ± {short_data['volatility_48'].std():.6f}"
            )
            print(
                f"  相對中軌: {short_data['close_bb_middle_ratio'].mean():.6f} ± {short_data['close_bb_middle_ratio'].std():.6f}"
            )

        return df


def main():
    engineer = BinanceFeatureEngineer()

    # 載入數據
    print("載入BTC 15分鐘數據...")
    df = engineer.load_data("training_data/btc_12months_15m.csv")
    print(f"數據形狀: {df.shape}")

    # 創建幣安特徵
    print("創建幣安app相同參數的技術指標...")
    df_features = engineer.create_binance_features(df)

    # 創建目標標籤
    print("創建布林帶策略標籤...")
    df_labeled = engineer.create_target_labels(df_features)

    # 分析模式
    df_analyzed = engineer.analyze_bb_patterns(df_labeled)

    # 保存結果
    df_analyzed.to_csv("training_data/btc_binance_features.csv", index=False)
    print(f"\n數據已保存到 training_data/btc_binance_features.csv")

    return df_analyzed


if __name__ == "__main__":
    df_result = main()
