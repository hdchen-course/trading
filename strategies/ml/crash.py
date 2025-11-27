#!/usr/bin/env python3
"""
崩盤預測策略
"""

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import talib
from typing import Dict

from ..base import BaseStrategy, StrategyConfig


class CrashPredictionStrategy(BaseStrategy):
    """8小時+1小時crash預測策略"""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.crash_threshold = config.params.get("crash_threshold", 0.75)
        try:
            self.model = joblib.load("models/crash/balanced_crash_prediction_model_8h_1h.pkl")
            self.feature_names = joblib.load("models/crash/balanced_feature_names_8h_1h.pkl")
        except Exception:
            self.model = None
            self.feature_names = None

    def get_description(self) -> str:
        """返回策略描述"""
        return f"8小時crash預測策略 (閾值: {self.crash_threshold})"

    def extract_features(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """提取前8小時的特徵"""
        start_idx = max(0, idx - 32)
        segment = df.iloc[start_idx : idx + 1]

        if len(segment) < 10:
            return None

        features = {}

        # 價格趨勢特徵
        features["price_trend_8h"] = (
            segment["close"].iloc[-1] - segment["close"].iloc[0]
        ) / segment["close"].iloc[0]
        features["price_volatility_8h"] = segment["close"].pct_change().std()
        features["high_low_ratio_8h"] = (
            segment["high"].max() - segment["low"].min()
        ) / segment["close"].iloc[-1]

        # 成交量特徵
        features["volume_trend_8h"] = (
            segment["volume"].iloc[-1] - segment["volume"].iloc[0]
        ) / segment["volume"].iloc[0]
        features["volume_spike_8h"] = segment["volume"].max() / segment["volume"].mean()

        # ATR相關特徵
        current_atr = segment["atr"].iloc[-1]
        if current_atr > 0:
            features["atr_ratio_8h"] = (
                abs(segment["close"].iloc[-1] - segment["close"].iloc[0]) / current_atr
            )
        else:
            features["atr_ratio_8h"] = 0

        # RSI變化
        if len(segment) >= 14:
            rsi_values = talib.RSI(segment["close"].values, timeperiod=14)
            if not np.isnan(rsi_values[-1]) and not np.isnan(rsi_values[0]):
                features["rsi_change_8h"] = rsi_values[-1] - rsi_values[0]
            else:
                features["rsi_change_8h"] = 0
        else:
            features["rsi_change_8h"] = 0

        # MACD特徵
        if len(segment) >= 26:
            macd, signal, hist = talib.MACD(segment["close"].values)
            if not np.isnan(macd[-1]):
                features["macd_8h"] = macd[-1]
                features["macd_signal_8h"] = (
                    signal[-1] if not np.isnan(signal[-1]) else 0
                )
            else:
                features["macd_8h"] = 0
                features["macd_signal_8h"] = 0
        else:
            features["macd_8h"] = 0
            features["macd_signal_8h"] = 0

        # 布林帶特徵
        if len(segment) >= 20:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                segment["close"].values, timeperiod=20
            )
            if not np.isnan(bb_upper[-1]):
                features["bb_position_8h"] = (
                    segment["close"].iloc[-1] - bb_lower[-1]
                ) / (bb_upper[-1] - bb_lower[-1])
            else:
                features["bb_position_8h"] = 0.5
        else:
            features["bb_position_8h"] = 0.5

        # 移動平均特徵
        if len(segment) >= 20:
            ma20 = segment["close"].rolling(20).mean().iloc[-1]
            features["ma20_distance_8h"] = (
                (segment["close"].iloc[-1] - ma20) / ma20 if not np.isnan(ma20) else 0
            )
        else:
            features["ma20_distance_8h"] = 0

        return features

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """生成交易信號"""
        if self.model is None:
            return pd.Series(0, index=df.index)

        # 計算ATR
        df = df.copy()
        df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)

        signals = pd.Series(0, index=df.index)

        for i in range(32, len(df)):
            features = self.extract_features(df, i)
            if features:
                # 轉換為DataFrame並確保特徵順序正確
                feature_df = pd.DataFrame([features])
                feature_df = feature_df.reindex(
                    columns=self.feature_names, fill_value=0
                )

                # 預測crash概率
                crash_prob = self.model.predict_proba(feature_df)[0][1]

                # 高信心度crash預測時做空
                if crash_prob > self.crash_threshold:
                    signals.iloc[i] = -1

        return signals

