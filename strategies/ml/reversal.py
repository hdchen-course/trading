#!/usr/bin/env python3
"""
反轉預測策略
"""

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import talib
from typing import Dict

from ..base import BaseStrategy, StrategyConfig


class ReversalStrategy(BaseStrategy):
    """基於A轉/V轉模式的反轉交易策略"""

    def __init__(self, name="ReversalStrategy"):
        super().__init__(name)
        self.rf_model = None
        self.feature_names = [
            "stochrsi_k",
            "stochrsi_d",
            "macd_hist",
            "volume_ratio",
            "bb_position",
            "price_change_3",
            "price_change_5",
        ]
        self.load_models()

    def load_models(self):
        """載入訓練好的模型"""
        try:
            import joblib

            self.rf_model = joblib.load("models/reversal/reversal_rf_model.pkl")
            print("反轉模型載入成功")
        except Exception as e:
            print(f"模型載入失敗: {e}")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if len(df) < 50 or self.rf_model is None:
            return pd.Series(0, index=df.index)

        # 計算技術指標
        df_calc = df.copy()

        # 布林通道
        df_calc["bb_upper"], df_calc["bb_middle"], df_calc["bb_lower"] = talib.BBANDS(
            df_calc["close"], timeperiod=48, nbdevup=2.4, nbdevdn=2.4
        )

        # StochRSI
        df_calc["stochrsi_k"], df_calc["stochrsi_d"] = talib.STOCHRSI(
            df_calc["close"], timeperiod=14, fastk_period=3, fastd_period=3
        )

        # MACD
        df_calc["macd"], df_calc["macd_signal"], df_calc["macd_hist"] = talib.MACD(
            df_calc["close"]
        )

        # 成交量比率
        df_calc["volume_sma"] = df_calc["volume"].rolling(20).mean()
        df_calc["volume_ratio"] = df_calc["volume"] / df_calc["volume_sma"]

        # 布林位置
        df_calc["bb_position"] = (df_calc["close"] - df_calc["bb_lower"]) / (
            df_calc["bb_upper"] - df_calc["bb_lower"]
        )

        # 價格動量
        df_calc["price_change_3"] = df_calc["close"].pct_change(3)
        df_calc["price_change_5"] = df_calc["close"].pct_change(5)

        signals = pd.Series(0, index=df.index)

        for i in range(50, len(df)):
            current = df_calc.iloc[i]

            # 基本條件篩選
            if pd.isna(current["stochrsi_k"]) or pd.isna(current["volume_ratio"]):
                continue

            # A轉信號條件 (做空)
            if current["stochrsi_k"] > 95 and current["volume_ratio"] > 1.3:
                features = [
                    current[col] if pd.notna(current[col]) else 0
                    for col in self.feature_names
                ]
                rf_pred = self.rf_model.predict([features])[0]
                rf_prob = self.rf_model.predict_proba([features])[0]

                if rf_pred == 1 and rf_prob[1] > 0.3:
                    signals.iloc[i] = -1

            # V轉信號條件 (做多)
            elif current["stochrsi_k"] < 5 and current["volume_ratio"] > 1.3:
                features = [
                    current[col] if pd.notna(current[col]) else 0
                    for col in self.feature_names
                ]
                rf_pred = self.rf_model.predict([features])[0]
                rf_prob = self.rf_model.predict_proba([features])[0]

                if rf_pred == 2 and rf_prob[2] > 0.3:
                    signals.iloc[i] = 1

        return signals

    def get_description(self) -> str:
        return "Reversal(A轉V轉模式)"



class MLReversalStrategy(BaseStrategy):
    """基於機器學習的A/V轉折預測策略"""

    def __init__(self, config: StrategyConfig, model_path="models/reversal/reversal_model_v2.pkl"):
        super().__init__(config)
        self.model = None
        self.load_model(model_path)

    def get_description(self) -> str:
        """返回策略描述"""
        return "基於機器學習模型預測A轉V轉的交易策略,使用技術指標特徵進行轉折點識別"

    def load_model(self, model_path):
        """載入訓練好的模型"""
        try:
            self.model = joblib.load(model_path)
            print(f"✅ 模型載入成功: {model_path}")
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")

    def calculate_features(self, df):
        """計算模型所需特徵"""
        # RSI
        df["rsi"] = ta.rsi(df["close"], length=14)

        # MACD
        macd_data = ta.macd(df["close"])
        df["macd"] = macd_data["MACD_12_26_9"]
        df["macd_hist"] = macd_data["MACDh_12_26_9"]

        # 布林帶
        bb_data = ta.bbands(df["close"], length=20)
        if bb_data is not None and len(bb_data.columns) >= 3:
            df["bb_upper"] = bb_data.iloc[:, -1]  # 最後一列通常是上軌
            df["bb_lower"] = bb_data.iloc[:, 0]  # 第一列通常是下軌
            df["bb_middle"] = bb_data.iloc[:, 1]  # 中間列是中軌
        else:
            # 手動計算布林帶
            df["bb_middle"] = df["close"].rolling(20).mean()
            bb_std = df["close"].rolling(20).std()
            df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
            df["bb_lower"] = df["bb_middle"] - (bb_std * 2)

        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

        # 成交量比率
        df["volume_sma"] = ta.sma(df["volume"], length=20)
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # 價格動量
        df["price_momentum_5"] = df["close"].pct_change(5)
        df["price_momentum_10"] = df["close"].pct_change(10)

        # 波動率
        df["volatility_10"] = (
            df["close"].rolling(10).std() / df["close"].rolling(10).mean()
        )

        # 成交量趨勢
        df["volume_trend"] = (
            df["volume"]
            .rolling(10)
            .apply(
                lambda x: (np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0)
            )
        )

        return df

    def detect_potential_reversal(self, df):
        """檢測潛在轉折點"""
        if len(df) < 20:
            return pd.Series([False] * len(df), index=df.index)

        # 檢查是否接近局部極值
        high_20 = df["high"].rolling(20, center=True).max()
        low_20 = df["low"].rolling(20, center=True).min()

        is_near_high = df["close"] >= high_20 * 0.995
        is_near_low = df["close"] <= low_20 * 1.005

        return is_near_high | is_near_low

    def predict_reversal(self, features):
        """使用模型預測轉折類型"""
        if self.model is None:
            return None, 0

        try:
            feature_cols = [
                "rsi",
                "macd",
                "macd_hist",
                "bb_position",
                "volume_ratio",
                "price_momentum_5",
                "price_momentum_10",
                "volatility_10",
                "volume_trend",
            ]

            X = features[feature_cols].values.reshape(1, -1)

            prediction = self.model.predict(X)[0]
            probability = max(self.model.predict_proba(X)[0])

            reversal_type = "A_turn" if prediction == 1 else "V_turn"
            return reversal_type, probability

        except Exception:
            return None, 0

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """生成交易信號"""
        # 計算特徵
        df = self.calculate_features(df.copy())

        # 檢測潛在轉折點
        potential_reversals = self.detect_potential_reversal(df)

        # 初始化信號
        signals = pd.Series(0, index=df.index)

        if self.model is None:
            return signals

        # 向量化預測
        feature_cols = [
            "rsi",
            "macd",
            "macd_hist",
            "bb_position",
            "volume_ratio",
            "price_momentum_5",
            "price_momentum_10",
            "volatility_10",
            "volume_trend",
        ]

        # 過濾有效數據
        valid_mask = potential_reversals & ~df[feature_cols].isna().any(axis=1)
        valid_indices = df.index[valid_mask]

        if len(valid_indices) == 0:
            return signals

        # 批量預測
        X = df.loc[valid_indices, feature_cols].values
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X).max(axis=1)

        # 高信心度信號
        high_conf_mask = probabilities > 0.8
        high_conf_indices = valid_indices[high_conf_mask]
        high_conf_predictions = predictions[high_conf_mask]

        # 設置信號
        v_turn_mask = high_conf_predictions == 2  # V轉
        a_turn_mask = high_conf_predictions == 1  # A轉

        signals.loc[high_conf_indices[v_turn_mask]] = 1  # 買入
        signals.loc[high_conf_indices[a_turn_mask]] = -1  # 賣出

        return signals

