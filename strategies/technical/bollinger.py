#!/usr/bin/env python3
"""
Bollinger Bands 相關策略
"""

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import talib
import tensorflow as tf

from ..base import BaseStrategy, StrategyConfig


class BollingerStrategy(BaseStrategy):
    """布林通道策略"""

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="Bollinger Bands",
                params={"period": 32, "std": 3.2, "multiplier": 3.2},
            )
        super().__init__(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        std_val = self.params["std"]
        bb = ta.bbands(
            df["close"],
            length=self.params["period"],
            lower_std=std_val,
            upper_std=std_val,
        )
        signals = pd.Series(0, index=df.index)

        period = self.params["period"]
        std_val = self.params["std"]

        upper_band = bb[f"BBU_{period}_{std_val}_{std_val}"]
        lower_band = bb[f"BBL_{period}_{std_val}_{std_val}"]

        signals[df["close"] <= lower_band] = 1
        signals[df["close"] >= upper_band] = -1

        return signals

    def get_description(self) -> str:
        return (
            f"Bollinger Bands({self.params['period']},{self.params['std']}) "
            f"standard deviation bands"
        )


class BollingerExtremeStrategy(BaseStrategy):
    """布林極限策略"""

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            params = {"period": 32, "std": 3.2}
            config = StrategyConfig(name="Bollinger Extreme", params=params)
        super().__init__(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        std_val = self.params["std"]
        bb = ta.bbands(
            df["close"],
            length=self.params["period"],
            lower_std=std_val,
            upper_std=std_val,
        )
        signals = pd.Series(0, index=df.index)

        period = self.params["period"]
        signals[df["close"] < bb[f"BBL_{period}_{std_val}_{std_val}"]] = 1
        signals[df["close"] > bb[f"BBU_{period}_{std_val}_{std_val}"]] = -1

        return signals

    def get_description(self) -> str:
        return (
            f"Bollinger Extreme({self.params['period']},{self.params['std']}) "
            f"extreme breakout strategy"
        )


class EnhancedBollingerStrategy(BaseStrategy):
    """增強版布林帶+StochRSI策略 - 基於機器學習模型"""

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="EnhancedBollingerStochRSI",
                params={
                    "bb_period": 48,
                    "bb_std": 2.4,
                    "stochrsi_period": 14,
                    "model_threshold": 0.7,
                },
            )
        super().__init__(config)

        self.rf_model = None
        self.nn_model = None
        self.scaler = None
        self._load_models()

    def _load_models(self):
        """載入機器學習模型"""
        try:
            self.rf_model = joblib.load("models/bollinger/bb_stochrsi_rf_model.pkl")
            self.nn_model = tf.keras.models.load_model(
                "models/bollinger/bb_stochrsi_nn_model.keras"
            )
            self.scaler = joblib.load("models/bollinger/bb_stochrsi_scaler.pkl")
            print("✅ EnhancedBollingerStrategy模型載入成功")
        except Exception as e:
            print(f"⚠️ EnhancedBollingerStrategy模型載入失敗: {e}")
            self.rf_model = None
            self.nn_model = None
            self.scaler = None

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算技術指標特徵"""
        data = df.copy()

        data["price_change"] = data["close"].pct_change()
        data["high_low_ratio"] = data["high"] / data["low"]

        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            data["close"],
            timeperiod=self.params["bb_period"],
            nbdevup=self.params["bb_std"],
            nbdevdn=self.params["bb_std"],
        )
        data["bb_position"] = (data["close"] - bb_lower) / (bb_upper - bb_lower)
        data["bb_width"] = (bb_upper - bb_lower) / bb_middle
        data["close_bb_middle_ratio"] = data["close"] / bb_middle

        fastk, fastd = talib.STOCHRSI(
            data["close"], timeperiod=self.params["stochrsi_period"]
        )
        data["stochrsi_k"] = fastk
        data["stochrsi_d"] = fastd

        data["sma_5"] = talib.SMA(data["close"], 5)
        data["sma_10"] = talib.SMA(data["close"], 10)
        data["close_sma5_ratio"] = data["close"] / data["sma_5"]
        data["close_sma10_ratio"] = data["close"] / data["sma_10"]

        data["volatility_48"] = data["price_change"].rolling(48).std()
        data["volatility_24"] = data["price_change"].rolling(24).std()
        data["volatility_12"] = data["price_change"].rolling(12).std()

        data["rsi_14"] = talib.RSI(data["close"], 14)
        data["volume_sma_48"] = data["volume"].rolling(48).mean()
        data["volume_ratio"] = data["volume"] / data["volume_sma_48"]

        return data

    def _predict_signal(self, features: pd.Series) -> tuple:
        """使用模型預測交易信號"""
        if self.rf_model is None or self.nn_model is None:
            return 0, 0.0

        try:
            feature_cols = [
                "bb_position",
                "bb_width",
                "close_bb_middle_ratio",
                "stochrsi_k",
                "stochrsi_d",
                "volatility_48",
                "volatility_24",
                "volatility_12",
                "rsi_14",
                "close_sma5_ratio",
                "close_sma10_ratio",
                "volume_ratio",
                "high_low_ratio",
            ]

            feature_values = []
            for col in feature_cols:
                val = features.get(col, 0.0)
                feature_values.append(val if not pd.isna(val) else 0.0)

            X_df = pd.DataFrame([feature_values], columns=feature_cols)

            rf_pred = self.rf_model.predict(X_df)[0]
            rf_proba = self.rf_model.predict_proba(X_df)[0]

            X_scaled = self.scaler.transform(X_df)
            nn_proba = self.nn_model.predict(X_scaled, verbose=0)[0]
            nn_pred = np.argmax(nn_proba)

            if rf_pred == nn_pred and rf_pred != 0:
                confidence = max(rf_proba[int(rf_pred)], nn_proba[int(rf_pred)])
                return int(rf_pred), confidence
            else:
                return 0, 0.0

        except Exception as e:
            print(f"預測錯誤: {e}")
            return 0, 0.0

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """生成交易信號"""
        df_features = self._calculate_features(df)
        signals = pd.Series(0, index=df.index)

        for i in range(50, len(df)):
            row = df_features.iloc[i]

            bb_pos = row.get("bb_position", 0.5)
            stochrsi = row.get("stochrsi_k", 50)

            if (bb_pos < 0.3 and stochrsi < 30) or (bb_pos > 0.7 and stochrsi > 70):
                pred_signal, confidence = self._predict_signal(row)

                if confidence >= self.params["model_threshold"]:
                    if pred_signal == 1:
                        signals.iloc[i] = 1
                    elif pred_signal == 2:
                        signals.iloc[i] = -1

        return signals

    def get_description(self) -> str:
        return (
            f"EnhancedBollingerStochRSI("
            f"BB:{self.params['bb_period']},{self.params['bb_std']}, "
            f"StochRSI:{self.params['stochrsi_period']}, "
            f"ML:{self.params['model_threshold']})"
        )
