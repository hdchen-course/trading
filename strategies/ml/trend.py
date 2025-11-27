#!/usr/bin/env python3
"""
趨勢預測策略
"""

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import talib
from typing import Dict

from ..base import BaseStrategy, StrategyConfig


class TrendMLStrategy(BaseStrategy):
    """趨勢機器學習策略"""

    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="Trend ML",
                params={
                    "ema_fast": 20,
                    "ema_slow": 50,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "volume_period": 20,
                    "model_threshold": 0.7,
                    "af_start": 0.02,
                    "af_max": 0.2,
                },
            )
        super().__init__(config)

        # 載入模型
        try:
            self.rf_model = joblib.load("models/trend/trend_rf_model.pkl")
            self.nn_model = joblib.load("models/trend/trend_nn_model.pkl")
        except FileNotFoundError:
            print("Warning: Trend models not found. Please train models first.")
            self.rf_model = None
            self.nn_model = None

    def _calculate_ema(self, data, period):
        return data.ewm(span=period).mean()

    def _calculate_macd(self, data, fast=12, slow=26, signal=9):
        ema_fast = self._calculate_ema(data, fast)
        ema_slow = self._calculate_ema(data, slow)
        macd = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd, signal)
        return macd, signal_line

    def _calculate_sar(self, high, low, close):
        sar = np.zeros(len(close))
        trend = np.zeros(len(close))
        af = self.params["af_start"]
        ep = high[0]
        sar[0] = low[0]
        trend[0] = 1

        for i in range(1, len(close)):
            if trend[i - 1] == 1:
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
                if low[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep
                    ep = low[i]
                    af = self.params["af_start"]
                else:
                    trend[i] = 1
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + self.params["af_start"], self.params["af_max"])
            else:
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
                if high[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep
                    ep = high[i]
                    af = self.params["af_start"]
                else:
                    trend[i] = -1
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + self.params["af_start"], self.params["af_max"])
        return sar

    def _calculate_features(self, df):
        df_calc = df.copy()

        # EMA計算
        df_calc["ema_20"] = self._calculate_ema(
            df_calc["close"], self.params["ema_fast"]
        )
        df_calc["ema_50"] = self._calculate_ema(
            df_calc["close"], self.params["ema_slow"]
        )

        # EMA斜率
        df_calc["ema_slope_20"] = df_calc["ema_20"].pct_change(5)
        df_calc["ema_slope_50"] = df_calc["ema_50"].pct_change(5)

        # MACD
        macd_fast = self.params.get("macd_fast", 12)
        macd_slow = self.params.get("macd_slow", 26)
        macd_signal = self.params.get("macd_signal", 9)
        df_calc["macd"], df_calc["macd_signal"] = self._calculate_macd(
            df_calc["close"], macd_fast, macd_slow, macd_signal
        )
        df_calc["macd_histogram"] = df_calc["macd"] - df_calc["macd_signal"]

        # 成交量比率
        volume_period = self.params.get("volume_period", 20)
        df_calc["volume_sma"] = df_calc["volume"].rolling(volume_period).mean()
        df_calc["volume_ratio"] = df_calc["volume"] / df_calc["volume_sma"]

        return df_calc

    def _predict_signal(self, row):
        if self.rf_model is None or self.nn_model is None:
            return 0, 0.0

        features = [
            "ema_slope_20",
            "ema_slope_50",
            "macd",
            "macd_histogram",
            "volume_ratio",
        ]
        X = pd.DataFrame([row[features]], columns=features)

        # 使用Random Forest預測
        rf_pred = self.rf_model.predict(X)[0]
        rf_proba = self.rf_model.predict_proba(X)[0]

        # 獲取最高概率
        max_prob = max(rf_proba)

        return rf_pred, max_prob

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df_features = self._calculate_features(df)
        signals = pd.Series(0, index=df.index)

        for i in range(50, len(df)):
            row = df_features.iloc[i]

            # 使用模型預測
            pred_signal, confidence = self._predict_signal(row)

            if confidence >= self.params["model_threshold"]:
                if pred_signal == 1:  # 做多
                    signals.iloc[i] = 1
                elif pred_signal == 2:  # 做空
                    signals.iloc[i] = -1

        return signals

    def get_description(self) -> str:
        return (
            f"TrendML(EMA:{self.params['ema_fast']},"
            f"{self.params['ema_slow']}, "
            f"Threshold:{self.params['model_threshold']})"
        )



class TrendPredictionStrategy(BaseStrategy):
    """
    基於三狀態趨勢預測的交易策略

    使用機器學習模型預測市場趨勢狀態:
    - 狀態 0: 下跌趨勢 → 賣出信號
    - 狀態 1: 橫盤整理 → 無信號
    - 狀態 2: 上漲趨勢 → 買入信號

    特徵工程包括:
    - RSI: 相對強弱指標,衡量超買超賣
    - MACD: 移動平均收斂發散,趨勢動量指標
    - 布林帶: 價格通道,衡量波動性
    - 成交量指標: 市場參與度
    - 價格變化率: 短期動量
    """

    def __init__(self, config: StrategyConfig, model_path="models/trend/trend_prediction_model.pkl"):
        """
        初始化趨勢預測策略

        Args:
            config: 策略配置對象
            model_path: 預訓練模型文件路徑 (.pkl 格式)
        """
        super().__init__(config)  # 調用父類初始化方法
        self.model = None  # 機器學習模型對象
        self.load_model(model_path)  # 載入預訓練模型

    def get_description(self) -> str:
        """返回策略描述,用於日誌和報告"""
        return "基於機器學習預測趨勢狀態（上漲/下跌/橫盤）的交易策略"

    def load_model(self, model_path):
        """
        載入預訓練的機器學習模型

        使用 joblib 載入 scikit-learn 或其他兼容模型
        模型應該是分類器,輸出 0/1/2 三個狀態
        """
        try:
            self.model = joblib.load(model_path)  # 載入序列化模型
            print(f"✅ 趨勢預測模型載入成功: {model_path}")
        except Exception as e:
            print(f"❌ 趨勢預測模型載入失敗: {e}")

    def calculate_trend_features(self, df):
        """
        計算趨勢預測所需的技術指標特徵

        這些特徵將作為機器學習模型的輸入:
        1. RSI: 衡量價格動量和超買超賣狀態
        2. MACD: 趨勢跟隨指標,捕捉趨勢變化
        3. 布林帶: 價格通道,衡量波動性和支撐阻力
        4. 成交量指標: 確認價格走勢的有效性
        5. 價格變化率: 短期動量指標

        Args:
            df: 包含 OHLCV 數據的 DataFrame
        """
        # === RSI (相對強弱指標) ===
        # 範圍 0-100,>70 超買,<30 超賣
        df["rsi"] = ta.rsi(df["close"], length=14)

        # === MACD (移動平均收斂發散) ===
        # 由快線、慢線和柱狀圖組成,用於判斷趨勢轉折
        macd_data = ta.macd(df["close"])
        df["macd"] = macd_data["MACD_12_26_9"]  # MACD 主線
        df["macd_hist"] = macd_data["MACDh_12_26_9"]  # MACD 柱狀圖

        # === 布林帶 (Bollinger Bands) ===
        # 由中軌（移動平均）和上下軌（標準差通道）組成
        bb_data = ta.bbands(df["close"], length=20)
        if bb_data is not None and len(bb_data.columns) >= 3:
            df["bb_upper"] = bb_data.iloc[:, -1]
            df["bb_lower"] = bb_data.iloc[:, 0]
        else:
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
        df["price_momentum_20"] = df["close"].pct_change(20)

        # 波動率
        df["volatility_10"] = (
            df["close"].rolling(10).std() / df["close"].rolling(10).mean()
        )
        df["volatility_20"] = (
            df["close"].rolling(20).std() / df["close"].rolling(20).mean()
        )

        # ATR比率
        atr_data = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr_ratio"] = atr_data / df["close"]

        return df

    def predict_trend(self, features):
        """預測趨勢狀態"""
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
                "price_momentum_20",
                "volatility_10",
                "volatility_20",
                "atr_ratio",
            ]

            X = features[feature_cols].values.reshape(1, -1)
            prediction = self.model.predict(X)[0]
            probability = max(self.model.predict_proba(X)[0])

            trend_map = {0: "downtrend", 1: "sideways", 2: "uptrend"}
            return trend_map[prediction], probability

        except Exception:
            return None, 0

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self.calculate_trend_features(df.copy())
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
            "price_momentum_20",
            "volatility_10",
            "volatility_20",
            "atr_ratio",
        ]

        # 過濾有效數據
        valid_mask = ~df[feature_cols].isna().any(axis=1)
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

        # 設置信號 (0=下跌, 1=橫盤, 2=上漲)
        signals.loc[high_conf_indices[high_conf_predictions == 2]] = 1  # 上漲
        signals.loc[high_conf_indices[high_conf_predictions == 0]] = -1  # 下跌

        return signals



class SeparateTrendStrategy(BaseStrategy):
    """基於分離上漲下跌模型的交易策略"""

    def __init__(
        self,
        config: StrategyConfig,
        uptrend_model_path="models/trend/uptrend_model.pkl",
        downtrend_model_path="models/trend/downtrend_model.pkl",
    ):
        super().__init__(config)
        self.uptrend_model = None
        self.downtrend_model = None
        self.load_models(uptrend_model_path, downtrend_model_path)

    def get_description(self) -> str:
        return "基於分離的上漲下跌預測模型的交易策略,準確率>93%"

    def load_models(self, uptrend_path, downtrend_path):
        try:
            self.uptrend_model = joblib.load(uptrend_path)
            self.downtrend_model = joblib.load(downtrend_path)
            print("✅ 分離趨勢模型載入成功")
        except Exception as e:
            print(f"❌ 分離趨勢模型載入失敗: {e}")

    def calculate_trend_features(self, df):
        """計算趨勢預測所需特徵"""
        # RSI
        df["rsi"] = ta.rsi(df["close"], length=14)

        # MACD
        macd_data = ta.macd(df["close"])
        df["macd"] = macd_data["MACD_12_26_9"]
        df["macd_hist"] = macd_data["MACDh_12_26_9"]

        # 布林帶
        bb_data = ta.bbands(df["close"], length=20)
        if bb_data is not None and len(bb_data.columns) >= 3:
            df["bb_upper"] = bb_data.iloc[:, -1]
            df["bb_lower"] = bb_data.iloc[:, 0]
        else:
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
        df["price_momentum_20"] = df["close"].pct_change(20)

        # 波動率
        df["volatility_10"] = (
            df["close"].rolling(10).std() / df["close"].rolling(10).mean()
        )
        df["volatility_20"] = (
            df["close"].rolling(20).std() / df["close"].rolling(20).mean()
        )

        # ATR比率
        atr_data = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr_ratio"] = atr_data / df["close"]

        return df

    def predict_separate_trends(self, features):
        """使用分離模型預測趨勢"""
        if self.uptrend_model is None or self.downtrend_model is None:
            return None, 0, 0

        try:
            feature_cols = [
                "rsi",
                "macd",
                "macd_hist",
                "bb_position",
                "volume_ratio",
                "price_momentum_5",
                "price_momentum_10",
                "price_momentum_20",
                "volatility_10",
                "volatility_20",
                "atr_ratio",
            ]

            X = features[feature_cols].values.reshape(1, -1)

            up_prob = self.uptrend_model.predict_proba(X)[0][1]
            down_prob = self.downtrend_model.predict_proba(X)[0][1]

            if up_prob > 0.7:
                return "uptrend", up_prob, down_prob
            elif down_prob > 0.7:
                return "downtrend", up_prob, down_prob
            else:
                return "sideways", up_prob, down_prob

        except Exception:
            return None, 0, 0

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self.calculate_trend_features(df.copy())
        signals = pd.Series(0, index=df.index)

        if self.uptrend_model is None or self.downtrend_model is None:
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
            "price_momentum_20",
            "volatility_10",
            "volatility_20",
            "atr_ratio",
        ]

        # 過濾有效數據
        valid_mask = ~df[feature_cols].isna().any(axis=1)
        valid_indices = df.index[valid_mask]

        if len(valid_indices) == 0:
            return signals

        # 批量預測
        X = df.loc[valid_indices, feature_cols].values
        up_probs = self.uptrend_model.predict_proba(X)[:, 1]
        down_probs = self.downtrend_model.predict_proba(X)[:, 1]

        # 高信心度信號
        uptrend_mask = up_probs > 0.8
        downtrend_mask = down_probs > 0.8

        signals.loc[valid_indices[uptrend_mask]] = 1  # 上漲
        signals.loc[valid_indices[downtrend_mask]] = -1  # 下跌

        return signals

