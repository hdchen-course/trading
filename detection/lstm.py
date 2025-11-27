#!/usr/bin/env python3
"""
LSTM Detection Methods
LSTM 模型檢測方法
"""

import numpy as np
import joblib
from tensorflow.keras.models import load_model
from .base import DetectionMethod
from utils.indicators import analyze_cumulative_metrics


class LSTMDetection(DetectionMethod):
    """LSTM模型檢測"""

    def __init__(self):
        super().__init__("LSTM Model")
        self.model = load_model("models/lstm/bidirectional_lstm_model_fixed.keras")
        with open("models/lstm/scaler.pkl", "rb") as f:
            self.scaler = joblib.load(f)

    def detect(self, df):
        n = len(df)
        timesteps = 96

        if n < timesteps:
            return np.zeros(n, dtype=bool)

        # 使用與訓練時相同的特徵處理
        analysis_df = analyze_cumulative_metrics(df, window_size=96)

        feature_columns = [
            "close_price",
            "ema_100",
            "ema_50",
            "avg_amplitude",
            "avg_volume",
            "avg_macd",
            "avg_rsi",
            "price_above_sar_ratio",
            "avg_bb_position",
            "volatility",
            "price_change_pct",
            "price_atr_ratio_1h",
            "volume_trend",
            "macd_2nd",
            "price_2nd",
            "vol_2nd",
            "atr_2nd",
        ]

        # 直接使用 analysis_df 的每一行作為一個時間步
        features = analysis_df[feature_columns].values
        features = np.nan_to_num(features)

        # 創建滑動窗口 - 每96行為一個序列
        X_sequences = []
        valid_indices = []

        for i in range(timesteps - 1, len(features)):
            sequence = features[i - timesteps + 1 : i + 1]
            X_sequences.append(sequence)
            valid_indices.append(i)

        if not X_sequences:
            return np.zeros(n, dtype=bool)

        X_sequences = np.array(X_sequences)

        # 標準化
        n_samples, n_timesteps, n_features = X_sequences.shape
        X_reshaped = X_sequences.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)

        # 預測
        predictions = self.model.predict(X_scaled, verbose=0)
        prediction_signals = predictions.flatten() > 0.8

        # 映射回原始索引
        signals = np.zeros(n, dtype=bool)
        for i, signal in enumerate(prediction_signals):
            if signal and i < len(valid_indices):
                # 映射到 analysis_df 的索引，然後再映射到原始 df
                analysis_idx = valid_indices[i]
                # analysis_df 的索引對應原始 df 的索引 + 95 (window_size - 1)
                original_idx = analysis_idx + 95
                if original_idx < n:
                    signals[original_idx] = True

        return signals

    def get_color(self):
        return "green"
