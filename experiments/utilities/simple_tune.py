#!/usr/bin/env python3
"""簡潔版LSTM調優"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")
import time


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


log("🚀 開始調優...")

# 載入數據
df = pd.read_csv("data/btc_12months_15m.csv")
features = df[["close", "volume"]].values
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
log(f"✅ 數據準備完成: {len(df)} 筆")


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length : i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def evaluate_accuracy(y_true, y_pred):
    true_dir = np.diff(y_true) > 0
    pred_dir = np.diff(y_pred) > 0
    return np.mean(true_dir == pred_dir) * 100


best_acc = 0
configs = [
    {"units": 64, "dropout": 0.3},
    {"units": 128, "dropout": 0.4},
]

for seq_len in [30, 60]:
    log(f"📊 序列長度: {seq_len}")
    X, y = create_sequences(features_scaled, seq_len)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    for i, config in enumerate(configs):
        log(f"  🧠 配置 {i+1}: LSTM({config['units']})")

        model = Sequential(
            [
                LSTM(config["units"], input_shape=(seq_len, 2)),
                Dropout(config["dropout"]),
                Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mse")

        # 訓練 (靜默模式)
        early_stop = EarlyStopping(patience=3, restore_best_weights=True)
        model.fit(
            X_train,
            y_train,
            epochs=15,
            batch_size=16,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0,
        )

        pred = model.predict(X_test, verbose=0).flatten()
        acc = evaluate_accuracy(y_test, pred)

        log(f"    準確率: {acc:.1f}%")

        if acc > best_acc:
            best_acc = acc
            model.save("best_simple_model.keras")
            log(f"    🎯 新最佳!")

log(f"🏆 完成! 最佳準確率: {best_acc:.1f}%")
