#!/usr/bin/env python3
"""快速LSTM模型調優"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

print("🚀 開始快速LSTM調優...")

# 載入數據
try:
    df = pd.read_csv("data/btc_12months_15m.csv")
    print(f"✅ 數據載入成功: {len(df)} 筆記錄")
except:
    print("❌ 數據載入失敗")
    exit(1)

# 準備數據
df = df.sort_values("timestamp").reset_index(drop=True)
features = df[["close", "volume"]].values
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length : i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def evaluate_direction_accuracy(y_true, y_pred):
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    return np.mean(true_direction == pred_direction) * 100


# 快速配置
configs = [
    {"units": 50, "dropout": 0.2, "batch": 32},
    {"units": 100, "dropout": 0.3, "batch": 16},
    {"units": 128, "dropout": 0.4, "batch": 8},
]

best_acc = 0
best_config = None

for seq_len in [30, 60]:
    print(f"\n📊 測試序列長度: {seq_len}")

    X, y = create_sequences(features_scaled, seq_len)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    for i, config in enumerate(configs):
        print(
            f"  Config {i+1}: LSTM({config['units']}) dropout={config['dropout']}"
        )

        try:
            model = Sequential(
                [
                    LSTM(config["units"], input_shape=(seq_len, 2)),
                    Dropout(config["dropout"]),
                    Dense(25, activation="relu"),
                    Dense(1),
                ]
            )

            model.compile(optimizer="adam", loss="mse")

            early_stop = EarlyStopping(patience=5, restore_best_weights=True)

            model.fit(
                X_train,
                y_train,
                epochs=20,
                batch_size=config["batch"],
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=0,
            )

            pred = model.predict(X_test, verbose=0).flatten()
            acc = evaluate_direction_accuracy(y_test, pred)

            print(f"    方向準確率: {acc:.1f}%")

            if acc > best_acc:
                best_acc = acc
                best_config = {"seq_len": seq_len, **config}
                model.save("quick_model.h5")
                print(f"    🎯 新最佳模型! 準確率: {acc:.1f}%")

        except Exception as e:
            print(f"    ❌ 失敗: {e}")

print(f"\n🏆 最佳結果:")
print(f"   準確率: {best_acc:.1f}%")
print(f"   配置: {best_config}")

if best_acc >= 60:
    print("✅ 達到可用準確率!")
else:
    print("⚠️ 準確率偏低，需要更多調優")
