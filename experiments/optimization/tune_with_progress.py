#!/usr/bin/env python3
"""帶進度顯示的LSTM調優"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")
import sys
import time


def print_progress(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


print_progress("🚀 開始LSTM調優...")

# 載入數據
print_progress("📂 載入數據...")
df = pd.read_csv("data/btc_12months_15m.csv")
df = df.sort_values("timestamp").reset_index(drop=True)
print_progress(f"✅ 數據載入完成: {len(df)} 筆記錄")

# 準備數據
print_progress("🔧 準備特徵...")
features = df[["close", "volume"]].values
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
print_progress("✅ 特徵準備完成")


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


# 配置
configs = [
    {"units": 50, "dropout": 0.2, "batch": 32},
    {"units": 100, "dropout": 0.3, "batch": 16},
    {"units": 128, "dropout": 0.4, "batch": 8},
]

best_acc = 0
total_tests = len([30, 60]) * len(configs)
current_test = 0

for seq_len in [30, 60]:
    print_progress(f"📊 測試序列長度: {seq_len}")

    print_progress("🔄 創建序列...")
    X, y = create_sequences(features_scaled, seq_len)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print_progress(f"✅ 序列創建完成: 訓練{len(X_train)}, 測試{len(X_test)}")

    for i, config in enumerate(configs):
        current_test += 1
        print_progress(
            f"🧠 [{current_test}/{total_tests}] 測試配置: LSTM({config['units']}) dropout={config['dropout']}"
        )

        try:
            print_progress("  🏗️ 建立模型...")
            model = Sequential(
                [
                    LSTM(config["units"], input_shape=(seq_len, 2)),
                    Dropout(config["dropout"]),
                    Dense(25, activation="relu"),
                    Dense(1),
                ]
            )

            model.compile(optimizer="adam", loss="mse")
            print_progress("  ✅ 模型建立完成")

            print_progress("  🎯 開始訓練...")
            early_stop = EarlyStopping(
                patience=5, restore_best_weights=True, verbose=1
            )

            history = model.fit(
                X_train,
                y_train,
                epochs=20,
                batch_size=config["batch"],
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=1,
            )  # 顯示訓練進度

            print_progress("  📊 評估模型...")
            pred = model.predict(X_test, verbose=1).flatten()
            acc = evaluate_direction_accuracy(y_test, pred)

            print_progress(f"  📈 方向準確率: {acc:.1f}%")

            if acc > best_acc:
                best_acc = acc
                model.save("best_model.keras")
                print_progress(f"  🎯 新最佳模型! 準確率: {acc:.1f}%")

            print_progress(f"  ✅ 配置 {current_test} 完成")

        except Exception as e:
            print_progress(f"  ❌ 配置失敗: {e}")

print_progress(f"🏆 調優完成!")
print_progress(f"   最佳準確率: {best_acc:.1f}%")

if best_acc >= 55:
    print_progress("✅ 達到可用準確率!")
else:
    print_progress("⚠️ 準確率需要改善")
