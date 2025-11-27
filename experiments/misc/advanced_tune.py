#!/usr/bin/env python3
"""進階LSTM模型調優 - 添加技術指標"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

print("🚀 開始進階LSTM調優...")

# 載入數據
df = pd.read_csv("data/btc_12months_15m.csv")
df = df.sort_values("timestamp").reset_index(drop=True)


# 計算技術指標
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, fast=12, slow=26):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    return ema_fast - ema_slow


# 添加技術指標
df["rsi"] = calculate_rsi(df["close"])
df["macd"] = calculate_macd(df["close"])
df["sma_20"] = df["close"].rolling(window=20).mean()
df["price_change"] = df["close"].pct_change()

# 移除NaN
df = df.dropna().reset_index(drop=True)

# 使用更多特徵
features = df[
    ["close", "volume", "rsi", "macd", "sma_20", "price_change"]
].values
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

print(f"✅ 數據準備完成: {len(df)} 筆記錄, {features.shape[1]} 個特徵")


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


# 進階配置
configs = [
    {"layers": [128, 64], "dropout": 0.3, "batch": 16, "lr": 0.001},
    {"layers": [150, 100, 50], "dropout": 0.4, "batch": 8, "lr": 0.0005},
    {"layers": [200, 100], "dropout": 0.5, "batch": 16, "lr": 0.0003},
]

best_acc = 0
best_config = None

for seq_len in [45, 90]:
    print(f"\n📊 測試序列長度: {seq_len}")

    X, y = create_sequences(features_scaled, seq_len)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    for i, config in enumerate(configs):
        print(
            f"  Config {i+1}: LSTM{config['layers']} dropout={config['dropout']}"
        )

        try:
            model = Sequential()

            # 第一層LSTM
            model.add(
                LSTM(
                    config["layers"][0],
                    return_sequences=len(config["layers"]) > 1,
                    input_shape=(seq_len, features.shape[1]),
                )
            )
            model.add(BatchNormalization())
            model.add(Dropout(config["dropout"]))

            # 額外LSTM層
            for j in range(1, len(config["layers"])):
                return_seq = j < len(config["layers"]) - 1
                model.add(
                    LSTM(config["layers"][j], return_sequences=return_seq)
                )
                model.add(BatchNormalization())
                model.add(Dropout(config["dropout"]))

            # 輸出層
            model.add(Dense(50, activation="relu"))
            model.add(Dropout(0.3))
            model.add(Dense(1))

            from tensorflow.keras.optimizers import Adam

            model.compile(
                optimizer=Adam(learning_rate=config["lr"]), loss="mse"
            )

            early_stop = EarlyStopping(patience=10, restore_best_weights=True)

            model.fit(
                X_train,
                y_train,
                epochs=50,
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
                model.save("advanced_model.keras")
                print(f"    🎯 新最佳模型! 準確率: {acc:.1f}%")

        except Exception as e:
            print(f"    ❌ 失敗: {e}")

print(f"\n🏆 最佳結果:")
print(f"   準確率: {best_acc:.1f}%")
print(f"   配置: {best_config}")

if best_acc >= 55:
    print("✅ 達到改善的準確率!")
elif best_acc >= 50:
    print("⚠️ 略有改善，但仍需優化")
else:
    print("❌ 準確率仍然偏低")
