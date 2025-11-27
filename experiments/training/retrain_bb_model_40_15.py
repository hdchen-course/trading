import pandas as pd
import numpy as np
import talib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


def detect_reversal_patterns(df):
    patterns = {
        "HAMMER": talib.CDLHAMMER(
            df["open"], df["high"], df["low"], df["close"]
        ),
        "SHOOTINGSTAR": talib.CDLSHOOTINGSTAR(
            df["open"], df["high"], df["low"], df["close"]
        ),
        "DOJI": talib.CDLDOJI(df["open"], df["high"], df["low"], df["close"]),
        "ENGULFING": talib.CDLENGULFING(
            df["open"], df["high"], df["low"], df["close"]
        ),
        "HARAMI": talib.CDLHARAMI(
            df["open"], df["high"], df["low"], df["close"]
        ),
    }

    reversal_signal = np.zeros(len(df), dtype=bool)
    for pattern_values in patterns.values():
        reversal_signal |= np.abs(pattern_values) >= 85

    return reversal_signal


def create_features(df, lookback=21):
    features = []
    labels = []

    # 計算技術指標
    bb_upper, bb_middle, bb_lower = talib.BBANDS(
        df["close"], timeperiod=40, nbdevup=1.5, nbdevdn=1.5
    )
    volume_ma = df["volume"].rolling(20).mean()

    # 信號條件
    bb_condition = (df["close"] <= bb_lower) | (df["close"] >= bb_upper)
    volume_condition = df["volume"] >= (volume_ma * 1.2)
    reversal_condition = detect_reversal_patterns(df)

    signal_points = bb_condition & volume_condition & reversal_condition
    signal_indices = np.where(signal_points)[0]

    print(f"找到 {len(signal_indices)} 個信號點")

    for idx in signal_indices:
        if idx >= lookback and idx < len(df) - 5:
            # 特徵：前21根K線的OHLCV
            feature_window = df.iloc[idx - lookback : idx][
                ["open", "high", "low", "close", "volume"]
            ].values
            features.append(feature_window.flatten())

            # 標籤：未來5根K線的方向
            future_close = df.iloc[idx + 5]["close"]
            current_close = df.iloc[idx]["close"]
            label = 1 if future_close > current_close else 0
            labels.append(label)

    return np.array(features), np.array(labels)


# 讀取數據
print("讀取訓練數據...")
df = pd.read_csv("training_data/btc_12months_15m.csv")
print(f"數據長度: {len(df)}")

# 創建特徵和標籤
print("創建特徵和標籤...")
X, y = create_features(df)

print(f"特徵形狀: {X.shape}")
print(f"標籤分布: 上漲={np.sum(y)}, 下跌={len(y)-np.sum(y)}")
print(f"上漲比例: {np.mean(y):.2%}")

# 標準化特徵
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 重塑為LSTM格式
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 21, 5)

# 分割數據
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42
)

print(f"訓練集: {len(X_train)} 樣本")
print(f"測試集: {len(X_test)} 樣本")

# 建立模型
model = Sequential(
    [
        LSTM(50, return_sequences=True, input_shape=(21, 5)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

# 訓練模型
print("開始訓練模型...")
history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_test, y_test),
    verbose=1,
)

# 評估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"測試準確率: {test_accuracy:.4f}")

# 保存模型
model.save("model/bb_model_40_15.keras")
print("模型已保存為 model/bb_model_40_15.keras")
