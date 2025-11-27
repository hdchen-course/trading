import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


def create_features(data):
    """創建TA-Lib技術指標"""
    df = data.copy()

    # 基本技術指標
    df["sma_5"] = talib.SMA(df["close"], timeperiod=5)
    df["sma_10"] = talib.SMA(df["close"], timeperiod=10)
    df["ema_12"] = talib.EMA(df["close"], timeperiod=12)
    df["ema_26"] = talib.EMA(df["close"], timeperiod=26)

    # MACD
    macd, macd_signal, macd_hist = talib.MACD(df["close"])
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # RSI
    df["rsi_14"] = talib.RSI(df["close"], timeperiod=14)

    # 布林帶
    bb_upper, bb_middle, bb_lower = talib.BBANDS(df["close"])
    df["bb_upper"] = bb_upper
    df["bb_middle"] = bb_middle
    df["bb_lower"] = bb_lower

    # ATR
    df["atr"] = talib.ATR(df["high"], df["low"], df["close"])

    # 成交量指標
    df["obv"] = talib.OBV(df["close"], df["volume"])

    return df


def create_sequences(data, seq_length):
    """創建序列數據"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length : i])
        y.append(data[i, -1])  # 最後一列是目標
    return np.array(X), np.array(y)


def test_sequence_length(seq_length):
    print(f"\n=== 測試序列長度: {seq_length} ({seq_length*0.25:.1f}小時) ===")

    # 載入數據
    data = pd.read_csv("data/btc_12months_15m.csv")

    # 創建特徵
    data = create_features(data)

    # 創建目標變量（下一個15分鐘漲跌）
    data["target"] = (data["close"].shift(-1) > data["close"]).astype(int)

    # 選擇特徵
    feature_cols = [
        "close",
        "volume",
        "sma_5",
        "sma_10",
        "ema_12",
        "ema_26",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi_14",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "atr",
        "obv",
        "target",
    ]

    data = data[feature_cols].dropna()

    # 標準化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # 創建序列
    X, y = create_sequences(scaled_data, seq_length)

    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 移除目標列
    X_train = X_train[:, :, :-1]
    X_test = X_test[:, :, :-1]

    print(f"訓練集: {X_train.shape}, 測試集: {X_test.shape}")

    # 建立模型
    model = Sequential(
        [
            LSTM(
                64,
                return_sequences=True,
                input_shape=(seq_length, X_train.shape[2]),
            ),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            BatchNormalization(),
            Dense(16, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    # 訓練（只訓練10個epoch快速測試）
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.5),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0,
    )

    # 評估
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"測試準確率: {test_accuracy:.4f}")

    return test_accuracy


# 測試不同序列長度
sequence_lengths = [12, 24, 36, 48]
results = {}

for seq_len in sequence_lengths:
    try:
        accuracy = test_sequence_length(seq_len)
        results[seq_len] = accuracy
    except Exception as e:
        print(f"序列長度 {seq_len} 測試失敗: {e}")
        results[seq_len] = 0

# 顯示結果
print("\n=== 測試結果總結 ===")
for seq_len, acc in results.items():
    hours = seq_len * 0.25
    print(f"序列長度 {seq_len} ({hours:.1f}小時): {acc:.4f}")

best_seq = max(results, key=results.get)
print(
    f"\n最佳序列長度: {best_seq} ({best_seq*0.25:.1f}小時), 準確率: {results[best_seq]:.4f}"
)
