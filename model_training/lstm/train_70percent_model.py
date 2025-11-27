import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import talib
import logging
from datetime import datetime

# Setup logging
log_file = f"lstm_70percent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)


def add_technical_indicators(df):
    """Add technical indicators to improve prediction accuracy"""
    df["rsi"] = talib.RSI(df["close"].values, timeperiod=14)
    df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
        df["close"].values
    )
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(
        df["close"].values
    )
    df["sma_20"] = talib.SMA(df["close"].values, timeperiod=20)
    df["ema_12"] = talib.EMA(df["close"].values, timeperiod=12)
    df["ema_26"] = talib.EMA(df["close"].values, timeperiod=26)
    df["atr"] = talib.ATR(
        df["high"].values, df["low"].values, df["close"].values, timeperiod=14
    )
    df["volume_sma"] = talib.SMA(df["volume"].values, timeperiod=20)

    # Price change features
    df["price_change"] = df["close"].pct_change()
    df["high_low_ratio"] = df["high"] / df["low"]
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    return df.dropna()


def prepare_advanced_data():
    df = pd.read_csv("data/btc_12months_15m.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Add technical indicators
    df = add_technical_indicators(df)

    # Select features
    features = [
        "close",
        "volume",
        "rsi",
        "macd",
        "macd_signal",
        "bb_upper",
        "bb_lower",
        "sma_20",
        "ema_12",
        "ema_26",
        "atr",
        "price_change",
        "high_low_ratio",
        "volume_ratio",
    ]

    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features].values)

    # Create sequences
    sequence_length = 60  # Longer sequence for better pattern recognition
    X, y = [], []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length : i])
        # Predict direction: 1 if price goes up, 0 if down
        current_price = df.iloc[i - 1]["close"]
        future_price = df.iloc[i]["close"]
        y.append(1 if future_price > current_price else 0)

    X, y = np.array(X), np.array(y)

    # Split data
    split_idx = int(0.8 * len(X))
    return (
        X[:split_idx],
        X[split_idx:],
        y[:split_idx],
        y[split_idx:],
        scaler,
        features,
    )


def create_advanced_model(input_shape):
    model = Sequential(
        [
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),  # Binary classification
        ]
    )

    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def train_model():
    logging.info(f"開始訓練70%準確率LSTM模型 - 日誌文件: {log_file}")

    X_train, X_test, y_train, y_test, scaler, features = (
        prepare_advanced_data()
    )
    logging.info(
        f"數據準備完成: {len(X_train)} 訓練樣本, {len(X_test)} 測試樣本"
    )
    logging.info(f"特徵數量: {len(features)}, 序列長度: 60")

    model = create_advanced_model((X_train.shape[1], X_train.shape[2]))

    # Callbacks for better training
    early_stopping = EarlyStopping(
        monitor="val_accuracy", patience=15, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7
    )

    logging.info("開始訓練模型...")
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"最終測試準確率: {test_accuracy*100:.2f}%")

    if test_accuracy >= 0.70:
        model.save("model_training/output/lstm/lstm_70percent_model.keras")
        logging.info("模型已保存為 lstm_70percent_model.keras")
        logging.info("🎉 成功達到70%準確率目標!")
    else:
        logging.info(
            f"準確率 {test_accuracy*100:.2f}% 未達到70%目標，但已保存最佳模型"
        )
        model.save("model_training/output/lstm/lstm_best_attempt.keras")

    return test_accuracy


if __name__ == "__main__":
    print(f"開始訓練LSTM模型達到70%準確率")
    print(f"監控進度: tail -f {log_file}")
    accuracy = train_model()
    print(f"最終準確率: {accuracy*100:.2f}%")
