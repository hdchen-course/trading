import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import logging
from datetime import datetime

# 設置日誌
log_filename = (
    f"lstm_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)


def calculate_all_talib_features(df):
    """計算所有可用的TA-Lib技術指標"""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    open_price = df["open"].values
    volume = df["volume"].values

    features = df[["open", "high", "low", "close", "volume"]].copy()

    # 趨勢指標
    features["SMA_5"] = talib.SMA(close, 5)
    features["SMA_10"] = talib.SMA(close, 10)
    features["SMA_20"] = talib.SMA(close, 20)
    features["EMA_12"] = talib.EMA(close, 12)
    features["EMA_26"] = talib.EMA(close, 26)
    features["MACD"], features["MACD_signal"], features["MACD_hist"] = (
        talib.MACD(close)
    )
    features["ADX"] = talib.ADX(high, low, close, 14)
    features["AROON_up"], features["AROON_down"] = talib.AROON(high, low, 14)

    # 動量指標
    features["RSI"] = talib.RSI(close, 14)
    features["MOM"] = talib.MOM(close, 10)
    features["ROC"] = talib.ROC(close, 10)
    features["CCI"] = talib.CCI(high, low, close, 14)
    features["WILLR"] = talib.WILLR(high, low, close, 14)
    features["STOCH_k"], features["STOCH_d"] = talib.STOCH(high, low, close)

    # 波動率指標
    features["ATR"] = talib.ATR(high, low, close, 14)
    features["NATR"] = talib.NATR(high, low, close, 14)
    features["TRANGE"] = talib.TRANGE(high, low, close)

    # 布林帶
    features["BB_upper"], features["BB_middle"], features["BB_lower"] = (
        talib.BBANDS(close)
    )
    features["BB_width"] = features["BB_upper"] - features["BB_lower"]
    features["BB_position"] = (close - features["BB_lower"]) / features[
        "BB_width"
    ]

    # 成交量指標
    features["OBV"] = talib.OBV(close, volume)
    features["AD"] = talib.AD(high, low, close, volume)
    features["ADOSC"] = talib.ADOSC(high, low, close, volume)

    # K線型態識別 (返回0或100)
    features["DOJI"] = talib.CDLDOJI(open_price, high, low, close)
    features["HAMMER"] = talib.CDLHAMMER(open_price, high, low, close)
    features["ENGULFING"] = talib.CDLENGULFING(open_price, high, low, close)
    features["HARAMI"] = talib.CDLHARAMI(open_price, high, low, close)
    features["SHOOTING_STAR"] = talib.CDLSHOOTINGSTAR(
        open_price, high, low, close
    )
    features["MORNING_STAR"] = talib.CDLMORNINGSTAR(
        open_price, high, low, close
    )
    features["EVENING_STAR"] = talib.CDLEVENINGSTAR(
        open_price, high, low, close
    )
    features["THREE_WHITE_SOLDIERS"] = talib.CDL3WHITESOLDIERS(
        open_price, high, low, close
    )
    features["THREE_BLACK_CROWS"] = talib.CDL3BLACKCROWS(
        open_price, high, low, close
    )

    # 價格變化特徵
    features["price_change"] = close / np.roll(close, 1) - 1
    features["high_low_ratio"] = high / low
    features["volume_sma_ratio"] = volume / talib.SMA(volume, 20)

    return features


def prepare_sequences(data, sequence_length=60):
    """準備LSTM序列數據"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length : i])
        # 預測下一個時間點價格是否上漲
        y.append(
            1 if data[i, 3] > data[i - 1, 3] else 0
        )  # close price comparison
    return np.array(X), np.array(y)


def main():
    logging.info("開始載入數據...")
    df = pd.read_csv("data/btc_12months_15m.csv")
    logging.info(f"原始數據形狀: {df.shape}")

    # 計算所有技術指標
    logging.info("計算TA-Lib技術指標...")
    features_df = calculate_all_talib_features(df)

    # 移除NaN值
    features_df = features_df.dropna()
    logging.info(f"處理後數據形狀: {features_df.shape}")
    logging.info(f"特徵數量: {features_df.shape[1]}")

    # 標準化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_df.values)

    # 準備序列數據
    sequence_length = 60
    X, y = prepare_sequences(scaled_data, sequence_length)
    logging.info(f"序列數據形狀: X={X.shape}, y={y.shape}")

    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"訓練集: {X_train.shape}, 測試集: {X_test.shape}")

    # 建立模型
    model = Sequential(
        [
            LSTM(
                128,
                return_sequences=True,
                input_shape=(sequence_length, features_df.shape[1]),
            ),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            Dense(16, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    logging.info(f"模型參數數量: {model.count_params():,}")

    # 訓練回調
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy", patience=20, restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7
        ),
    ]

    # 訓練模型
    logging.info("開始訓練模型...")
    history = model.fit(
        X_train,
        y_train,
        epochs=150,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    # 評估模型
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"最終測試準確率: {test_accuracy:.4f}")

    if test_accuracy >= 0.70:
        model.save("lstm_70percent_achieved.keras")
        logging.info("🎉 達到70%準確率目標！模型已保存")
    else:
        model.save("lstm_comprehensive_attempt.keras")
        logging.info(f"未達到70%目標，當前準確率: {test_accuracy:.4f}")

    logging.info(f"訓練日誌已保存至: {log_filename}")


if __name__ == "__main__":
    main()
