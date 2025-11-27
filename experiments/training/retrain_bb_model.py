import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

warnings.filterwarnings("ignore")


def collect_precise_patterns(data, lookback=10, lookahead=5):
    """收集精確的布林帶+MACD模式"""
    df = data.copy()

    # 計算指標
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(
        df["close"], timeperiod=32, nbdevup=3.2, nbdevdn=3.2
    )
    df["macd"], df["macd_signal"], _ = talib.MACD(df["close"])
    df["volume_ma"] = df["volume"].rolling(window=20).mean()

    patterns = []
    labels = []

    for i in range(lookback, len(df) - lookahead):
        current = df.iloc[i]

        # 跳過NaN值
        if (
            pd.isna(current["bb_upper"])
            or pd.isna(current["macd"])
            or pd.isna(current["volume_ma"])
        ):
            continue

        # 檢查條件
        price_at_top = (
            current["close"] >= current["bb_upper"] * 0.98
        )  # 接近上軌
        price_at_bottom = (
            current["close"] <= current["bb_lower"] * 1.02
        )  # 接近下軌

        # MACD交叉
        macd_cross_up = (
            current["macd"] > current["macd_signal"]
            and df.iloc[i - 1]["macd"] <= df.iloc[i - 1]["macd_signal"]
        )
        macd_cross_down = (
            current["macd"] < current["macd_signal"]
            and df.iloc[i - 1]["macd"] >= df.iloc[i - 1]["macd_signal"]
        )

        # 成交量條件：1.2倍平均成交量
        volume_spike = current["volume"] > current["volume_ma"] * 1.2

        # 精確條件：上軌+死叉 或 下軌+金叉
        condition_met = False
        signal_type = ""

        if price_at_top and macd_cross_down and volume_spike:
            condition_met = True
            signal_type = "上軌死叉"
        elif price_at_bottom and macd_cross_up and volume_spike:
            condition_met = True
            signal_type = "下軌金叉"

        if condition_met:
            # 收集前lookback根K線的OHLCV
            pattern_data = []
            for j in range(i - lookback, i):
                candle = df.iloc[j]
                pattern_data.extend(
                    [
                        candle["open"],
                        candle["high"],
                        candle["low"],
                        candle["close"],
                        candle["volume"],
                    ]
                )

            # 計算未來價格變化
            future_price = df.iloc[i + lookahead]["close"]
            current_price = current["close"]
            price_change = (future_price - current_price) / current_price

            # 標籤：上漲(1) 或 下跌(0)
            label = 1 if price_change > 0.01 else 0  # 1%閾值

            patterns.append(pattern_data)
            labels.append(label)

            if len(patterns) % 50 == 0:  # 每50個輸出一次
                print(f"已找到 {len(patterns)} 個模式...")

    return np.array(patterns), np.array(labels)


def train_precise_model(patterns, labels):
    """訓練精確模式預測模型"""
    if len(patterns) < 20:
        print(f"數據太少({len(patterns)}筆)，無法訓練模型")
        return None

    # 重塑數據
    X = patterns.reshape(patterns.shape[0], 10, 5)  # 10根K線，每根5個特徵

    # 分割數據
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]

    # 標準化
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 50)).reshape(
        -1, 10, 5
    )
    X_test_scaled = scaler.transform(X_test.reshape(-1, 50)).reshape(-1, 10, 5)

    # 建立模型
    model = Sequential(
        [
            LSTM(32, return_sequences=True, input_shape=(10, 5)),
            Dropout(0.3),
            LSTM(16),
            Dropout(0.3),
            Dense(8, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    # 訓練
    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=50,
        batch_size=8,
        verbose=0,
    )

    # 評估
    test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)[1]
    print(f"測試準確率: {test_acc:.4f}")

    return model, scaler


if __name__ == "__main__":
    # 載入數據
    data = pd.read_csv("data/btc_12months_15m.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    print("開始收集精確布林帶模式...")
    print("條件：")
    print("1. 上軌 + MACD死叉 + 成交量1.2倍")
    print("2. 下軌 + MACD金叉 + 成交量1.2倍")
    print("特徵：前10根K線的OHLCV")

    patterns, labels = collect_precise_patterns(data)

    if len(patterns) > 0:
        print(f"\n收集到 {len(patterns)} 個符合條件的模式")
        print(
            f"上漲標籤: {sum(labels)} 個 ({sum(labels)/len(labels)*100:.1f}%)"
        )
        print(
            f"下跌標籤: {len(labels)-sum(labels)} 個 ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)"
        )

        model, scaler = train_precise_model(patterns, labels)
        if model:
            model.save("bb_precise_model.keras")
            print("精確模型已保存為 bb_precise_model.keras")
    else:
        print("沒有找到符合條件的模式")
