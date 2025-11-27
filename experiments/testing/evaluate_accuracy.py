import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def load_and_prepare_data():
    df = pd.read_csv("data/btc_12months_15m.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["close"]].values)

    sequence_length = 30
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length : i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split_idx = int(0.8 * len(X))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:], scaler


def create_best_model():
    model = Sequential(
        [
            LSTM(100, return_sequences=False, input_shape=(30, 1)),
            Dropout(0.3),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def evaluate_direction_accuracy():
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()

    model = create_best_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    predictions = model.predict(X_test, verbose=0)

    # Convert back to original scale for direction comparison
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_original = scaler.inverse_transform(predictions).flatten()

    # Get previous prices for direction calculation
    X_test_last = X_test[:, -1, 0]  # Last price in sequence
    prev_prices = scaler.inverse_transform(
        X_test_last.reshape(-1, 1)
    ).flatten()

    # Calculate direction accuracy
    actual_direction = y_test_original > prev_prices
    predicted_direction = predictions_original > prev_prices

    accuracy = np.mean(actual_direction == predicted_direction) * 100

    print(f"方向預測正確率: {accuracy:.2f}%")
    print(f"測試樣本數: {len(y_test)}")

    return accuracy


if __name__ == "__main__":
    accuracy = evaluate_direction_accuracy()
