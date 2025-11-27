import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import logging
import sys
from datetime import datetime

# Setup logging
log_file = f"lstm_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)


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


def create_model(lstm_units, dropout_rate):
    model = Sequential(
        [
            LSTM(lstm_units, return_sequences=False, input_shape=(30, 1)),
            Dropout(dropout_rate),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def tune_hyperparameters():
    logging.info(f"Starting LSTM hyperparameter tuning - Log file: {log_file}")

    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    logging.info(
        f"Data loaded: {len(X_train)} train, {len(X_test)} test samples"
    )

    param_grid = [
        {"lstm_units": 50, "dropout_rate": 0.2},
        {"lstm_units": 100, "dropout_rate": 0.2},
        {"lstm_units": 50, "dropout_rate": 0.3},
        {"lstm_units": 100, "dropout_rate": 0.3},
    ]

    best_loss = float("inf")
    best_params = None

    for i, params in enumerate(param_grid, 1):
        logging.info(f"Testing config {i}/{len(param_grid)}: {params}")

        model = create_model(params["lstm_units"], params["dropout_rate"])

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0,
        )

        final_loss = min(history.history["val_loss"])
        logging.info(f"Config {i} final validation loss: {final_loss:.6f}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params
            logging.info(f"New best config found: {best_params}")

    logging.info(
        f"Tuning complete. Best params: {best_params}, Best loss: {best_loss:.6f}"
    )
    return best_params, best_loss


if __name__ == "__main__":
    print(f"LSTM tuning started. Monitor progress: tail -f {log_file}")
    tune_hyperparameters()
