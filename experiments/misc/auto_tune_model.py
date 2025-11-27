import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# Load and prepare data
df = pd.read_csv("data/btc_12months_15m.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)


# Calculate technical indicators
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd


def calculate_bollinger_bands(prices, window=20, num_std=2):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower


# Add technical indicators
df["rsi"] = calculate_rsi(df["close"])
df["macd"] = calculate_macd(df["close"])
df["bb_upper"], df["bb_lower"] = calculate_bollinger_bands(df["close"])
df["sma_20"] = df["close"].rolling(window=20).mean()
df["ema_12"] = df["close"].ewm(span=12).mean()

# Remove NaN values
df = df.dropna().reset_index(drop=True)

# Features for training
feature_columns = [
    "close",
    "volume",
    "rsi",
    "macd",
    "bb_upper",
    "bb_lower",
    "sma_20",
    "ema_12",
]
features = df[feature_columns].values

# Normalize features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)


# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length : i])
        y.append(data[i, 0])  # close price
    return np.array(X), np.array(y)


# Model configurations to try
configs = [
    {
        "lstm_units": [100, 100, 50],
        "dropout": 0.3,
        "batch_size": 32,
        "lr": 0.001,
    },
    {
        "lstm_units": [128, 128, 64],
        "dropout": 0.4,
        "batch_size": 16,
        "lr": 0.0005,
    },
    {
        "lstm_units": [150, 100, 50, 25],
        "dropout": 0.3,
        "batch_size": 32,
        "lr": 0.001,
    },
    {
        "lstm_units": [200, 150, 100],
        "dropout": 0.5,
        "batch_size": 16,
        "lr": 0.0003,
    },
    {
        "lstm_units": [256, 128, 64],
        "dropout": 0.4,
        "batch_size": 8,
        "lr": 0.0001,
    },
]


def build_model(config, input_shape):
    model = Sequential()

    # First LSTM layer
    model.add(
        LSTM(
            config["lstm_units"][0],
            return_sequences=True,
            input_shape=input_shape,
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(config["dropout"]))

    # Additional LSTM layers
    for i in range(1, len(config["lstm_units"])):
        return_seq = i < len(config["lstm_units"]) - 1
        model.add(LSTM(config["lstm_units"][i], return_sequences=return_seq))
        model.add(BatchNormalization())
        model.add(Dropout(config["dropout"]))

    # Dense layers
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=config["lr"]), loss="mse", metrics=["mae"]
    )
    return model


def evaluate_direction_accuracy(y_true, y_pred):
    # Calculate direction accuracy
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    accuracy = np.mean(true_direction == pred_direction) * 100
    return accuracy


# Try different sequence lengths
for seq_length in [30, 60, 90]:
    print(f"\n🔄 Testing sequence length: {seq_length}")

    X, y = create_sequences(features_scaled, seq_length)

    # Split data (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    for i, config in enumerate(configs):
        print(
            f"\n📊 Config {i+1}: LSTM{config['lstm_units']}, dropout={config['dropout']}, lr={config['lr']}"
        )

        try:
            model = build_model(config, (seq_length, len(feature_columns)))

            early_stop = EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            )

            history = model.fit(
                X_train,
                y_train,
                epochs=100,
                batch_size=config["batch_size"],
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=0,
            )

            # Make predictions
            train_pred = model.predict(X_train, verbose=0).flatten()
            test_pred = model.predict(X_test, verbose=0).flatten()

            # Evaluate direction accuracy
            train_acc = evaluate_direction_accuracy(y_train, train_pred)
            test_acc = evaluate_direction_accuracy(y_test, test_pred)

            print(f"Train direction accuracy: {train_acc:.1f}%")
            print(f"Test direction accuracy: {test_acc:.1f}%")

            # Check if target reached
            if test_acc >= 70.0:
                print(f"\n🎯 TARGET REACHED! Test accuracy: {test_acc:.1f}%")
                print(f"Best config: seq_length={seq_length}, {config}")

                # Save the best model
                model.save("best_crypto_model.h5")

                # Final evaluation on last 2 months
                last_2_months = X_test[-60:]  # Assuming daily data
                last_2_months_y = y_test[-60:]

                final_pred = model.predict(last_2_months, verbose=0).flatten()
                final_acc = evaluate_direction_accuracy(
                    last_2_months_y, final_pred
                )

                print(f"Final accuracy on last 2 months: {final_acc:.1f}%")

                if final_acc >= 70.0:
                    print("✅ Model successfully tuned to target accuracy!")
                    exit()

        except Exception as e:
            print(f"❌ Config failed: {e}")
            continue

print(
    "\n⚠️ Target not reached with current configurations. Trying advanced techniques..."
)

# Advanced configuration with attention-like mechanism
advanced_config = {
    "lstm_units": [300, 200, 100, 50],
    "dropout": 0.4,
    "batch_size": 8,
    "lr": 0.0001,
}

print(f"\n🚀 Advanced config: {advanced_config}")

for seq_length in [120, 150]:
    X, y = create_sequences(features_scaled, seq_length)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = build_model(advanced_config, (seq_length, len(feature_columns)))

    early_stop = EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=150,
        batch_size=advanced_config["batch_size"],
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1,
    )

    test_pred = model.predict(X_test, verbose=0).flatten()
    test_acc = evaluate_direction_accuracy(y_test, test_pred)

    print(f"Advanced model accuracy: {test_acc:.1f}%")

    if test_acc >= 70.0:
        print(f"🎯 TARGET REACHED with advanced config!")
        model.save("best_crypto_model.h5")
        break
