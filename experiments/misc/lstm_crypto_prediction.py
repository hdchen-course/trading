#!/usr/bin/env python3
"""
LSTM加密貨幣價格預測模型
整合到現有交易系統中
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow未安裝，LSTM功能不可用")
    TENSORFLOW_AVAILABLE = False


class CryptoLSTMPredictor:
    def __init__(
        self, sequence_length=60, features=["close", "volume", "high", "low"]
    ):
        self.sequence_length = sequence_length
        self.features = features
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False

    def prepare_data(self, df):
        """準備LSTM訓練數據"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow未安裝")

        # 選擇特徵
        data = df[self.features].values

        # 標準化
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - self.sequence_length : i])
            y.append(scaled_data[i, 0])  # 預測close價格

        return np.array(X), np.array(y)

    def create_model(self, input_shape):
        """創建LSTM模型"""
        model = Sequential(
            [
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation="relu"),
                Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def train(self, df, test_size=0.2, epochs=50, batch_size=32):
        """訓練模型"""
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow未安裝，無法訓練LSTM模型")
            return False

        print("🧠 準備LSTM訓練數據...")
        X, y = self.prepare_data(df)

        # 分割數據
        split = int((1 - test_size) * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(f"📊 訓練集: {len(X_train)}, 測試集: {len(X_test)}")

        # 創建模型
        self.model = self.create_model((X_train.shape[1], X_train.shape[2]))

        # 早停回調
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        # 訓練
        print("🚀 開始訓練LSTM模型...")
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1,
        )

        # 評估
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        # 反標準化
        train_pred_rescaled = self.inverse_transform_predictions(train_pred)
        test_pred_rescaled = self.inverse_transform_predictions(test_pred)
        y_train_rescaled = self.inverse_transform_predictions(
            y_train.reshape(-1, 1)
        )
        y_test_rescaled = self.inverse_transform_predictions(
            y_test.reshape(-1, 1)
        )

        # 計算指標
        train_rmse = np.sqrt(
            mean_squared_error(y_train_rescaled, train_pred_rescaled)
        )
        test_rmse = np.sqrt(
            mean_squared_error(y_test_rescaled, test_pred_rescaled)
        )
        train_mae = mean_absolute_error(y_train_rescaled, train_pred_rescaled)
        test_mae = mean_absolute_error(y_test_rescaled, test_pred_rescaled)

        print(f"\n📈 模型評估結果:")
        print(f"訓練集 RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}")
        print(f"測試集 RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}")

        self.is_trained = True
        return True

    def inverse_transform_predictions(self, predictions):
        """反標準化預測結果"""
        # 創建與原始特徵相同形狀的數組
        dummy = np.zeros((predictions.shape[0], len(self.features)))
        dummy[:, 0] = predictions.flatten()

        # 反標準化
        rescaled = self.scaler.inverse_transform(dummy)
        return rescaled[:, 0]

    def predict_next(self, df, steps=1):
        """預測未來價格"""
        if not self.is_trained:
            print("❌ 模型尚未訓練")
            return None

        # 獲取最後sequence_length個數據點
        last_data = df[self.features].tail(self.sequence_length).values
        scaled_data = self.scaler.transform(last_data)

        predictions = []
        current_sequence = scaled_data.copy()

        for _ in range(steps):
            # 預測下一個點
            X = current_sequence.reshape(
                1, self.sequence_length, len(self.features)
            )
            pred = self.model.predict(X, verbose=0)[0, 0]
            predictions.append(pred)

            # 更新序列（滑動窗口）
            new_point = np.zeros(len(self.features))
            new_point[0] = pred  # close價格
            # 其他特徵使用最後一個值
            new_point[1:] = current_sequence[-1, 1:]

            current_sequence = np.vstack([current_sequence[1:], new_point])

        # 反標準化
        predictions_rescaled = self.inverse_transform_predictions(
            np.array(predictions).reshape(-1, 1)
        )
        return predictions_rescaled

    def save_model(self, filepath):
        """保存模型"""
        if self.model:
            self.model.save(filepath)
            print(f"💾 模型已保存至: {filepath}")


def main():
    """主函數 - 演示LSTM預測"""
    if not TENSORFLOW_AVAILABLE:
        print("❌ 請安裝TensorFlow: pip install tensorflow")
        return

    # 讀取12個月數據
    try:
        df = pd.read_csv("data/btc_12months_15m.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        print(f"✅ 成功讀取12個月數據: {len(df)} 條記錄")
        print(
            f"📅 數據範圍: {df['timestamp'].min()} 到 {df['timestamp'].max()}"
        )
    except FileNotFoundError:
        print("❌ 數據文件不存在: data/btc_12months_15m.csv")
        return

    # 分割數據：前10個月訓練，後2個月驗證
    split_date = df["timestamp"].min() + pd.DateOffset(months=10)
    train_df = df[df["timestamp"] < split_date].copy()
    test_df = df[df["timestamp"] >= split_date].copy()

    print(
        f"📊 訓練數據: {len(train_df)} 條 ({train_df['timestamp'].min()} 到 {train_df['timestamp'].max()})"
    )
    print(
        f"📊 測試數據: {len(test_df)} 條 ({test_df['timestamp'].min()} 到 {test_df['timestamp'].max()})"
    )

    # 創建預測器
    predictor = CryptoLSTMPredictor(sequence_length=60)

    # 訓練模型
    print("\n🚀 開始LSTM訓練...")
    success = predictor.train(train_df, test_size=0.2, epochs=50)

    if success:
        # 在測試集上評估
        print("\n🔍 在最後2個月數據上測試...")

        # 預測測試集
        test_predictions = []
        test_actuals = []

        for i in range(predictor.sequence_length, len(test_df)):
            # 獲取前60個數據點
            input_data = test_df.iloc[i - predictor.sequence_length : i]
            actual_price = test_df.iloc[i]["close"]

            # 預測下一個價格
            pred_price = predictor.predict_next(input_data, steps=1)[0]

            test_predictions.append(pred_price)
            test_actuals.append(actual_price)

        # 計算準確度
        test_predictions = np.array(test_predictions)
        test_actuals = np.array(test_actuals)

        # 方向準確度 (上漲/下跌預測正確率)
        pred_direction = np.diff(test_predictions) > 0
        actual_direction = np.diff(test_actuals) > 0
        direction_accuracy = np.mean(pred_direction == actual_direction) * 100

        # 價格準確度
        rmse = np.sqrt(np.mean((test_predictions - test_actuals) ** 2))
        mae = np.mean(np.abs(test_predictions - test_actuals))
        mape = (
            np.mean(np.abs((test_predictions - test_actuals) / test_actuals))
            * 100
        )

        print(f"\n📈 測試結果 (最後2個月):")
        print(f"方向預測準確率: {direction_accuracy:.1f}%")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"MAPE: {mape:.2f}%")

        # 判斷是否達到目標
        if direction_accuracy >= 70:
            print(
                f"🎉 達到目標！方向預測準確率 {direction_accuracy:.1f}% >= 70%"
            )

            # 預測未來24小時
            future_prices = predictor.predict_next(
                df.tail(predictor.sequence_length), steps=96
            )
            current_price = df["close"].iloc[-1]

            print(f"\n🔮 未來24小時價格預測:")
            print(f"當前價格: ${current_price:.2f}")
            print(f"1小時後: ${future_prices[3]:.2f}")
            print(f"6小時後: ${future_prices[23]:.2f}")
            print(f"12小時後: ${future_prices[47]:.2f}")
            print(f"24小時後: ${future_prices[95]:.2f}")

            change_24h = (
                (future_prices[95] - current_price) / current_price * 100
            )
            print(f"24小時預期變化: {change_24h:+.2f}%")

            # 保存模型
            predictor.save_model("data/lstm_crypto_model.h5")

        else:
            print(
                f"❌ 未達目標，方向預測準確率 {direction_accuracy:.1f}% < 70%"
            )
            print("🔧 建議調整模型參數...")

            # 自動調整建議
            if direction_accuracy < 55:
                print("- 準確率太低，建議增加LSTM層數或神經元數量")
            elif direction_accuracy < 65:
                print("- 準確率偏低，建議調整序列長度或添加技術指標")
            else:
                print("- 接近目標，建議微調學習率或增加訓練輪數")


if __name__ == "__main__":
    main()
