#!/usr/bin/env python3
"""
基於布林帶+StochRSI的機器學習模型訓練
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib


class BBStochRSIModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.rf_model = None
        self.nn_model = None

    def prepare_features(self, df):
        """準備特徵數據"""
        # 核心特徵：布林帶 + StochRSI
        feature_cols = [
            "bb_position",
            "bb_width",
            "close_bb_middle_ratio",
            "stochrsi_k",
            "stochrsi_d",
            "volatility_48",
            "volatility_24",
            "volatility_12",
            "rsi_14",
            "close_sma5_ratio",
            "close_sma10_ratio",
            "volume_ratio",
            "high_low_ratio",
        ]

        # 移除缺失值
        df_clean = df.dropna(
            subset=feature_cols
            + ["stochrsi_long_signal", "stochrsi_short_signal"]
        )

        X = df_clean[feature_cols]

        # 創建三分類標籤：0=無信號, 1=做多, 2=做空
        y = np.zeros(len(df_clean))
        y[df_clean["stochrsi_long_signal"] == 1] = 1
        y[df_clean["stochrsi_short_signal"] == 1] = 2

        return X, y, df_clean

    def train_random_forest(self, X, y):
        """訓練隨機森林模型"""
        print("訓練隨機森林模型...")

        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 訓練模型
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        )

        self.rf_model.fit(X_train, y_train)

        # 評估
        train_score = self.rf_model.score(X_train, y_train)
        test_score = self.rf_model.score(X_test, y_test)

        print(f"隨機森林準確度:")
        print(f"  訓練集: {train_score:.4f}")
        print(f"  測試集: {test_score:.4f}")

        # 交叉驗證
        cv_scores = cross_val_score(self.rf_model, X, y, cv=5)
        print(f"  交叉驗證: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # 特徵重要性
        feature_importance = pd.DataFrame(
            {
                "feature": X.columns,
                "importance": self.rf_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print(f"\n特徵重要性 (前8名):")
        for _, row in feature_importance.head(8).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # 詳細分類報告
        y_pred = self.rf_model.predict(X_test)
        print(f"\n分類報告:")
        print(
            classification_report(
                y_test, y_pred, target_names=["無信號", "做多", "做空"]
            )
        )

        return X_test, y_test, y_pred

    def train_neural_network(self, X, y):
        """訓練神經網絡模型"""
        print("\n訓練神經網絡模型...")

        # 標準化特徵
        X_scaled = self.scaler.fit_transform(X)

        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # 轉換為one-hot編碼
        y_train_cat = tf.keras.utils.to_categorical(y_train, 3)
        y_test_cat = tf.keras.utils.to_categorical(y_test, 3)

        # 構建模型
        self.nn_model = Sequential(
            [
                Dense(64, activation="relu", input_shape=(X.shape[1],)),
                Dropout(0.3),
                Dense(32, activation="relu"),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(3, activation="softmax"),
            ]
        )

        self.nn_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # 訓練
        history = self.nn_model.fit(
            X_train,
            y_train_cat,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test_cat),
            verbose=0,
        )

        # 評估
        train_loss, train_acc = self.nn_model.evaluate(
            X_train, y_train_cat, verbose=0
        )
        test_loss, test_acc = self.nn_model.evaluate(
            X_test, y_test_cat, verbose=0
        )

        print(f"神經網絡準確度:")
        print(f"  訓練集: {train_acc:.4f}")
        print(f"  測試集: {test_acc:.4f}")

        return history

    def analyze_signal_performance(self, df_clean, X_test, y_test, y_pred):
        """分析信號表現"""
        print(f"\n=== 信號表現分析 ===")

        # 統計各類信號數量
        total_samples = len(df_clean)
        long_signals = (df_clean["stochrsi_long_signal"] == 1).sum()
        short_signals = (df_clean["stochrsi_short_signal"] == 1).sum()

        print(f"總樣本: {total_samples}")
        print(
            f"做多信號: {long_signals} ({long_signals/total_samples*100:.2f}%)"
        )
        print(
            f"做空信號: {short_signals} ({short_signals/total_samples*100:.2f}%)"
        )

        # 分析預測準確度
        long_mask = y_test == 1
        short_mask = y_test == 2

        if long_mask.sum() > 0:
            long_accuracy = (y_pred[long_mask] == 1).mean()
            print(f"做多信號預測準確度: {long_accuracy:.4f}")

        if short_mask.sum() > 0:
            short_accuracy = (y_pred[short_mask] == 2).mean()
            print(f"做空信號預測準確度: {short_accuracy:.4f}")

        # 混淆矩陣
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n混淆矩陣:")
        print(f"實際\\預測  無信號  做多  做空")
        for i, row_name in enumerate(["無信號", "做多", "做空"]):
            print(f"{row_name:>6}  {cm[i][0]:>6} {cm[i][1]:>4} {cm[i][2]:>4}")

    def save_models(self):
        """保存模型"""
        if self.rf_model:
            joblib.dump(self.rf_model, "model/bb_stochrsi_rf_model.pkl")
            print("隨機森林模型已保存到 model/bb_stochrsi_rf_model.pkl")

        if self.nn_model:
            self.nn_model.save("model/bb_stochrsi_nn_model.keras")
            joblib.dump(self.scaler, "model/bb_stochrsi_scaler.pkl")
            print("神經網絡模型已保存到 model/bb_stochrsi_nn_model.keras")


def main():
    # 載入數據
    print("載入布林帶+StochRSI數據...")
    df = pd.read_csv("training_data/btc_rsi_stochrsi_comparison.csv")

    # 初始化模型
    model = BBStochRSIModel()

    # 準備特徵
    X, y, df_clean = model.prepare_features(df)
    print(f"特徵維度: {X.shape}")
    print(
        f"標籤分布: 無信號={np.sum(y==0)}, 做多={np.sum(y==1)}, 做空={np.sum(y==2)}"
    )

    # 訓練隨機森林
    X_test, y_test, y_pred = model.train_random_forest(X, y)

    # 訓練神經網絡
    history = model.train_neural_network(X, y)

    # 分析表現
    model.analyze_signal_performance(df_clean, X_test, y_test, y_pred)

    # 保存模型
    model.save_models()

    return model


if __name__ == "__main__":
    trained_model = main()
