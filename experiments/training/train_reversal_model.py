#!/usr/bin/env python3
"""
基於A轉/V轉特徵訓練反轉預測模型
"""

import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib


def prepare_features(df):
    """準備特徵數據"""
    df = df.copy()

    # 技術指標
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(
        df["close"], timeperiod=48, nbdevup=2.4, nbdevdn=2.4
    )
    df["stochrsi_k"], df["stochrsi_d"] = talib.STOCHRSI(
        df["close"], timeperiod=14, fastk_period=3, fastd_period=3
    )
    df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(df["close"])

    # 成交量特徵
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    # 布林位置
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )

    # 價格動量
    df["price_change_3"] = df["close"].pct_change(3)
    df["price_change_5"] = df["close"].pct_change(5)

    return df


def create_reversal_signals(df, lookback=5):
    """創建反轉信號標籤"""
    signals = []

    for i in range(lookback, len(df) - lookback):
        current = df.iloc[i]
        future_prices = df["close"].iloc[i + 1 : i + 1 + lookback].values

        # A轉條件: StochRSI K > 95 + 成交量放大
        if (
            current["stochrsi_k"] > 95
            and current["volume_ratio"] > 1.3
            and pd.notna(current["stochrsi_k"])
        ):

            # 檢查未來是否下跌
            max_decline = (min(future_prices) - current["close"]) / current[
                "close"
            ]
            if max_decline < -0.015:  # 至少1.5%跌幅
                signals.append(
                    {
                        "index": i,
                        "signal": "A_turn",
                        "future_return": max_decline,
                    }
                )
                continue

        # V轉條件: StochRSI K < 5 + 成交量放大
        if (
            current["stochrsi_k"] < 5
            and current["volume_ratio"] > 1.3
            and pd.notna(current["stochrsi_k"])
        ):

            # 檢查未來是否上漲
            max_gain = (max(future_prices) - current["close"]) / current[
                "close"
            ]
            if max_gain > 0.015:  # 至少1.5%漲幅
                signals.append(
                    {"index": i, "signal": "V_turn", "future_return": max_gain}
                )
                continue

        # 無信號
        signals.append({"index": i, "signal": "no_signal", "future_return": 0})

    return pd.DataFrame(signals)


def prepare_training_data(df, signals_df):
    """準備訓練數據"""
    features = []
    labels = []

    feature_cols = [
        "stochrsi_k",
        "stochrsi_d",
        "macd_hist",
        "volume_ratio",
        "bb_position",
        "price_change_3",
        "price_change_5",
    ]

    for _, signal in signals_df.iterrows():
        idx = signal["index"]
        row_data = df.iloc[idx]

        # 特徵向量
        feature_vector = []
        for col in feature_cols:
            val = row_data[col]
            feature_vector.append(val if pd.notna(val) else 0)

        features.append(feature_vector)

        # 標籤 (0: no_signal, 1: A_turn, 2: V_turn)
        if signal["signal"] == "A_turn":
            labels.append(1)
        elif signal["signal"] == "V_turn":
            labels.append(2)
        else:
            labels.append(0)

    return np.array(features), np.array(labels), feature_cols


def train_models(X, y):
    """訓練模型"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 隨機森林
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # 神經網路
    nn_model = MLPClassifier(
        hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42
    )
    nn_model.fit(X_train, y_train)
    nn_pred = nn_model.predict(X_test)

    return rf_model, nn_model, X_test, y_test, rf_pred, nn_pred


def main():
    # 載入數據
    print("載入數據...")
    df = pd.read_csv("training_data/btc_12months_15m.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 準備特徵
    print("計算技術指標...")
    df = prepare_features(df)

    # 創建信號標籤
    print("創建反轉信號...")
    signals_df = create_reversal_signals(df)

    # 統計信號分布
    signal_counts = signals_df["signal"].value_counts()
    print(f"信號分布:")
    print(f"  A轉信號: {signal_counts.get('A_turn', 0)}次")
    print(f"  V轉信號: {signal_counts.get('V_turn', 0)}次")
    print(f"  無信號: {signal_counts.get('no_signal', 0)}次")

    # 準備訓練數據
    print("準備訓練數據...")
    X, y, feature_names = prepare_training_data(df, signals_df)

    # 訓練模型
    print("訓練模型...")
    rf_model, nn_model, X_test, y_test, rf_pred, nn_pred = train_models(X, y)

    # 評估結果
    print("\n=== 隨機森林結果 ===")
    print(f"準確率: {accuracy_score(y_test, rf_pred):.3f}")
    print(
        classification_report(
            y_test, rf_pred, target_names=["無信號", "A轉", "V轉"]
        )
    )

    print("\n=== 神經網路結果 ===")
    print(f"準確率: {accuracy_score(y_test, nn_pred):.3f}")
    print(
        classification_report(
            y_test, nn_pred, target_names=["無信號", "A轉", "V轉"]
        )
    )

    # 特徵重要性
    print("\n=== 特徵重要性 (隨機森林) ===")
    importance = rf_model.feature_importances_
    for name, imp in zip(feature_names, importance):
        print(f"{name}: {imp:.3f}")

    # 保存模型
    joblib.dump(rf_model, "reversal_rf_model.pkl")
    joblib.dump(nn_model, "reversal_nn_model.pkl")

    # 保存特徵名稱
    with open("reversal_features.txt", "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")

    print("\n模型已保存: reversal_rf_model.pkl, reversal_nn_model.pkl")


if __name__ == "__main__":
    main()
