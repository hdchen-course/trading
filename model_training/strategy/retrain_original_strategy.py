#!/usr/bin/env python3
"""
使用原本策略完整邏輯重新訓練模型
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def collect_original_strategy_patterns(df):
    """收集原本策略的信號點 - 完整邏輯"""
    # 原本策略參數
    period = 54
    std = 2.4
    volume_period = 20
    volume_threshold = 1.25
    pattern_strength = 80

    # 計算布林通道
    bb = ta.bbands(df["close"], length=period, lower_std=std, upper_std=std)
    upper = bb[f"BBU_{period}_{std}_{std}"]
    lower = bb[f"BBL_{period}_{std}_{std}"]

    # 計算成交量比率
    volume_ma = df["volume"].rolling(volume_period).mean()
    volume_ratio = df["volume"] / volume_ma

    # 計算K線型態 - 完全照原本邏輯
    patterns = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"])

    # 優先型態邏輯
    priority_patterns = ["CDL_BELTHOLD", "CDL_CLOSINGMARUBOZU", "CDL_HIKKAKE"]
    priority_mask = patterns.columns.isin(priority_patterns)

    if priority_mask.any():
        # 優先型態：0.8倍門檻
        priority_bearish = (
            patterns.loc[:, priority_mask] <= -pattern_strength * 0.8
        ).any(axis=1)
        priority_bullish = (
            patterns.loc[:, priority_mask] >= pattern_strength * 0.8
        ).any(axis=1)

        # 一般型態：標準門檻
        other_bearish = (
            patterns.loc[:, ~priority_mask] <= -pattern_strength
        ).any(axis=1)
        other_bullish = (
            patterns.loc[:, ~priority_mask] >= pattern_strength
        ).any(axis=1)

        strong_bearish = priority_bearish | other_bearish
        strong_bullish = priority_bullish | other_bullish
    else:
        # 備用邏輯
        strong_bearish = (patterns <= -pattern_strength).any(axis=1)
        strong_bullish = (patterns >= pattern_strength).any(axis=1)

    # 原本策略的信號條件
    buy_conditions = (
        (df["close"] <= lower * 1.005)  # 接近下軌
        & (volume_ratio >= volume_threshold)  # 成交量確認
        & strong_bullish  # 強烈看漲型態
    )

    sell_conditions = (
        (df["close"] >= upper * 0.995)  # 接近上軌
        & (volume_ratio >= volume_threshold)  # 成交量確認
        & strong_bearish  # 強烈看跌型態
    )

    signal_points = buy_conditions | sell_conditions

    collected_patterns = []

    for i in range(len(df)):
        if signal_points.iloc[i] and i >= 10:
            # 提取前10根K線特徵
            features = []
            for j in range(i - 9, i + 1):
                features.extend(
                    [
                        df.iloc[j]["open"],
                        df.iloc[j]["high"],
                        df.iloc[j]["low"],
                        df.iloc[j]["close"],
                        df.iloc[j]["volume"],
                    ]
                )

            # 計算未來價格變化
            if i < len(df) - 1:
                future_price = df.iloc[i + 1]["close"]
                current_price = df.iloc[i]["close"]
                price_change = (future_price - current_price) / current_price

                # 標籤：上漲=1, 下跌=0
                label = 1 if price_change > 0 else 0

                signal_type = (
                    "買入信號" if buy_conditions.iloc[i] else "賣出信號"
                )

                collected_patterns.append(
                    {
                        "features": features,
                        "label": label,
                        "price_change": price_change,
                        "timestamp": df.iloc[i]["timestamp"],
                        "signal_type": signal_type,
                    }
                )

                if len(collected_patterns) % 50 == 0:
                    print(
                        f"已找到 {len(collected_patterns)} 個原本策略信號點..."
                    )

    return collected_patterns


def create_model(input_shape):
    """創建LSTM模型"""
    model = Sequential(
        [
            LSTM(32, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(16, return_sequences=False),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    # 載入數據
    data = pd.read_csv("data/btc_12months_15m.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    print("開始收集原本策略信號點...")
    print("條件：布林帶邊緣(54,2.4) + 成交量1.25倍 + K線型態")
    print("特徵：前10根K線的OHLCV")

    # 收集模式
    patterns = collect_original_strategy_patterns(data)

    if len(patterns) > 20:
        print(f"\n收集到 {len(patterns)} 個信號點")

        # 統計標籤分布
        labels = [p["label"] for p in patterns]
        up_count = sum(labels)
        down_count = len(labels) - up_count

        print(f"上漲標籤: {up_count} 個 ({up_count/len(labels)*100:.1f}%)")
        print(f"下跌標籤: {down_count} 個 ({down_count/len(labels)*100:.1f}%)")

        # 統計信號類型
        buy_signals = sum(
            1 for p in patterns if p["signal_type"] == "買入信號"
        )
        sell_signals = sum(
            1 for p in patterns if p["signal_type"] == "賣出信號"
        )
        print(f"買入信號: {buy_signals} 個")
        print(f"賣出信號: {sell_signals} 個")

        # 準備訓練數據
        X = np.array([p["features"] for p in patterns])
        y = np.array([p["label"] for p in patterns])

        # 重塑特徵
        X = X.reshape(len(patterns), 10, 5)

        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 50)).reshape(
            -1, 10, 5
        )
        X_test_scaled = scaler.transform(X_test.reshape(-1, 50)).reshape(
            -1, 10, 5
        )

        # 創建和訓練模型
        model = create_model((10, 5))

        print(f"\n開始訓練模型...")
        print(f"訓練集: {len(X_train)} 樣本")
        print(f"測試集: {len(X_test)} 樣本")

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
        test_loss, test_accuracy = model.evaluate(
            X_test_scaled, y_test, verbose=0
        )
        print(f"測試準確率: {test_accuracy:.4f}")

        # 保存模型
        if model:
            model.save("model_training/output/strategy/original_strategy_model.keras")
            print("模型已保存為 model/original_strategy_model.keras")
    else:
        print("沒有找到足夠的信號點")
