#!/usr/bin/env python3
"""
使用優化參數和反轉型態重新訓練模型
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


def collect_optimized_patterns(df):
    """收集優化參數的信號點"""
    # 優化參數 - 使用原始策略參數
    period = 54
    std = 2.4
    volume_period = 20
    volume_threshold = 1.2
    pattern_strength = 85

    # 最適合反轉的K線型態 (3-5個)
    reversal_patterns = [
        "CDL_HAMMER",  # 錘子線 - 底部反轉
        "CDL_SHOOTINGSTAR",  # 流星線 - 頂部反轉
        "CDL_DOJI",  # 十字星 - 反轉信號
        "CDL_ENGULFING",  # 吞噬型態 - 強烈反轉
        "CDL_HARAMI",  # 孕線 - 反轉信號
    ]

    # 計算布林通道
    bb = ta.bbands(df["close"], length=period, lower_std=std, upper_std=std)
    upper = bb[f"BBU_{period}_{std}_{std}"]
    lower = bb[f"BBL_{period}_{std}_{std}"]

    # 計算成交量比率
    volume_ma = df["volume"].rolling(volume_period).mean()
    volume_ratio = df["volume"] / volume_ma

    # 計算K線型態
    patterns = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"])

    # 篩選反轉型態
    if not patterns.empty:
        available_patterns = [
            p for p in reversal_patterns if p in patterns.columns
        ]
        if available_patterns:
            reversal_signals = patterns[available_patterns]

            # 強烈反轉信號 (強度超過85)
            strong_reversal = (reversal_signals >= pattern_strength).any(
                axis=1
            ) | (reversal_signals <= -pattern_strength).any(axis=1)
        else:
            strong_reversal = pd.Series(False, index=df.index)
    else:
        strong_reversal = pd.Series(False, index=df.index)

    # 信號條件：布林帶邊緣 + 成交量 + 反轉型態
    at_upper = df["close"] >= upper * 0.995
    at_lower = df["close"] <= lower * 1.005
    high_volume = volume_ratio >= volume_threshold

    signal_points = (at_upper | at_lower) & high_volume & strong_reversal

    collected_patterns = []

    for i in range(10, len(df) - 10):  # 前後各10個K線
        if signal_points.iloc[i]:
            # 提取前後各10根K線特徵 (共21根)
            features = []
            for j in range(i - 10, i + 11):  # i-10 到 i+10 (包含當前)
                features.extend(
                    [
                        df.iloc[j]["open"],
                        df.iloc[j]["high"],
                        df.iloc[j]["low"],
                        df.iloc[j]["close"],
                        df.iloc[j]["volume"],
                    ]
                )

            # 計算未來價格變化 (下一根K線)
            if i < len(df) - 11:
                future_price = df.iloc[i + 11]["close"]  # 當前K線後的第11根
                current_price = df.iloc[i]["close"]
                price_change = (future_price - current_price) / current_price

                # 標籤：上漲=1, 下跌=0
                label = 1 if price_change > 0 else 0

                signal_type = "上軌反轉" if at_upper.iloc[i] else "下軌反轉"

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
                    print(f"已找到 {len(collected_patterns)} 個反轉信號點...")

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

    print("開始收集優化反轉信號點...")
    print("條件：布林帶邊緣(54,2.4) + 成交量1.2倍 + 反轉型態(強度>85)")
    print("特徵：前後各10根K線的OHLCV (共21根)")

    # 收集模式
    patterns = collect_optimized_patterns(data)

    if len(patterns) > 20:
        print(f"\n收集到 {len(patterns)} 個反轉信號點")

        # 統計標籤分布
        labels = [p["label"] for p in patterns]
        up_count = sum(labels)
        down_count = len(labels) - up_count

        print(f"上漲標籤: {up_count} 個 ({up_count/len(labels)*100:.1f}%)")
        print(f"下跌標籤: {down_count} 個 ({down_count/len(labels)*100:.1f}%)")

        # 統計信號類型
        upper_signals = sum(
            1 for p in patterns if p["signal_type"] == "上軌反轉"
        )
        lower_signals = sum(
            1 for p in patterns if p["signal_type"] == "下軌反轉"
        )
        print(f"上軌反轉: {upper_signals} 個")
        print(f"下軌反轉: {lower_signals} 個")

        # 準備訓練數據
        X = np.array([p["features"] for p in patterns])
        y = np.array([p["label"] for p in patterns])

        # 重塑特徵 (21根K線 x 5個特徵)
        X = X.reshape(len(patterns), 21, 5)

        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(
            X_train.reshape(-1, 105)
        ).reshape(-1, 21, 5)
        X_test_scaled = scaler.transform(X_test.reshape(-1, 105)).reshape(
            -1, 21, 5
        )

        # 創建和訓練模型
        model = create_model((21, 5))

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
            model.save("model_training/output/strategy/optimized_reversal_model.keras")
            print("模型已保存為 model/optimized_reversal_model.keras")
    else:
        print("沒有找到足夠的反轉信號點")
