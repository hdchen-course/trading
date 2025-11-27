import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def calculate_technical_indicators(df, window=14):
    """計算技術指標"""
    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["close"].ewm(span=12).mean()
    exp2 = df["close"].ewm(span=26).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 布林帶
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
    df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )

    # 成交量指標
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    return df


def extract_features_around_reversals(df, reversals_df, lookback=10):
    """提取轉折點周圍的特徵"""
    features = []

    for _, reversal in reversals_df.iterrows():
        # 找到對應的數據索引
        reversal_time = pd.to_datetime(reversal["timestamp"])
        matching_rows = df[df["timestamp"] == reversal_time]

        if len(matching_rows) == 0:
            continue

        idx = matching_rows.index[0]

        # 確保有足夠的前後數據
        if idx < lookback or idx >= len(df) - lookback:
            continue

        # 提取特徵
        feature_row = {
            "timestamp": reversal_time,
            "price": reversal["price"],
            "type": reversal["type"],
            "change": reversal["change"],
        }

        # 當前時點的技術指標
        current_data = df.iloc[idx]
        feature_row.update(
            {
                "rsi": current_data["rsi"],
                "macd": current_data["macd"],
                "macd_hist": current_data["macd_hist"],
                "bb_position": current_data["bb_position"],
                "volume_ratio": current_data["volume_ratio"],
            }
        )

        # 價格動量特徵
        before_prices = df.iloc[idx - lookback : idx]["close"].values
        after_prices = df.iloc[idx + 1 : idx + lookback + 1]["close"].values

        if len(before_prices) >= lookback and len(after_prices) >= lookback:
            feature_row.update(
                {
                    "price_momentum_5": (
                        current_data["close"] - df.iloc[idx - 5]["close"]
                    )
                    / df.iloc[idx - 5]["close"],
                    "price_momentum_10": (
                        current_data["close"] - df.iloc[idx - 10]["close"]
                    )
                    / df.iloc[idx - 10]["close"],
                    "volatility_10": df.iloc[idx - 10 : idx]["close"].std()
                    / df.iloc[idx - 10 : idx]["close"].mean(),
                    "volume_trend": np.polyfit(
                        range(10), df.iloc[idx - 10 : idx]["volume"].values, 1
                    )[0],
                }
            )

            features.append(feature_row)

    return pd.DataFrame(features)


# 讀取數據
print("📊 讀取數據...")
df = pd.read_csv("training_data/btc_12months_15m.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

reversals_df = pd.read_csv("btc_reversals_final.csv")
reversals_df["timestamp"] = pd.to_datetime(reversals_df["timestamp"])

print(f"原始數據: {len(df)} 行")
print(f"轉折點: {len(reversals_df)} 個")

# 計算技術指標
print("🔧 計算技術指標...")
df = calculate_technical_indicators(df)

# 提取轉折點特徵
print("🎯 提取轉折點特徵...")
reversal_features = extract_features_around_reversals(df, reversals_df)

print(f"成功提取特徵: {len(reversal_features)} 個轉折點")

# 創建標籤
reversal_features["label"] = reversal_features["type"].map(
    {"A_turn": 1, "V_turn": 2}
)

# 準備訓練數據
feature_columns = [
    "rsi",
    "macd",
    "macd_hist",
    "bb_position",
    "volume_ratio",
    "price_momentum_5",
    "price_momentum_10",
    "volatility_10",
    "volume_trend",
]

# 移除缺失值
clean_features = reversal_features.dropna(subset=feature_columns + ["label"])
print(f"清理後特徵: {len(clean_features)} 個")

if len(clean_features) < 20:
    print("❌ 特徵數量不足，無法訓練模型")
    exit()

X = clean_features[feature_columns].values
y = clean_features["label"].values

# 分割數據
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"訓練集: {len(X_train)} 個")
print(f"測試集: {len(X_test)} 個")
print(f"A轉: {sum(y == 1)} 個")
print(f"V轉: {sum(y == 2)} 個")

# 訓練隨機森林模型
print("🤖 訓練隨機森林模型...")
rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced"
)
rf_model.fit(X_train, y_train)

# 預測和評估
y_pred = rf_model.predict(X_test)
accuracy = rf_model.score(X_test, y_test)

print(f"\n📈 模型性能:")
print(f"準確率: {accuracy:.3f}")
print(f"\n分類報告:")
print(classification_report(y_test, y_pred, target_names=["A_turn", "V_turn"]))

# 特徵重要性
feature_importance = pd.DataFrame(
    {"feature": feature_columns, "importance": rf_model.feature_importances_}
).sort_values("importance", ascending=False)

print(f"\n🎯 特徵重要性:")
for _, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# 保存模型和特徵
joblib.dump(rf_model, "model_training/output/reversal/reversal_model_v2.pkl")
clean_features.to_csv("reversal_features_v2.csv", index=False)

print(f"\n✅ 模型已保存: reversal_model_v2.pkl")
print(f"✅ 特徵已保存: reversal_features_v2.csv")
