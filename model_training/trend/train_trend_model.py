import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def identify_trend_by_slope(df, window=10):
    """識別趨勢狀態"""
    df = df.copy()
    df["ma_short"] = df["close"].rolling(window=window).mean()
    df["slope"] = df["ma_short"].diff(5) / df["ma_short"].shift(5)

    trend_threshold = 0.001
    df["trend"] = 0  # 橫盤
    df.loc[df["slope"] > trend_threshold, "trend"] = 1  # 上漲
    df.loc[df["slope"] < -trend_threshold, "trend"] = -1  # 下跌

    return df


def calculate_features(df, window=14):
    """計算技術指標特徵"""
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

    # 價格動量
    df["price_momentum_5"] = df["close"].pct_change(5)
    df["price_momentum_10"] = df["close"].pct_change(10)
    df["price_momentum_20"] = df["close"].pct_change(20)

    # 波動率
    df["volatility_10"] = (
        df["close"].rolling(10).std() / df["close"].rolling(10).mean()
    )
    df["volatility_20"] = (
        df["close"].rolling(20).std() / df["close"].rolling(20).mean()
    )

    # ATR
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = np.abs(df["high"] - df["close"].shift())
    df["low_close"] = np.abs(df["low"] - df["close"].shift())
    df["true_range"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    df["atr"] = df["true_range"].rolling(window=14).mean()
    df["atr_ratio"] = df["atr"] / df["close"]

    return df


# 讀取數據
print("📊 讀取BTC數據...")
df = pd.read_csv("training_data/btc_12months_15m.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

print(f"原始數據: {len(df)} 行")

# 識別趨勢
print("🔍 識別趨勢狀態...")
df = identify_trend_by_slope(df)

# 計算技術指標
print("🔧 計算技術指標...")
df = calculate_features(df)

# 準備訓練數據
feature_columns = [
    "rsi",
    "macd",
    "macd_hist",
    "bb_position",
    "volume_ratio",
    "price_momentum_5",
    "price_momentum_10",
    "price_momentum_20",
    "volatility_10",
    "volatility_20",
    "atr_ratio",
]

# 移除缺失值
clean_df = df.dropna(subset=feature_columns + ["trend"])
print(f"清理後數據: {len(clean_df)} 行")

# 統計趨勢分布
trend_counts = clean_df["trend"].value_counts()
print(f"\n📊 趨勢分布:")
print(
    f"上漲 (1): {trend_counts.get(1, 0)} 個 ({trend_counts.get(1, 0)/len(clean_df)*100:.1f}%)"
)
print(
    f"下跌 (-1): {trend_counts.get(-1, 0)} 個 ({trend_counts.get(-1, 0)/len(clean_df)*100:.1f}%)"
)
print(
    f"橫盤 (0): {trend_counts.get(0, 0)} 個 ({trend_counts.get(0, 0)/len(clean_df)*100:.1f}%)"
)

# 準備特徵和標籤
X = clean_df[feature_columns].values
y = clean_df["trend"].values

# 將標籤轉換為0,1,2 (方便分類)
y_mapped = np.where(
    y == -1, 0, np.where(y == 0, 1, 2)
)  # 下跌=0, 橫盤=1, 上漲=2

# 分割數據
X_train, X_test, y_train, y_test = train_test_split(
    X, y_mapped, test_size=0.3, random_state=42, stratify=y_mapped
)

print(f"\n🎯 訓練數據:")
print(f"訓練集: {len(X_train)} 個")
print(f"測試集: {len(X_test)} 個")

# 訓練隨機森林模型
print("🤖 訓練趨勢預測模型...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    max_depth=15,
    min_samples_split=10,
)
rf_model.fit(X_train, y_train)

# 預測和評估
y_pred = rf_model.predict(X_test)
accuracy = rf_model.score(X_test, y_test)

print(f"\n📈 模型性能:")
print(f"準確率: {accuracy:.3f}")

print(f"\n分類報告:")
target_names = ["下跌", "橫盤", "上漲"]
print(classification_report(y_test, y_pred, target_names=target_names))

print(f"\n混淆矩陣:")
cm = confusion_matrix(y_test, y_pred)
print("實際\\預測  下跌   橫盤   上漲")
for i, (actual, row) in enumerate(zip(target_names, cm)):
    print(f"{actual:>6}  {row[0]:>6} {row[1]:>6} {row[2]:>6}")

# 特徵重要性
feature_importance = pd.DataFrame(
    {"feature": feature_columns, "importance": rf_model.feature_importances_}
).sort_values("importance", ascending=False)

print(f"\n🎯 特徵重要性:")
for _, row in feature_importance.head(8).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# 保存模型
joblib.dump(rf_model, "model_training/output/trend/trend_prediction_model.pkl")
print(f"\n✅ 模型已保存: trend_prediction_model.pkl")

# 保存特徵數據
feature_data = clean_df[
    feature_columns + ["trend", "timestamp", "close"]
].copy()
feature_data.to_csv("trend_features.csv", index=False)
print(f"✅ 特徵數據已保存: trend_features.csv")

# 測試預測
print(f"\n🔮 最新數據預測測試:")
latest_features = X[-5:]  # 最後5個數據點
latest_predictions = rf_model.predict(latest_features)
latest_probabilities = rf_model.predict_proba(latest_features)

trend_names = {0: "下跌", 1: "橫盤", 2: "上漲"}
for i, (pred, prob) in enumerate(
    zip(latest_predictions, latest_probabilities)
):
    max_prob = max(prob)
    print(f"  數據點 {i+1}: 預測 {trend_names[pred]} (信心度: {max_prob:.2%})")
