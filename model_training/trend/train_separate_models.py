import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# 讀取特徵數據
print("📊 讀取特徵數據...")
df = pd.read_csv("trend_features.csv")

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

# 準備上漲vs非上漲模型
print("🔺 訓練上漲預測模型...")
uptrend_data = df.copy()
uptrend_data["is_uptrend"] = (uptrend_data["trend"] == 1).astype(int)

X_up = uptrend_data[feature_columns].values
y_up = uptrend_data["is_uptrend"].values

X_up_train, X_up_test, y_up_train, y_up_test = train_test_split(
    X_up, y_up, test_size=0.3, random_state=42, stratify=y_up
)

uptrend_model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
uptrend_model.fit(X_up_train, y_up_train)

up_accuracy = uptrend_model.score(X_up_test, y_up_test)
print(f"上漲模型準確率: {up_accuracy:.3f}")

# 準備下跌vs非下跌模型
print("🔻 訓練下跌預測模型...")
downtrend_data = df.copy()
downtrend_data["is_downtrend"] = (downtrend_data["trend"] == -1).astype(int)

X_down = downtrend_data[feature_columns].values
y_down = downtrend_data["is_downtrend"].values

X_down_train, X_down_test, y_down_train, y_down_test = train_test_split(
    X_down, y_down, test_size=0.3, random_state=42, stratify=y_down
)

downtrend_model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
downtrend_model.fit(X_down_train, y_down_train)

down_accuracy = downtrend_model.score(X_down_test, y_down_test)
print(f"下跌模型準確率: {down_accuracy:.3f}")

# 測試組合預測
print(f"\n🔮 組合模型測試:")
latest_features = X_up[-5:]

up_predictions = uptrend_model.predict_proba(latest_features)[:, 1]  # 上漲概率
down_predictions = downtrend_model.predict_proba(latest_features)[
    :, 1
]  # 下跌概率

for i, (up_prob, down_prob) in enumerate(
    zip(up_predictions, down_predictions)
):
    sideways_prob = 1 - up_prob - down_prob

    if up_prob > 0.6:
        prediction = "上漲"
        confidence = up_prob
    elif down_prob > 0.6:
        prediction = "下跌"
        confidence = down_prob
    else:
        prediction = "橫盤"
        confidence = max(sideways_prob, 0)

    print(
        f"  數據點 {i+1}: {prediction} (上漲:{up_prob:.2%}, 下跌:{down_prob:.2%})"
    )

# 保存分離模型
joblib.dump(uptrend_model, "model_training/output/trend/uptrend_model.pkl")
joblib.dump(downtrend_model, "model_training/output/trend/downtrend_model.pkl")
print(f"\n✅ 分離模型已保存: uptrend_model.pkl, downtrend_model.pkl")

print(f"\n📊 模型對比:")
print(f"單一三狀態模型: 87.5%")
print(f"上漲專門模型: {up_accuracy:.1%}")
print(f"下跌專門模型: {down_accuracy:.1%}")
print(
    f"\n💡 建議: {'使用分離模型' if (up_accuracy + down_accuracy)/2 > 0.875 else '使用單一模型'}"
)
