#!/usr/bin/env python3
"""
XGBoost 反轉點預測模型訓練
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def load_data():
    """載入準備好的訓練數據"""
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_val = np.load("X_val.npy")
    y_val = np.load("y_val.npy")

    with open("feature_names.txt", "r") as f:
        feature_names = [line.strip() for line in f.readlines()]

    return X_train, y_train, X_val, y_val, feature_names


def train_model(X_train, y_train, X_val, y_val):
    """訓練XGBoost模型"""
    # 處理類別不平衡 - 大幅提高正樣本權重
    pos_ratio = np.sum(y_train) / len(y_train)
    scale_pos_weight = (1 - pos_ratio) / pos_ratio * 1000  # 增加10倍權重
    print(f"scale_weight = {scale_pos_weight}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50  # 減少輸出
    )

    return model


def evaluate_model(model, X_val, y_val, feature_names):
    """評估模型"""
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # 嘗試不同閾值
    thresholds = [0.1, 0.05, 0.01, 0.005]

    print("\n=== 不同閾值下的表現 ===")
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        tp = np.sum((y_pred == 1) & (y_val == 1))
        fp = np.sum((y_pred == 1) & (y_val == 0))
        fn = np.sum((y_pred == 0) & (y_val == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0

        print(
            f"閾值 {threshold}: 精確率={precision:.3f}, 召回率={recall:.3f}, F1={f1:.3f}, 預測反轉點={np.sum(y_pred)}")

    # 特徵重要性
    importance = model.feature_importances_
    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print("\n=== Top 10 重要特徵 ===")
    for name, imp in feature_importance[:10]:
        print(f"{name}: {imp:.4f}")


def main():
    print("=== 載入數據 ===")
    X_train, y_train, X_val, y_val, feature_names = load_data()
    print(f"訓練集: {X_train.shape}, 驗證集: {X_val.shape}")

    print("\n=== 訓練模型 ===")
    model = train_model(X_train, y_train, X_val, y_val)

    print("\n=== 評估模型 ===")
    evaluate_model(model, X_val, y_val, feature_names)

    print("\n=== 保存模型 ===")
    joblib.dump(model, "model_training/output/xgboost/xgboost_reversal_model.pkl")
    print("模型已保存為 xgboost_reversal_model.pkl")


if __name__ == "__main__":
    main()
