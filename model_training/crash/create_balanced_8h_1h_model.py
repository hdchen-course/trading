#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

def create_balanced_model():
    """創建平衡的8h+1h crash預測模型"""
    
    # 讀取數據集
    df = pd.read_csv('crash_prediction_dataset_8h_1h.csv')
    
    print("Original dataset:")
    print(f"Total samples: {len(df)}")
    print(f"Positive samples: {len(df[df['is_crash'] == 1])}")
    print(f"Negative samples: {len(df[df['is_crash'] == 0])}")
    print(f"Class ratio: 1:{len(df[df['is_crash'] == 0]) / len(df[df['is_crash'] == 1]):.1f}")
    
    # 準備特徵和標籤
    feature_columns = [col for col in df.columns if col not in ['is_crash', 'month', 'timestamp']]
    X = df[feature_columns]
    y = df['is_crash']
    
    print(f"\nFeatures used: {feature_columns}")
    
    # 處理缺失值
    X = X.fillna(0)
    
    # 使用SMOTE平衡數據
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    print(f"After SMOTE:")
    print(f"Total samples: {len(X_balanced)}")
    print(f"Positive samples: {sum(y_balanced == 1)}")
    print(f"Negative samples: {sum(y_balanced == 0)}")
    
    # 分割訓練和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    # 訓練隨機森林模型
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    
    # 評估模型
    y_pred = rf_model.predict(X_test)
    
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 特徵重要性
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importance:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    # 保存模型和特徵名稱
    joblib.dump(rf_model, 'model_training/output/crash/balanced_crash_prediction_model_8h_1h.pkl')
    joblib.dump(feature_columns, 'model_training/output/crash/balanced_feature_names_8h_1h.pkl')
    
    print(f"\nModel saved to models/crash/balanced_crash_prediction_model_8h_1h.pkl")
    print(f"Feature names saved to models/crash/balanced_feature_names_8h_1h.pkl")
    
    return rf_model, feature_columns

if __name__ == "__main__":
    create_balanced_model()
