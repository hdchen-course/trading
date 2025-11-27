#!/usr/bin/env python3
"""
準備XGBoost訓練數據
使用crypto_detect.py的PeakValleyDetection找出反轉點標記
"""
import os
import sys
import numpy as np
import pandas as pd

# 添加父目錄到路徑以導入模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa

from crypto_detect import PeakValleyDetection
from crypto_util import calculate_technical_indicators, analyze_cumulative_metrics


def load_data(start_date, end_date):
    """載入指定時間範圍的數據"""
    df = pd.read_csv("../training_data/btc_15m_2022_2025.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
    filtered_df = df[mask].copy().reset_index(drop=True)

    return filtered_df


def prepare_features_and_labels(df):
    """準備特徵和標籤"""
    print(f"Processing {len(df)} records...")

    # 1. 計算基本技術指標
    df_with_indicators = calculate_technical_indicators(df)

    # 2. 用同一個df計算累積指標，直接添加到df中
    df_final = analyze_cumulative_metrics(df_with_indicators)
    print(f"analyze_cumulative_metrics 返回類型: {type(df_final)}")
    print(
        f"df_final columns: {df_final.columns if hasattr(df_final, 'columns') else 'No columns'}")
    print(
        f"df_final shape: {df_final.shape if hasattr(df_final, 'shape') else 'No shape'}")
    print(
        f"df_final head:\n{df_final.head() if hasattr(df_final, 'head') else df_final}")

    # 3. 選擇特徵
    feature_columns = [
        'close', 'volume', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'ema_50', 'ema_100',
        'atr', 'sar', 'stoch_k', 'stoch_d', 'amplitude', 'price_change_pct',
        'volume_sma', 'macd_2nd', 'price_2nd', 'vol_2nd', 'atr_2nd',
        'avg_amplitude', 'avg_volume', 'avg_macd', 'avg_rsi',
        'price_above_sar_ratio', 'avg_bb_position', 'volatility',
        'price_atr_ratio_1h', 'volume_trend'
    ]

    # 選擇存在的特徵
    available_features = [col for col in feature_columns if col in df_final.columns]
    X = df_final[available_features].fillna(0)

    # 使用PeakValleyDetection找反轉點
    detector = PeakValleyDetection(distance=20, height_threshold=0.001)
    reversal_signals = detector.detect(df_final)
    print("print reversal signals")
    print(reversal_signals)
    # 確保標籤長度匹配
    if len(reversal_signals) != len(X):
        print("長度不同")
        min_len = min(len(reversal_signals), len(X))
        X = X.iloc[:min_len]
        reversal_signals = reversal_signals[:min_len]

    y = reversal_signals.astype(int)

    print(f"Features: {X.shape[1]} columns")
    print(f"Feature names: {list(X.columns)}")
    print(f"Positive samples (reversals): {y.sum()}")
    print(f"Negative samples: {len(y) - y.sum()}")
    print(f"Positive ratio: {y.sum() / len(y) * 100:.2f}%")
    df_final["reverse"] = reversal_signals
    
    # 保存包含反轉點標籤的完整數據
    df_final.to_csv("data_with_reversal_labels.csv")
    print(f"已保存包含反轉點標籤的數據到 data_with_reversal_labels.csv")

    return X, y, list(X.columns)


def main():
    # 準備訓練數據 (1-8月)
    print("=== 準備訓練數據 (2025-01-01 to 2025-08-31) ===")
    train_df = load_data("2025-01-01", "2025-08-31")
    X_train, y_train, feature_names = prepare_features_and_labels(train_df)

    # 準備驗證數據 (9-10月)
    print("\n=== 準備驗證數據 (2025-09-01 to 2025-10-31) ===")
    val_df = load_data("2025-09-01", "2025-10-31")
    X_val, y_val, _ = prepare_features_and_labels(val_df)

    # 保存數據
    print("\n=== 保存數據 ===")
    np.save("model_training/lstm/data/X_train.npy", X_train.values)
    np.save("model_training/lstm/data/y_train.npy", y_train)
    np.save("model_training/xgboost/data/X_val.npy", X_val.values)
    np.save("model_training/xgboost/data/y_val.npy", y_val)

    # 保存特徵名稱
    with open("feature_names.txt", "w") as f:
        for feature in feature_names:
            f.write(f"{feature}\n")

    print(f"Training data saved: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Validation data saved: X_val {X_val.shape}, y_val {y_val.shape}")
    print("Feature names saved to feature_names.txt")


if __name__ == "__main__":
    main()
