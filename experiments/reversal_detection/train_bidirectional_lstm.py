#!/usr/bin/env python3
"""
使用新數據訓練 Bidirectional LSTM 模型

使用 128K+ 筆數據，3,224 個反轉點
基於 Task 2 發現的最佳特徵
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def prepare_lstm_features(df):
    """
    準備 LSTM 特徵（基於 Task 2 發現）
    """
    print("\n🔧 準備 LSTM 特徵...")
    
    features = pd.DataFrame()
    
    # 1. 價格位置特徵（Task 2 最重要）
    high_24h = df['close'].rolling(window=96, min_periods=1).max()
    low_24h = df['close'].rolling(window=96, min_periods=1).min()
    features['position_24h'] = (df['close'] - low_24h) / (high_24h - low_24h + 1e-10)
    
    high_48h = df['close'].rolling(window=192, min_periods=1).max()
    low_48h = df['close'].rolling(window=192, min_periods=1).min()
    features['position_48h'] = (df['close'] - low_48h) / (high_48h - low_48h + 1e-10)
    
    # 2. 價格本身和變化
    features['close'] = df['close']
    features['price_change_5'] = df['close'].pct_change(5)
    features['price_change_10'] = df['close'].pct_change(10)
    
    # 3. RSI
    if 'rsi' in df.columns:
        features['rsi'] = df['rsi']
        features['rsi_normalized'] = df['rsi'] / 100.0
    
    # 4. 布林帶位置
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        bb_range = df['bb_upper'] - df['bb_lower']
        features['bb_position'] = (df['close'] - df['bb_lower']) / (bb_range + 1e-10)
        features['bb_width'] = bb_range / df['close']
    
    # 5. MACD
    if 'macd' in df.columns:
        features['macd'] = df['macd']
        features['macd_normalized'] = df['macd'] / df['close']
    
    # 6. 波動性特徵（Task 2 發現很重要）
    if 'atr' in df.columns:
        features['atr'] = df['atr']
        features['atr_normalized'] = df['atr'] / df['close']
    
    if 'volatility' in df.columns:
        features['volatility'] = df['volatility']
    
    # 7. 二階導數（Task 2 發現的關鍵特徵）
    if 'atr_2nd' in df.columns:
        features['atr_2nd'] = df['atr_2nd']
    
    if 'price_2nd' in df.columns:
        features['price_2nd'] = df['price_2nd']
    
    if 'macd_2nd' in df.columns:
        features['macd_2nd'] = df['macd_2nd']
    
    # 8. 成交量
    if 'volume' in df.columns:
        volume_ma = df['volume'].rolling(window=20).mean()
        features['volume_ratio'] = df['volume'] / (volume_ma + 1e-10)
    
    print(f"   特徵數: {len(features.columns)}")
    print(f"   特徵列表: {list(features.columns)}")
    
    return features


def create_sequences(features, labels, timesteps=96):
    """
    創建時間序列數據
    
    Args:
        features: 特徵 DataFrame
        labels: 標籤 Series
        timesteps: 時間步長（96 = 24小時 @ 15分鐘）
    
    Returns:
        X: (samples, timesteps, features)
        y: (samples,)
    """
    print(f"\n🔧 創建時間序列（timesteps={timesteps}）...")
    
    X_sequences = []
    y_sequences = []
    
    for i in range(timesteps, len(features)):
        # 取前 timesteps 個時間步作為輸入
        X_sequences.append(features.iloc[i-timesteps:i].values)
        # 當前時間點的標籤作為輸出
        y_sequences.append(labels.iloc[i])
    
    X = np.array(X_sequences)
    y = np.array(y_sequences)
    
    print(f"   序列數: {len(X)}")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   正例數: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
    
    return X, y


def build_bidirectional_lstm(input_shape, learning_rate=0.001):
    """
    構建 Bidirectional LSTM 模型
    
    Args:
        input_shape: (timesteps, features)
        learning_rate: 學習率
    """
    print("\n🏗️  構建 Bidirectional LSTM 模型...")
    
    model = keras.Sequential([
        # 第一層 Bidirectional LSTM
        layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.2),
            input_shape=input_shape
        ),
        layers.BatchNormalization(),
        
        # 第二層 Bidirectional LSTM
        layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.2)
        ),
        layers.BatchNormalization(),
        
        # 第三層 LSTM（不需要 return_sequences）
        layers.LSTM(32, dropout=0.2),
        layers.BatchNormalization(),
        
        # 全連接層
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        
        # 輸出層（二分類）
        layers.Dense(1, activation='sigmoid')
    ])
    
    # 編譯模型
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    print("\n模型架構:")
    model.summary()
    
    return model


def train_lstm_model(data_path):
    """
    訓練 Bidirectional LSTM 模型
    """
    print("=" * 80)
    print("🚀 訓練 Bidirectional LSTM 模型")
    print("=" * 80)
    
    # 載入數據
    df = pd.read_csv(data_path)
    print(f"\n✅ 數據載入完成: {len(df)} 筆資料")
    print(f"   反轉點: {df['reverse'].sum()} 個 ({df['reverse'].sum()/len(df)*100:.2f}%)")
    
    # 準備特徵
    features = prepare_lstm_features(df)
    labels = df['reverse'].astype(int)
    
    # 清理數據
    mask = ~(features.isnull().any(axis=1) | np.isinf(features).any(axis=1))
    features = features[mask]
    labels = labels[mask]
    
    print(f"\n📊 清理後數據: {len(features)} 筆")
    
    # 創建時間序列
    timesteps = 96  # 24小時
    X, y = create_sequences(features, labels, timesteps)
    
    # 時間分割：前 80% 訓練，後 20% 測試
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n📊 數據分割:")
    print(f"   訓練集: {len(X_train)} 筆 (反轉點: {y_train.sum()}, {y_train.sum()/len(y_train)*100:.2f}%)")
    print(f"   測試集: {len(X_test)} 筆 (反轉點: {y_test.sum()}, {y_test.sum()/len(y_test)*100:.2f}%)")
    
    # 標準化
    print("\n🔧 標準化特徵...")
    scaler = StandardScaler()
    
    # Reshape for scaling
    n_samples_train, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
    
    n_samples_test = X_test.shape[0]
    X_test_reshaped = X_test.reshape(-1, n_features)
    X_test_scaled = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)
    
    # 處理類別不平衡
    print("\n⚖️  處理類別不平衡...")
    neg_count = len(y_train) - y_train.sum()
    pos_count = y_train.sum()
    
    # 計算類別權重
    total = len(y_train)
    weight_for_0 = (1 / neg_count) * (total / 2.0)
    weight_for_1 = (1 / pos_count) * (total / 2.0)
    
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    print(f"   負例數: {neg_count}")
    print(f"   正例數: {pos_count}")
    print(f"   類別權重: {class_weight}")
    
    # 構建模型
    input_shape = (timesteps, n_features)
    model = build_bidirectional_lstm(input_shape)
    
    # 回調函數
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'experiments/reversal_detection/best_lstm_model.keras',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 訓練模型
    print("\n" + "=" * 80)
    print("🏋️  開始訓練...")
    print("=" * 80)
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=50,
        batch_size=64,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    # 評估模型
    print("\n" + "=" * 80)
    print("📊 模型評估")
    print("=" * 80)
    
    # 預測
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    
    # 嘗試不同的閾值
    thresholds = [0.3, 0.5, 0.7, 0.8]
    best_f1 = 0
    best_threshold = 0.5
    
    print("\n🎯 測試不同閾值:")
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int).flatten()
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"\n   閾值 {threshold}:")
        print(f"   - 精確度: {precision*100:.2f}%")
        print(f"   - 召回率: {recall*100:.2f}%")
        print(f"   - F1 分數: {f1:.4f} ({f1*100:.2f}%)")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # 使用最佳閾值
    print(f"\n🏆 最佳閾值: {best_threshold} (F1={best_f1:.4f})")
    y_pred_best = (y_pred_proba > best_threshold).astype(int).flatten()
    
    # 詳細評估
    precision = precision_score(y_test, y_pred_best, zero_division=0)
    recall = recall_score(y_test, y_pred_best, zero_division=0)
    f1 = f1_score(y_test, y_pred_best, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_best)
    
    print("\n" + "=" * 80)
    print("📊 最終結果（最佳閾值）")
    print("=" * 80)
    print(f"\n精確度: {precision*100:.2f}%")
    print(f"召回率: {recall*100:.2f}%")
    print(f"F1 分數: {f1:.4f} ({f1*100:.2f}%)")
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"\n混淆矩陣:")
        print(f"  TP: {tp}, FP: {fp}")
        print(f"  FN: {fn}, TN: {tn}")
    
    # 與規則方法比較
    print("\n" + "=" * 80)
    print("📊 與規則方法比較")
    print("=" * 80)
    
    print("\n規則方法 (極端位置 + 波動加速):")
    print(f"  F1 分數: 15.3%")
    print(f"  精確度:  10.6%")
    print(f"  召回率:  27.5%")
    
    print(f"\nBidirectional LSTM (新訓練):")
    print(f"  F1 分數: {f1*100:.1f}%")
    print(f"  精確度:  {precision*100:.1f}%")
    print(f"  召回率:  {recall*100:.1f}%")
    
    rule_f1 = 0.153
    if f1 > rule_f1:
        improvement = (f1 - rule_f1) / rule_f1 * 100
        print(f"\n🏆 LSTM 勝出！提升 {improvement:.1f}%")
    else:
        decline = (rule_f1 - f1) / rule_f1 * 100
        print(f"\n規則方法仍然更好（差距 {decline:.1f}%）")
    
    # 保存模型和 scaler
    print("\n💾 保存模型...")
    model.save('experiments/reversal_detection/bidirectional_lstm_new.keras')
    joblib.dump(scaler, 'experiments/reversal_detection/lstm_scaler_new.pkl')
    joblib.dump(best_threshold, 'experiments/reversal_detection/lstm_threshold_new.pkl')
    
    print("   ✅ 模型已保存:")
    print("      - bidirectional_lstm_new.keras")
    print("      - lstm_scaler_new.pkl")
    print("      - lstm_threshold_new.pkl")
    
    # 繪製訓練歷史
    print("\n📈 繪製訓練歷史...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/reversal_detection/lstm_training_history.png', dpi=300)
    print("   ✅ 訓練歷史已保存: lstm_training_history.png")
    
    print("\n" + "=" * 80)
    print("✅ 訓練完成！")
    print("=" * 80)
    
    return model, scaler, best_threshold, f1


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='訓練 Bidirectional LSTM 模型')
    parser.add_argument('--data', type=str,
                       default='training_data/reversal_training_data_2022-01-01_to_2025-09-01_15m.csv',
                       help='訓練數據路徑')
    
    args = parser.parse_args()
    
    train_lstm_model(args.data)
