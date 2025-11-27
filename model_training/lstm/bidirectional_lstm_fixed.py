#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def build_bidirectional_lstm(input_shape):
    """構建雙向LSTM模型"""
    model = Sequential([
        Bidirectional(LSTM(32, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(16, return_sequences=False)),
        Dropout(0.3),
        Dense(8, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # 使用更小的學習率和梯度裁剪
    model.compile(
        optimizer=Adam(learning_rate=0.0001, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_bidirectional_lstm():
    """訓練雙向LSTM模型"""
    
    print("載入訓練數據...")
    X = np.load('X_train.npy')
    y = np.load('y_train.npy')
    
    print(f"數據形狀: X={X.shape}, y={y.shape}")
    
    # 檢查數據中是否有NaN或無窮大值
    print("檢查數據質量...")
    print(f"X中NaN數量: {np.isnan(X).sum()}")
    print(f"X中無窮大數量: {np.isinf(X).sum()}")
    print(f"X的範圍: [{np.min(X):.4f}, {np.max(X):.4f}]")
    
    # 處理異常值
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 標準化特徵
    print("標準化特徵...")
    n_samples, n_timesteps, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # 再次檢查標準化後的數據
    print(f"標準化後X的範圍: [{np.min(X_scaled):.4f}, {np.max(X_scaled):.4f}]")
    print(f"標準化後X中NaN數量: {np.isnan(X_scaled).sum()}")
    
    # 保存標準化器
    joblib.dump(scaler, 'model_training/output/lstm/scaler.pkl')
    
    # 分割訓練和驗證集
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"訓練集: {X_train.shape}, 驗證集: {X_val.shape}")
    print(f"訓練集標籤分布: {np.bincount(y_train)}")
    print(f"驗證集標籤分布: {np.bincount(y_val)}")
    
    # 構建模型
    print("構建雙向LSTM模型...")
    model = build_bidirectional_lstm((n_timesteps, n_features))
    model.summary()
    
    # 訓練模型
    print("開始訓練...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
        ]
    )
    
    # 保存模型 (使用新的Keras格式)
    model.save('model_training/output/lstm/bidirectional_lstm_model_fixed.keras')
    print("模型已保存到 bidirectional_lstm_model_fixed.keras")
    
    # 評估模型
    print("\n模型評估:")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"訓練集 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"驗證集 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    # 預測示例
    print("\n預測示例:")
    y_pred = model.predict(X_val[:10])
    for i in range(10):
        print(f"樣本 {i}: 真實={y_val[i]}, 預測={y_pred[i][0]:.4f}")
    
    return model, history

if __name__ == "__main__":
    train_bidirectional_lstm()
