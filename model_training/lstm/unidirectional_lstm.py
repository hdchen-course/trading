#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import joblib
import sys
sys.path.append('..')
from detection import MacdReversalDetection
from utils.indicators import analyze_cumulative_metrics

def build_unidirectional_lstm(input_shape):
    """構建單向LSTM模型（用於實時預測）"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def generate_validation_data():
    """生成驗證數據：2025/09-10的數據"""
    
    print("分析2025年9-10月數據...")
    analysis_df = analyze_cumulative_metrics("2025-09-01", "2025-10-31")
    
    print("檢測信號點...")
    detector = MacdReversalDetection()
    signal_positions = detector.detect(analysis_df)
    
    print(f"找到 {len(signal_positions)} 個信號點")
    
    # 特徵列
    feature_cols = [
        'close_price', 'ema_100', 'ema_50', 'avg_amplitude', 'avg_volume',
        'avg_macd', 'avg_rsi', 'price_above_sar_ratio', 'avg_bb_position',
        'volatility', 'price_change_pct', 'price_atr_ratio_1h', 'volume_trend',
        'macd_2nd', 'price_2nd', 'vol_2nd', 'atr_2nd'
    ]
    
    sequences = []
    labels = []
    
    # 單向LSTM只用過去24小時數據
    window_size = 96  # 24小時
    
    # 為每個時間點生成預測樣本（滑動窗口）
    for i in range(window_size, len(analysis_df)):
        sequence = analysis_df.iloc[i-window_size:i][feature_cols].values
        sequences.append(sequence)
        
        # 標籤：當前時間點是否為信號
        label = 1 if i in signal_positions else 0
        labels.append(label)
    
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"驗證數據: {X.shape}, 標籤: {y.shape}")
    print(f"信號樣本: {np.sum(y)}, 非信號樣本: {len(y) - np.sum(y)}")
    
    return X, y

def train_unidirectional_lstm():
    """訓練單向LSTM模型"""
    
    # 載入雙向LSTM訓練的標準化器
    scaler = joblib.load('lstm/scaler.pkl')
    
    # 生成單向訓練數據（只用過去數據）
    print("生成單向LSTM訓練數據...")
    X_train = np.load('lstm/X_train.npy')
    y_train = np.load('lstm/y_train.npy')
    
    # 只取前半部分作為輸入（過去24小時）
    X_train_uni = X_train[:, :96, :]  # 只用前96個時間步
    
    # 標準化
    n_samples, n_timesteps, n_features = X_train_uni.shape
    X_reshaped = X_train_uni.reshape(-1, n_features)
    X_scaled = scaler.transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # 構建單向模型
    print("構建單向LSTM模型...")
    model = build_unidirectional_lstm((n_timesteps, n_features))
    
    # 可選：從雙向模型初始化權重
    try:
        bidirectional_model = load_model('lstm/bidirectional_lstm_model.h5')
        print("從雙向模型初始化權重...")
        # 這裡可以添加權重遷移邏輯
    except:
        print("未找到雙向模型，從頭訓練...")
    
    # 訓練模型
    print("開始訓練單向LSTM...")
    history = model.fit(
        X_scaled, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
    )
    
    # 保存模型
    model.save('model_training/output/lstm/unidirectional_lstm_model.h5')
    print("單向模型已保存到 model_training/output/lstm/unidirectional_lstm_model.h5")
    
    return model

def validate_unidirectional_lstm():
    """在9-10月數據上驗證單向LSTM"""
    
    print("載入單向LSTM模型...")
    model = load_model('models/lstm/unidirectional_lstm_model.h5')
    scaler = joblib.load('lstm/scaler.pkl')
    
    # 生成驗證數據
    X_val, y_val = generate_validation_data()
    
    # 標準化
    n_samples, n_timesteps, n_features = X_val.shape
    X_reshaped = X_val.reshape(-1, n_features)
    X_scaled = scaler.transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # 預測
    print("進行預測...")
    y_pred_prob = model.predict(X_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # 評估
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n分類報告:")
    print(classification_report(y_val, y_pred))
    
    print("\n混淆矩陣:")
    print(confusion_matrix(y_val, y_pred))
    
    # 計算準確率
    accuracy = np.mean(y_pred.flatten() == y_val)
    print(f"\n準確率: {accuracy:.4f}")
    
    return y_pred_prob, y_pred

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'validate'], default='train')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_unidirectional_lstm()
    else:
        validate_unidirectional_lstm()
