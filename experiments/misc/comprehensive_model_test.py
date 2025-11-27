import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 传统机器学习
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, skipping...")

# TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available, skipping LSTM models...")

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def extract_features(df, idx):
    if idx < 384:
        return None
    
    data_slice = df.iloc[idx-384:idx]
    current_price = data_slice['close'].iloc[-1]
    
    features = {}
    
    # 动量特征
    features['momentum_96h'] = (current_price - data_slice['close'].iloc[0]) / data_slice['close'].iloc[0]
    features['momentum_48h'] = (current_price - data_slice['close'].iloc[-192]) / data_slice['close'].iloc[-192]
    features['momentum_24h'] = (current_price - data_slice['close'].iloc[-96]) / data_slice['close'].iloc[-96]
    features['momentum_4h'] = (current_price - data_slice['close'].iloc[-16]) / data_slice['close'].iloc[-16]
    features['momentum_1h'] = (current_price - data_slice['close'].iloc[-4]) / data_slice['close'].iloc[-4]
    
    # ATR特征
    atr_current = data_slice['atr'].iloc[-1] if not pd.isna(data_slice['atr'].iloc[-1]) else data_slice['atr'].mean()
    features['atr_ratio_96h'] = features['momentum_96h'] / (atr_current / current_price) if atr_current > 0 else 0
    features['atr_ratio_4h'] = features['momentum_4h'] / (atr_current / current_price) if atr_current > 0 else 0
    
    # 成交量特征
    features['volume_ratio'] = data_slice['volume'].iloc[-1] / data_slice['volume'].mean()
    features['volume_trend'] = data_slice['volume'].iloc[-16:].mean() / data_slice['volume'].iloc[-32:-16].mean()
    
    # 价格位置
    features['price_position'] = (current_price - data_slice['close'].min()) / (data_slice['close'].max() - data_slice['close'].min())
    
    # 波动率
    features['volatility'] = data_slice['close'].pct_change().std()
    features['volatility_4h'] = data_slice['close'].iloc[-16:].pct_change().std()
    
    return features

def prepare_lstm_data(df, lookback=96):
    """准备LSTM数据格式"""
    X, y = [], []
    
    for i in range(384, len(df) - 16):
        current_row = df.iloc[i]
        current_price = current_row['close']
        current_atr = current_row['atr']
        
        if pd.isna(current_atr) or current_atr <= 0:
            continue
        
        # 检查标签
        ninety_six_hours_ago_price = df.iloc[i - 384]['close']
        four_hours_ago_price = df.iloc[i - 16]['close']
        four_hours_later_price = df.iloc[i + 16]['close']
        
        rise_from_96h_ago = (current_price - ninety_six_hours_ago_price) / ninety_six_hours_ago_price
        required_rise_96h_ago = 3 * current_atr / current_price
        rise_vs_4h_later = (current_price - four_hours_later_price) / current_price
        required_rise_4h_later = 3 * current_atr / current_price
        
        is_spike = (rise_from_96h_ago > required_rise_96h_ago and 
                   current_price > four_hours_ago_price and
                   rise_vs_4h_later > required_rise_4h_later)
        
        # 准备序列数据
        sequence = df.iloc[i-lookback:i][['close', 'volume', 'high', 'low']].values
        if len(sequence) == lookback:
            X.append(sequence)
            y.append(1 if is_spike else 0)
    
    return np.array(X), np.array(y)

def collect_training_data():
    """收集2022-2024训练数据"""
    print("Loading BTC data...")
    
    # 尝试读取新数据，如果没有则使用旧数据
    try:
        df = pd.read_csv('training_data/btc_15m_2022_2025.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except FileNotFoundError:
        print("Using existing 12-month data...")
        df = pd.read_csv('training_data/btc_12months_15m.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['atr'] = calculate_atr(df)
    
    # 分割训练和测试数据
    train_df = df[df['timestamp'].dt.year <= 2024].copy().reset_index(drop=True)
    test_df = df[df['timestamp'].dt.year == 2025].copy().reset_index(drop=True)
    
    print(f"Training data: {len(train_df)} records ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"Test data: {len(test_df)} records ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    
    # 提取特征
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    
    # 训练数据
    for i in range(384, len(train_df) - 16):
        current_row = train_df.iloc[i]
        current_price = current_row['close']
        current_atr = current_row['atr']
        
        if pd.isna(current_atr) or current_atr <= 0:
            continue
        
        # 检查条件
        ninety_six_hours_ago_price = train_df.iloc[i - 384]['close']
        four_hours_ago_price = train_df.iloc[i - 16]['close']
        four_hours_later_price = train_df.iloc[i + 16]['close']
        
        rise_from_96h_ago = (current_price - ninety_six_hours_ago_price) / ninety_six_hours_ago_price
        required_rise_96h_ago = 3 * current_atr / current_price
        rise_vs_4h_later = (current_price - four_hours_later_price) / current_price
        required_rise_4h_later = 3 * current_atr / current_price
        
        is_spike = (rise_from_96h_ago > required_rise_96h_ago and 
                   current_price > four_hours_ago_price and
                   rise_vs_4h_later > required_rise_4h_later)
        
        features = extract_features(train_df, i)
        if features is not None:
            train_features.append(features)
            train_labels.append(1 if is_spike else 0)
    
    # 测试数据
    for i in range(384, len(test_df) - 16):
        current_row = test_df.iloc[i]
        current_price = current_row['close']
        current_atr = current_row['atr']
        
        if pd.isna(current_atr) or current_atr <= 0:
            continue
        
        ninety_six_hours_ago_price = test_df.iloc[i - 384]['close']
        four_hours_ago_price = test_df.iloc[i - 16]['close']
        four_hours_later_price = test_df.iloc[i + 16]['close']
        
        rise_from_96h_ago = (current_price - ninety_six_hours_ago_price) / ninety_six_hours_ago_price
        required_rise_96h_ago = 3 * current_atr / current_price
        rise_vs_4h_later = (current_price - four_hours_later_price) / current_price
        required_rise_4h_later = 3 * current_atr / current_price
        
        is_spike = (rise_from_96h_ago > required_rise_96h_ago and 
                   current_price > four_hours_ago_price and
                   rise_vs_4h_later > required_rise_4h_later)
        
        features = extract_features(test_df, i)
        if features is not None:
            test_features.append(features)
            test_labels.append(1 if is_spike else 0)
    
    return train_features, train_labels, test_features, test_labels, train_df, test_df

def test_traditional_models(X_train, y_train, X_test, y_test):
    """测试传统机器学习模型"""
    print("\n" + "="*60)
    print("TRADITIONAL MACHINE LEARNING MODELS")
    print("="*60)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'SVM': SVC(random_state=42, class_weight='balanced', probability=True, kernel='rbf')
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            model.fit(X_train, y_train)
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # 测试不同阈值
                for threshold in [0.5, 0.7, 0.8, 0.9]:
                    y_pred = (y_pred_proba > threshold).astype(int)
                    
                    if sum(y_pred) > 0:
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        
                        results.append({
                            'model': f"{name} (t={threshold})",
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'predicted': sum(y_pred),
                            'tp': sum((y_test == 1) & (y_pred == 1))
                        })
                        
                        if threshold == 0.7:  # 主要结果
                            print(f"  Threshold {threshold}: Predicted={sum(y_pred)}, TP={sum((y_test == 1) & (y_pred == 1))}, Precision={precision:.1%}, Recall={recall:.1%}")
            
        except Exception as e:
            print(f"  Error training {name}: {e}")
    
    return results

def create_lstm_model(input_shape, bidirectional=False):
    """创建LSTM模型"""
    model = Sequential()
    
    if bidirectional:
        model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(50)))
    else:
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
    
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def test_lstm_models(train_df, test_df):
    """测试LSTM模型"""
    if not TENSORFLOW_AVAILABLE:
        print("\nTensorFlow not available, skipping LSTM models...")
        return []
    
    print("\n" + "="*60)
    print("DEEP LEARNING MODELS (LSTM)")
    print("="*60)
    
    # 准备LSTM数据
    print("Preparing LSTM data...")
    X_train_lstm, y_train_lstm = prepare_lstm_data(train_df)
    X_test_lstm, y_test_lstm = prepare_lstm_data(test_df)
    
    if len(X_train_lstm) == 0 or len(X_test_lstm) == 0:
        print("No LSTM data available")
        return []
    
    print(f"LSTM Training data: {len(X_train_lstm)} samples")
    print(f"LSTM Test data: {len(X_test_lstm)} samples")
    
    # 标准化
    scaler = StandardScaler()
    X_train_lstm_scaled = scaler.fit_transform(X_train_lstm.reshape(-1, X_train_lstm.shape[-1])).reshape(X_train_lstm.shape)
    X_test_lstm_scaled = scaler.transform(X_test_lstm.reshape(-1, X_test_lstm.shape[-1])).reshape(X_test_lstm.shape)
    
    results = []
    
    # 测试LSTM和Bidirectional LSTM
    for model_name, bidirectional in [('LSTM', False), ('Bidirectional LSTM', True)]:
        print(f"\nTraining {model_name}...")
        try:
            model = create_lstm_model(X_train_lstm_scaled.shape[1:], bidirectional)
            
            # 训练模型
            history = model.fit(
                X_train_lstm_scaled, y_train_lstm,
                epochs=20,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # 预测
            y_pred_proba = model.predict(X_test_lstm_scaled, verbose=0).flatten()
            
            # 测试不同阈值
            for threshold in [0.5, 0.7, 0.8, 0.9]:
                y_pred = (y_pred_proba > threshold).astype(int)
                
                if sum(y_pred) > 0:
                    precision = precision_score(y_test_lstm, y_pred, zero_division=0)
                    recall = recall_score(y_test_lstm, y_pred, zero_division=0)
                    f1 = f1_score(y_test_lstm, y_pred, zero_division=0)
                    
                    results.append({
                        'model': f"{model_name} (t={threshold})",
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'predicted': sum(y_pred),
                        'tp': sum((y_test_lstm == 1) & (y_pred == 1))
                    })
                    
                    if threshold == 0.7:  # 主要结果
                        print(f"  Threshold {threshold}: Predicted={sum(y_pred)}, TP={sum((y_test_lstm == 1) & (y_pred == 1))}, Precision={precision:.1%}, Recall={recall:.1%}")
        
        except Exception as e:
            print(f"  Error training {model_name}: {e}")
    
    return results

def main():
    print("Comprehensive Model Testing")
    print("Training: 2022-2024, Testing: 2025")
    print("="*60)
    
    # 收集数据
    train_features, train_labels, test_features, test_labels, train_df, test_df = collect_training_data()
    
    print(f"\nTraining samples: {len(train_features)} ({sum(train_labels)} positive)")
    print(f"Test samples: {len(test_features)} ({sum(test_labels)} positive)")
    
    # 准备传统ML数据
    df_train = pd.DataFrame(train_features)
    df_test = pd.DataFrame(test_features)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train.fillna(0))
    X_test = scaler.transform(df_test.fillna(0))
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    
    # 测试传统模型
    traditional_results = test_traditional_models(X_train, y_train, X_test, y_test)
    
    # 测试LSTM模型
    lstm_results = test_lstm_models(train_df, test_df)
    
    # 汇总结果
    all_results = traditional_results + lstm_results
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Predicted':<10} {'TP':<5} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    print("-" * 80)
    
    for result in sorted(all_results, key=lambda x: x['precision'], reverse=True):
        print(f"{result['model']:<25} {result['predicted']:<10} {result['tp']:<5} {result['precision']:<10.1%} {result['recall']:<8.1%} {result['f1']:<8.3f}")
    
    # 找出最佳模型
    if all_results:
        best_precision = max(all_results, key=lambda x: x['precision'])
        print(f"\n🏆 Best Precision: {best_precision['model']} ({best_precision['precision']:.1%})")
        
        high_precision = [r for r in all_results if r['precision'] >= 0.4]
        if high_precision:
            print(f"\n🎉 Models achieving ≥40% precision:")
            for r in high_precision:
                print(f"  {r['model']}: {r['precision']:.1%}")
        else:
            print(f"\n❌ No models achieved ≥40% precision target")

if __name__ == "__main__":
    main()
