import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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

def extract_critical_features(df, idx):
    """只提取最關鍵的特徵"""
    if idx < 384:
        return None
    
    data_slice = df.iloc[idx-384:idx]
    current_price = data_slice['close'].iloc[-1]
    current_atr = calculate_atr(data_slice).iloc[-1]
    
    features = {}
    
    # 核心動量特徵
    features['momentum_96h'] = (current_price - data_slice['close'].iloc[0]) / data_slice['close'].iloc[0]
    features['momentum_4h'] = (current_price - data_slice['close'].iloc[-16]) / data_slice['close'].iloc[-16]
    
    # ATR標準化動量
    features['atr_momentum_96h'] = features['momentum_96h'] / (current_atr / current_price) if current_atr > 0 else 0
    features['atr_momentum_4h'] = features['momentum_4h'] / (current_atr / current_price) if current_atr > 0 else 0
    
    # 成交量異常
    current_volume = data_slice['volume'].iloc[-1]
    avg_volume = data_slice['volume'].mean()
    features['volume_spike'] = current_volume / avg_volume if avg_volume > 0 else 1
    
    # 價格位置
    recent_high = data_slice['close'].iloc[-96:].max()
    recent_low = data_slice['close'].iloc[-96:].min()
    features['price_position'] = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
    
    return features

def collect_high_quality_data():
    """收集高質量訓練數據"""
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['atr'] = calculate_atr(df)
    
    training_data = []
    labels = []
    
    for month in range(1, 9):
        print(f"Processing {month}月...")
        month_df = df[df['timestamp'].dt.month == month].copy().reset_index(drop=True)
        
        if len(month_df) == 0:
            continue
        
        # 收集極高質量的正樣本
        for i in range(384, len(month_df) - 16):
            current_row = month_df.iloc[i]
            current_price = current_row['close']
            current_atr = current_row['atr']
            
            if pd.isna(current_atr) or current_atr <= 0:
                continue
            
            ninety_six_hours_ago_price = month_df.iloc[i - 384]['close']
            four_hours_ago_price = month_df.iloc[i - 16]['close']
            four_hours_later_price = month_df.iloc[i + 16]['close']
            
            rise_from_96h_ago = (current_price - ninety_six_hours_ago_price) / ninety_six_hours_ago_price
            required_rise_96h_ago = 3 * current_atr / current_price
            rise_vs_4h_later = (current_price - four_hours_later_price) / current_price
            required_rise_4h_later = 3 * current_atr / current_price
            
            # 只選擇明顯超過閾值的樣本
            if (rise_from_96h_ago > required_rise_96h_ago * 1.5 and  # 超過1.5倍要求
                current_price > four_hours_ago_price * 1.01 and      # 明顯高於4小時前
                rise_vs_4h_later > required_rise_4h_later * 1.2):   # 超過1.2倍要求
                
                features = extract_critical_features(month_df, i)
                if features is not None:
                    training_data.append(features)
                    labels.append(1)
        
        # 收集明確的負樣本
        negative_count = 0
        target_negative = sum(labels) // 3  # 1:3 正負比例
        
        indices = np.random.choice(range(384, len(month_df) - 16), 
                                 size=min(target_negative * 5, len(month_df) - 400), 
                                 replace=False)
        
        for i in indices:
            if negative_count >= target_negative:
                break
                
            current_row = month_df.iloc[i]
            current_price = current_row['close']
            current_atr = current_row['atr']
            
            if pd.isna(current_atr) or current_atr <= 0:
                continue
            
            ninety_six_hours_ago_price = month_df.iloc[i - 384]['close']
            four_hours_ago_price = month_df.iloc[i - 16]['close']
            four_hours_later_price = month_df.iloc[i + 16]['close']
            
            rise_from_96h_ago = (current_price - ninety_six_hours_ago_price) / ninety_six_hours_ago_price
            required_rise_96h_ago = 3 * current_atr / current_price
            
            # 只選擇明顯不符合條件的樣本
            if (rise_from_96h_ago < required_rise_96h_ago * 0.3 or  # 遠低於要求
                current_price < four_hours_ago_price * 0.99):      # 明顯低於4小時前
                
                features = extract_critical_features(month_df, i)
                if features is not None:
                    training_data.append(features)
                    labels.append(0)
                    negative_count += 1
    
    return training_data, labels

def train_precision_model(training_data, labels):
    """訓練高精確度模型"""
    print(f"Training with {len(training_data)} samples, {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    df_train = pd.DataFrame(training_data)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train.fillna(0))
    y_train = np.array(labels)
    
    # 極度保守的模型參數
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,           # 很淺的樹
        min_samples_split=50,  # 很大的分割要求
        min_samples_leaf=25,   # 很大的葉節點要求
        max_features=3,        # 只用3個特徵
        random_state=42,
        class_weight={0: 1, 1: 10}  # 極高的正樣本權重
    )
    
    model.fit(X_train, y_train)
    return model, scaler, df_train.columns

def test_precision_model(model, scaler, feature_names):
    """測試高精確度模型"""
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['atr'] = calculate_atr(df)
    
    for month in [9, 10]:
        print(f"\n=== Testing {month}月 ===")
        month_df = df[df['timestamp'].dt.month == month].copy().reset_index(drop=True)
        
        if len(month_df) == 0:
            continue
        
        predictions = []
        actual_spikes = []
        
        for i in range(384, len(month_df) - 16):
            current_row = month_df.iloc[i]
            current_price = current_row['close']
            current_atr = current_row['atr']
            
            if pd.isna(current_atr) or current_atr <= 0:
                continue
            
            features = extract_critical_features(month_df, i)
            if features is None:
                continue
            
            # 預測
            feature_df = pd.DataFrame([features])
            feature_df = feature_df.reindex(columns=feature_names, fill_value=0)
            X_test = scaler.transform(feature_df.fillna(0))
            pred_prob = model.predict_proba(X_test)[0][1]
            
            # 檢查實際條件
            ninety_six_hours_ago_price = month_df.iloc[i - 384]['close']
            four_hours_ago_price = month_df.iloc[i - 16]['close']
            four_hours_later_price = month_df.iloc[i + 16]['close']
            
            rise_from_96h_ago = (current_price - ninety_six_hours_ago_price) / ninety_six_hours_ago_price
            required_rise_96h_ago = 3 * current_atr / current_price
            rise_vs_4h_later = (current_price - four_hours_later_price) / current_price
            required_rise_4h_later = 3 * current_atr / current_price
            
            actual_spike = (rise_from_96h_ago > required_rise_96h_ago and 
                          current_price > four_hours_ago_price and
                          rise_vs_4h_later > required_rise_4h_later)
            
            predictions.append(pred_prob)
            actual_spikes.append(actual_spike)
        
        # 測試極高閾值
        y_true = actual_spikes
        
        for threshold in [0.98, 0.99, 0.995]:
            y_pred = [p > threshold for p in predictions]
            predicted_count = sum(y_pred)
            
            if predicted_count > 0:
                true_positives = sum(a and p for a, p in zip(y_true, y_pred))
                precision = true_positives / predicted_count
                recall = true_positives / sum(y_true) if sum(y_true) > 0 else 0
                
                print(f"Threshold {threshold}: Predicted={predicted_count}, TP={true_positives}, Precision={precision:.1%}, Recall={recall:.1%}")

if __name__ == "__main__":
    print("Collecting high-quality training data...")
    training_data, labels = collect_high_quality_data()
    
    print("Training precision-focused model...")
    model, scaler, feature_names = train_precision_model(training_data, labels)
    
    print("Testing precision model...")
    test_precision_model(model, scaler, feature_names)
    
    print("Done!")
