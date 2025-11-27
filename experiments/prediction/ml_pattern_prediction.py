import pandas as pd
import numpy as np
import talib
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

def extract_pattern_features(df, idx):
    """使用talib提取K線形態特徵"""
    if idx < 384:
        return None
    
    data_slice = df.iloc[idx-384:idx]
    
    # 轉換為numpy arrays
    high = data_slice['high'].values
    low = data_slice['low'].values
    close = data_slice['close'].values
    open_price = data_slice['open'].values
    volume = data_slice['volume'].values
    
    features = {}
    
    # 基本價格特徵
    current_price = close[-1]
    features['momentum_96h'] = (current_price - close[0]) / close[0]
    features['momentum_4h'] = (current_price - close[-16]) / close[-16]
    
    # K線形態識別 - 反轉形態
    features['doji'] = talib.CDLDOJI(open_price, high, low, close)[-1]
    features['hammer'] = talib.CDLHAMMER(open_price, high, low, close)[-1]
    features['hanging_man'] = talib.CDLHANGINGMAN(open_price, high, low, close)[-1]
    features['shooting_star'] = talib.CDLSHOOTINGSTAR(open_price, high, low, close)[-1]
    features['inverted_hammer'] = talib.CDLINVERTEDHAMMER(open_price, high, low, close)[-1]
    features['dark_cloud'] = talib.CDLDARKCLOUDCOVER(open_price, high, low, close)[-1]
    features['engulfing_bearish'] = talib.CDLENGULFING(open_price, high, low, close)[-1]
    features['evening_star'] = talib.CDLEVENINGSTAR(open_price, high, low, close)[-1]
    
    # K線形態識別 - 持續形態
    features['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(open_price, high, low, close)[-1]
    features['three_black_crows'] = talib.CDL3BLACKCROWS(open_price, high, low, close)[-1]
    features['rising_three'] = talib.CDLRISEFALL3METHODS(open_price, high, low, close)[-1]
    
    # 近期形態統計
    recent_patterns = []
    for i in range(-20, 0):  # 最近20根K線
        if abs(i) < len(close):
            doji = talib.CDLDOJI(open_price, high, low, close)[i]
            hammer = talib.CDLHAMMER(open_price, high, low, close)[i]
            shooting = talib.CDLSHOOTINGSTAR(open_price, high, low, close)[i]
            recent_patterns.extend([doji, hammer, shooting])
    
    features['recent_reversal_patterns'] = sum(1 for p in recent_patterns if p != 0)
    
    # 技術指標
    features['rsi'] = talib.RSI(close, timeperiod=14)[-1]
    features['macd'], features['macd_signal'], _ = talib.MACD(close)
    features['macd'] = features['macd'][-1] if not np.isnan(features['macd'][-1]) else 0
    features['macd_signal'] = features['macd_signal'][-1] if not np.isnan(features['macd_signal'][-1]) else 0
    
    # 布林帶
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
    features['bb_position'] = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if bb_upper[-1] > bb_lower[-1] else 0.5
    
    # 成交量指標
    features['volume_sma'] = talib.SMA(volume, timeperiod=20)[-1]
    features['volume_ratio'] = volume[-1] / features['volume_sma'] if features['volume_sma'] > 0 else 1
    
    # ATR
    atr_values = talib.ATR(high, low, close, timeperiod=14)
    features['atr'] = atr_values[-1]
    features['atr_ratio_96h'] = features['momentum_96h'] / (features['atr'] / current_price) if features['atr'] > 0 else 0
    
    return features

def collect_pattern_training_data():
    """收集包含形態特徵的訓練數據"""
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    training_data = []
    labels = []
    
    for month in range(1, 9):
        print(f"Processing {month}月 with pattern features...")
        month_df = df[df['timestamp'].dt.month == month].copy().reset_index(drop=True)
        
        if len(month_df) == 0:
            continue
        
        # 添加ATR列
        month_df['atr'] = calculate_atr(month_df)
        
        # 收集正樣本
        for i in range(384, len(month_df) - 16):
            current_row = month_df.iloc[i]
            current_price = current_row['close']
            current_atr = current_row['atr']
            
            if pd.isna(current_atr) or current_atr <= 0:
                continue
            
            # 檢查基本條件
            ninety_six_hours_ago_price = month_df.iloc[i - 384]['close']
            four_hours_ago_price = month_df.iloc[i - 16]['close']
            four_hours_later_price = month_df.iloc[i + 16]['close']
            
            rise_from_96h_ago = (current_price - ninety_six_hours_ago_price) / ninety_six_hours_ago_price
            required_rise_96h_ago = 3 * current_atr / current_price
            rise_vs_4h_later = (current_price - four_hours_later_price) / current_price
            required_rise_4h_later = 3 * current_atr / current_price
            
            if (rise_from_96h_ago > required_rise_96h_ago and 
                current_price > four_hours_ago_price and
                rise_vs_4h_later > required_rise_4h_later):
                
                features = extract_pattern_features(month_df, i)
                if features is not None:
                    training_data.append(features)
                    labels.append(1)
        
        # 收集負樣本
        negative_count = 0
        target_negative = sum(labels) // 2
        
        indices = np.random.choice(range(384, len(month_df) - 16), 
                                 size=min(target_negative * 3, len(month_df) - 400), 
                                 replace=False)
        
        for i in indices:
            if negative_count >= target_negative:
                break
                
            current_row = month_df.iloc[i]
            current_price = current_row['close']
            current_atr = current_row['atr']
            
            if pd.isna(current_atr) or current_atr <= 0:
                continue
            
            # 確保是負樣本
            ninety_six_hours_ago_price = month_df.iloc[i - 384]['close']
            four_hours_ago_price = month_df.iloc[i - 16]['close']
            four_hours_later_price = month_df.iloc[i + 16]['close']
            
            rise_from_96h_ago = (current_price - ninety_six_hours_ago_price) / ninety_six_hours_ago_price
            required_rise_96h_ago = 3 * current_atr / current_price
            
            if (rise_from_96h_ago <= required_rise_96h_ago * 0.5 or 
                current_price <= four_hours_ago_price):
                
                features = extract_pattern_features(month_df, i)
                if features is not None:
                    training_data.append(features)
                    labels.append(0)
                    negative_count += 1
    
    return training_data, labels

def train_pattern_model(training_data, labels):
    """訓練包含形態特徵的模型"""
    print(f"Training with {len(training_data)} samples, {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    df_train = pd.DataFrame(training_data)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train.fillna(0))
    y_train = np.array(labels)
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight={0: 1, 1: 4}
    )
    
    model.fit(X_train, y_train)
    
    # 顯示特徵重要性
    feature_importance = pd.DataFrame({
        'feature': df_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    return model, scaler, df_train.columns

def test_pattern_model(model, scaler, feature_names):
    """測試形態模型"""
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    for month in [9, 10]:
        print(f"\n=== Testing {month}月 with Pattern Features ===")
        month_df = df[df['timestamp'].dt.month == month].copy().reset_index(drop=True)
        
        if len(month_df) == 0:
            continue
        
        month_df['atr'] = calculate_atr(month_df)
        
        predictions = []
        actual_spikes = []
        
        for i in range(384, len(month_df) - 16):
            current_row = month_df.iloc[i]
            current_price = current_row['close']
            current_atr = current_row['atr']
            
            if pd.isna(current_atr) or current_atr <= 0:
                continue
            
            features = extract_pattern_features(month_df, i)
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
        
        # 測試不同閾值
        y_true = actual_spikes
        
        for threshold in [0.7, 0.8, 0.9, 0.95]:
            y_pred = [p > threshold for p in predictions]
            predicted_count = sum(y_pred)
            
            if predicted_count > 0:
                true_positives = sum(a and p for a, p in zip(y_true, y_pred))
                precision = true_positives / predicted_count
                recall = true_positives / sum(y_true) if sum(y_true) > 0 else 0
                
                print(f"Threshold {threshold}: Predicted={predicted_count}, TP={true_positives}, Precision={precision:.1%}, Recall={recall:.1%}")

if __name__ == "__main__":
    print("Collecting pattern-based training data...")
    training_data, labels = collect_pattern_training_data()
    
    print("Training pattern model...")
    model, scaler, feature_names = train_pattern_model(training_data, labels)
    
    print("Testing pattern model...")
    test_pattern_model(model, scaler, feature_names)
    
    print("Done!")
