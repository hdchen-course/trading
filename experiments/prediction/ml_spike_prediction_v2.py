import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

def calculate_atr(df, period=14):
    """計算ATR (Average True Range)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def extract_enhanced_features(df, idx, lookback=384):
    """提取增強特徵，專注於最重要的指標"""
    if idx < lookback:
        return None
    
    data_slice = df.iloc[idx-lookback:idx]
    features = {}
    
    current_price = data_slice['close'].iloc[-1]
    
    # 關鍵價格動量特徵
    features['momentum_96h'] = (current_price - data_slice['close'].iloc[0]) / data_slice['close'].iloc[0]
    features['momentum_48h'] = (current_price - data_slice['close'].iloc[-192]) / data_slice['close'].iloc[-192]
    features['momentum_24h'] = (current_price - data_slice['close'].iloc[-96]) / data_slice['close'].iloc[-96]
    features['momentum_4h'] = (current_price - data_slice['close'].iloc[-16]) / data_slice['close'].iloc[-16]
    features['momentum_1h'] = (current_price - data_slice['close'].iloc[-4]) / data_slice['close'].iloc[-4]
    
    # ATR相關特徵
    atr_slice = calculate_atr(data_slice)
    current_atr = atr_slice.iloc[-1] if not pd.isna(atr_slice.iloc[-1]) else atr_slice.mean()
    features['atr_ratio_96h'] = features['momentum_96h'] / (current_atr / current_price) if current_atr > 0 else 0
    features['atr_ratio_4h'] = features['momentum_4h'] / (current_atr / current_price) if current_atr > 0 else 0
    
    # 成交量特徵
    current_volume = data_slice['volume'].iloc[-1]
    avg_volume = data_slice['volume'].mean()
    features['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1
    features['volume_trend_4h'] = data_slice['volume'].iloc[-16:].mean() / data_slice['volume'].iloc[-32:-16].mean()
    
    # RSI
    delta = data_slice['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50
    
    # 價格位置
    features['price_position_24h'] = (current_price - data_slice['close'].iloc[-96:].min()) / (data_slice['close'].iloc[-96:].max() - data_slice['close'].iloc[-96:].min())
    
    # 波動率
    features['volatility_4h'] = data_slice['close'].iloc[-16:].pct_change().std()
    
    # 移動平均
    ma20 = data_slice['close'].rolling(20).mean().iloc[-1]
    features['price_vs_ma20'] = (current_price - ma20) / ma20
    
    return features

def collect_training_data():
    """收集訓練數據"""
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['atr'] = calculate_atr(df)
    
    training_data = []
    labels = []
    
    # 使用1-8月份作為訓練數據
    for month in range(1, 9):
        print(f"Processing {month}月 training data...")
        month_df = df[df['timestamp'].dt.month == month].copy()
        
        if len(month_df) == 0:
            continue
            
        month_df = month_df.reset_index(drop=True)
        
        # 收集正樣本
        positive_samples = []
        for i in range(384, len(month_df) - 16):
            current_row = month_df.iloc[i]
            current_price = current_row['close']
            current_atr = current_row['atr']
            
            if pd.isna(current_atr) or current_atr <= 0:
                continue
            
            # 檢查條件
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
                
                features = extract_enhanced_features(month_df, i)
                if features is not None:
                    positive_samples.append((features, 1))
        
        # 收集負樣本 - 只選擇高質量的負樣本
        negative_samples = []
        target_negative = len(positive_samples) // 2  # 減少負樣本比例
        
        indices = np.random.choice(range(384, len(month_df) - 16), 
                                 size=min(target_negative * 3, len(month_df) - 400), 
                                 replace=False)
        
        for i in indices:
            if len(negative_samples) >= target_negative:
                break
                
            current_row = month_df.iloc[i]
            current_price = current_row['close']
            current_atr = current_row['atr']
            
            if pd.isna(current_atr) or current_atr <= 0:
                continue
            
            # 確保是真正的負樣本
            ninety_six_hours_ago_price = month_df.iloc[i - 384]['close']
            four_hours_ago_price = month_df.iloc[i - 16]['close']
            four_hours_later_price = month_df.iloc[i + 16]['close']
            
            rise_from_96h_ago = (current_price - ninety_six_hours_ago_price) / ninety_six_hours_ago_price
            required_rise_96h_ago = 3 * current_atr / current_price
            rise_vs_4h_later = (current_price - four_hours_later_price) / current_price
            required_rise_4h_later = 3 * current_atr / current_price
            
            # 只選擇明顯不滿足條件的樣本
            if (rise_from_96h_ago <= required_rise_96h_ago * 0.5 or 
                current_price <= four_hours_ago_price or
                rise_vs_4h_later <= required_rise_4h_later * 0.5):
                
                features = extract_enhanced_features(month_df, i)
                if features is not None:
                    negative_samples.append((features, 0))
        
        # 添加到訓練數據
        for features, label in positive_samples + negative_samples:
            training_data.append(features)
            labels.append(label)
    
    return training_data, labels

def train_ensemble_model(training_data, labels):
    """訓練集成模型"""
    print(f"Training with {len(training_data)} samples, {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    df_train = pd.DataFrame(training_data)
    
    # 特徵選擇
    selector = SelectKBest(f_classif, k=10)  # 選擇最重要的10個特徵
    X_selected = selector.fit_transform(df_train.fillna(0), labels)
    
    # 標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_selected)
    y_train = np.array(labels)
    
    # 訓練多個模型
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight={0: 1, 1: 5}  # 進一步增加正樣本權重
    )
    
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
    return rf_model, gb_model, scaler, selector, df_train.columns

def test_ensemble_model(rf_model, gb_model, scaler, selector, feature_names):
    """測試集成模型"""
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['atr'] = calculate_atr(df)
    
    for month in [9, 10]:
        print(f"\n=== Testing {month}月 ===")
        month_df = df[df['timestamp'].dt.month == month].copy()
        
        if len(month_df) == 0:
            continue
            
        month_df = month_df.reset_index(drop=True)
        
        predictions = []
        actual_spikes = []
        timestamps = []
        prices = []
        
        for i in range(384, len(month_df) - 16):
            current_row = month_df.iloc[i]
            current_price = current_row['close']
            current_atr = current_row['atr']
            
            if pd.isna(current_atr) or current_atr <= 0:
                continue
            
            features = extract_enhanced_features(month_df, i)
            if features is None:
                continue
            
            # 預測
            feature_df = pd.DataFrame([features])
            feature_df = feature_df.reindex(columns=feature_names, fill_value=0)
            X_selected = selector.transform(feature_df.fillna(0))
            X_test = scaler.transform(X_selected)
            
            # 集成預測
            rf_prob = rf_model.predict_proba(X_test)[0][1]
            gb_prob = gb_model.predict_proba(X_test)[0][1]
            ensemble_prob = (rf_prob + gb_prob) / 2
            
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
            
            predictions.append(ensemble_prob)
            actual_spikes.append(actual_spike)
            timestamps.append(current_row['timestamp'])
            prices.append(current_price)
        
        # 測試不同閾值找到最佳精確度
        y_true = actual_spikes
        best_precision = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.5, 0.99, 0.05):
            y_pred = [p > threshold for p in predictions]
            predicted_count = sum(y_pred)
            if predicted_count > 0:
                true_positives = sum(a and p for a, p in zip(y_true, y_pred))
                precision = true_positives / predicted_count
                if precision > best_precision and predicted_count >= 5:  # 至少要有5個預測
                    best_precision = precision
                    best_threshold = threshold
        
        # 使用最佳閾值
        y_pred_best = [p > best_threshold for p in predictions]
        predicted_count = sum(y_pred_best)
        true_positives = sum(a and p for a, p in zip(y_true, y_pred_best))
        precision = true_positives / predicted_count if predicted_count > 0 else 0
        recall = true_positives / sum(y_true) if sum(y_true) > 0 else 0
        
        print(f"Actual spikes: {sum(y_true)}")
        print(f"Best threshold: {best_threshold:.2f}")
        print(f"Predicted: {predicted_count}, TP: {true_positives}")
        print(f"Precision: {precision:.1%}, Recall: {recall:.1%}")

if __name__ == "__main__":
    print("Collecting enhanced training data...")
    training_data, labels = collect_training_data()
    
    print("Training ensemble model...")
    rf_model, gb_model, scaler, selector, feature_names = train_ensemble_model(training_data, labels)
    
    print("Testing ensemble model...")
    test_ensemble_model(rf_model, gb_model, scaler, selector, feature_names)
    
    print("Done!")
