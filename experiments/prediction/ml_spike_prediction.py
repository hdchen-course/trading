import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
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

def extract_features(df, idx, lookback=384):
    """提取前96小時(384個15分鐘)的特徵"""
    if idx < lookback:
        return None
    
    # 前96小時數據
    data_slice = df.iloc[idx-lookback:idx]
    
    features = {}
    
    # 價格特徵
    features['price_mean'] = data_slice['close'].mean()
    features['price_std'] = data_slice['close'].std()
    features['price_min'] = data_slice['close'].min()
    features['price_max'] = data_slice['close'].max()
    features['price_range'] = features['price_max'] - features['price_min']
    features['price_current'] = data_slice['close'].iloc[-1]
    
    # 趨勢特徵
    features['price_trend'] = (data_slice['close'].iloc[-1] - data_slice['close'].iloc[0]) / data_slice['close'].iloc[0]
    features['price_momentum_24h'] = (data_slice['close'].iloc[-1] - data_slice['close'].iloc[-96]) / data_slice['close'].iloc[-96]
    features['price_momentum_48h'] = (data_slice['close'].iloc[-1] - data_slice['close'].iloc[-192]) / data_slice['close'].iloc[-192]
    features['price_momentum_12h'] = (data_slice['close'].iloc[-1] - data_slice['close'].iloc[-48]) / data_slice['close'].iloc[-48]
    features['price_momentum_6h'] = (data_slice['close'].iloc[-1] - data_slice['close'].iloc[-24]) / data_slice['close'].iloc[-24]
    
    # 短期趨勢
    features['price_trend_1h'] = (data_slice['close'].iloc[-1] - data_slice['close'].iloc[-4]) / data_slice['close'].iloc[-4]
    features['price_trend_2h'] = (data_slice['close'].iloc[-1] - data_slice['close'].iloc[-8]) / data_slice['close'].iloc[-8]
    features['price_trend_4h'] = (data_slice['close'].iloc[-1] - data_slice['close'].iloc[-16]) / data_slice['close'].iloc[-16]
    
    # 成交量特徵
    features['volume_mean'] = data_slice['volume'].mean()
    features['volume_std'] = data_slice['volume'].std()
    features['volume_current'] = data_slice['volume'].iloc[-1]
    features['volume_ratio_current'] = data_slice['volume'].iloc[-1] / data_slice['volume'].mean()
    features['volume_trend'] = (data_slice['volume'].iloc[-24:].mean() - data_slice['volume'].iloc[:24].mean()) / data_slice['volume'].iloc[:24].mean()
    features['volume_trend_4h'] = (data_slice['volume'].iloc[-16:].mean() - data_slice['volume'].iloc[-32:-16].mean()) / data_slice['volume'].iloc[-32:-16].mean()
    
    # ATR特徵
    atr_slice = calculate_atr(data_slice)
    features['atr_mean'] = atr_slice.mean()
    features['atr_current'] = atr_slice.iloc[-1] if not pd.isna(atr_slice.iloc[-1]) else atr_slice.mean()
    features['atr_ratio'] = features['atr_current'] / features['atr_mean'] if features['atr_mean'] > 0 else 1
    
    # 波動率特徵
    returns = data_slice['close'].pct_change().dropna()
    features['volatility'] = returns.std()
    features['volatility_24h'] = data_slice['close'].iloc[-96:].pct_change().std()
    features['volatility_4h'] = data_slice['close'].iloc[-16:].pct_change().std()
    features['skewness'] = returns.skew()
    features['kurtosis'] = returns.kurtosis()
    
    # 技術指標特徵
    # RSI
    delta = data_slice['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50
    
    # 移動平均
    features['ma_5'] = data_slice['close'].rolling(5).mean().iloc[-1]
    features['ma_20'] = data_slice['close'].rolling(20).mean().iloc[-1]
    features['ma_50'] = data_slice['close'].rolling(50).mean().iloc[-1]
    features['price_vs_ma5'] = (features['price_current'] - features['ma_5']) / features['ma_5']
    features['price_vs_ma20'] = (features['price_current'] - features['ma_20']) / features['ma_20']
    features['price_vs_ma50'] = (features['price_current'] - features['ma_50']) / features['ma_50']
    
    # 布林帶
    bb_period = 20
    bb_std = 2
    bb_ma = data_slice['close'].rolling(bb_period).mean()
    bb_std_val = data_slice['close'].rolling(bb_period).std()
    bb_upper = bb_ma + (bb_std_val * bb_std)
    bb_lower = bb_ma - (bb_std_val * bb_std)
    features['bb_position'] = (features['price_current'] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
    
    # 價格位置特徵
    features['price_percentile_24h'] = (features['price_current'] - data_slice['close'].iloc[-96:].min()) / (data_slice['close'].iloc[-96:].max() - data_slice['close'].iloc[-96:].min())
    features['price_percentile_4h'] = (features['price_current'] - data_slice['close'].iloc[-16:].min()) / (data_slice['close'].iloc[-16:].max() - data_slice['close'].iloc[-16:].min())
    
    # 高低點特徵
    features['high_low_ratio'] = data_slice['high'].iloc[-1] / data_slice['low'].iloc[-1]
    features['close_position'] = (data_slice['close'].iloc[-1] - data_slice['low'].iloc[-1]) / (data_slice['high'].iloc[-1] - data_slice['low'].iloc[-1])
    
    # 連續上漲/下跌特徵
    price_changes = data_slice['close'].diff()
    features['consecutive_up'] = 0
    features['consecutive_down'] = 0
    for i in range(len(price_changes)-1, 0, -1):
        if price_changes.iloc[i] > 0:
            features['consecutive_up'] += 1
        else:
            break
    for i in range(len(price_changes)-1, 0, -1):
        if price_changes.iloc[i] < 0:
            features['consecutive_down'] += 1
        else:
            break
    
    return features

def collect_training_data():
    """收集訓練數據"""
    # 讀取完整數據
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
        
        # 檢測滿足條件的點
        for i in range(384, len(month_df) - 16):
            current_row = month_df.iloc[i]
            current_price = current_row['close']
            current_atr = current_row['atr']
            
            if pd.isna(current_atr) or current_atr <= 0:
                continue
            
            # 96小時前的價格
            ninety_six_hours_ago_price = month_df.iloc[i - 384]['close']
            # 4小時前的價格
            four_hours_ago_price = month_df.iloc[i - 16]['close']
            # 4小時後的價格
            four_hours_later_price = month_df.iloc[i + 16]['close']
            
            # 檢查是否滿足條件
            rise_from_96h_ago = (current_price - ninety_six_hours_ago_price) / ninety_six_hours_ago_price
            required_rise_96h_ago = 3 * current_atr / current_price
            
            if rise_from_96h_ago <= required_rise_96h_ago:
                continue
                
            if current_price <= four_hours_ago_price:
                continue
                
            rise_vs_4h_later = (current_price - four_hours_later_price) / current_price
            required_rise_4h_later = 3 * current_atr / current_price
            
            if rise_vs_4h_later <= required_rise_4h_later:
                continue
            
            # 提取特徵
            features = extract_features(month_df, i)
            if features is None:
                continue
            
            training_data.append(features)
            labels.append(1)  # 滿足條件的點標記為1
    
    # 添加負樣本 (隨機選擇不滿足條件的點)
    print("Adding negative samples...")
    negative_count = 0
    target_negative = len(training_data)  # 與正樣本數量相等
    
    for month in range(1, 9):
        month_df = df[df['timestamp'].dt.month == month].copy()
        if len(month_df) == 0:
            continue
        month_df = month_df.reset_index(drop=True)
        
        # 隨機選擇一些點作為負樣本
        indices = np.random.choice(range(384, len(month_df) - 16), 
                                 size=min(target_negative // 8, len(month_df) - 400), 
                                 replace=False)
        
        for i in indices:
            if negative_count >= target_negative:
                break
                
            # 檢查是否不滿足條件
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
            
            # 如果滿足條件就跳過
            if (rise_from_96h_ago > required_rise_96h_ago and 
                current_price > four_hours_ago_price and
                (current_price - four_hours_later_price) / current_price > 3 * current_atr / current_price):
                continue
            
            features = extract_features(month_df, i)
            if features is None:
                continue
            
            training_data.append(features)
            labels.append(0)  # 不滿足條件的點標記為0
            negative_count += 1
    
    return training_data, labels

def train_model(training_data, labels):
    """訓練模型"""
    print(f"Training with {len(training_data)} samples, {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    # 轉換為DataFrame
    df_train = pd.DataFrame(training_data)
    
    # 標準化特徵
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train.fillna(0))
    y_train = np.array(labels)
    
    # 訓練隨機森林 - 調整參數提高精確度
    model = RandomForestClassifier(
        n_estimators=200,  # 增加樹的數量
        max_depth=10,      # 限制深度避免過擬合
        min_samples_split=10,  # 增加分割最小樣本數
        min_samples_leaf=5,    # 增加葉節點最小樣本數
        random_state=42, 
        class_weight={0: 1, 1: 3}  # 增加正樣本權重
    )
    model.fit(X_train, y_train)
    
    return model, scaler, df_train.columns

def test_model(model, scaler, feature_names):
    """測試模型並繪製結果"""
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['atr'] = calculate_atr(df)
    
    # 測試9-10月份
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
            
            # 提取特徵
            features = extract_features(month_df, i)
            if features is None:
                continue
            
            # 預測
            feature_df = pd.DataFrame([features])
            feature_df = feature_df.reindex(columns=feature_names, fill_value=0)
            X_test = scaler.transform(feature_df.fillna(0))
            pred_prob = model.predict_proba(X_test)[0][1]
            
            # 檢查實際是否滿足條件
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
            timestamps.append(current_row['timestamp'])
            prices.append(current_price)
        
        # 繪製結果
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 上圖：價格和預測
        ax1.plot(timestamps, prices, 'b-', alpha=0.7, linewidth=1, label='BTC Price')
        
        # 標記高預測概率的點 - 提高閾值到0.85
        high_pred_indices = [i for i, p in enumerate(predictions) if p > 0.85]
        if high_pred_indices:
            high_pred_times = [timestamps[i] for i in high_pred_indices]
            high_pred_prices = [prices[i] for i in high_pred_indices]
            ax1.scatter(high_pred_times, high_pred_prices, color='orange', s=50, alpha=0.8, label='High Prediction (>0.85)')
        
        # 標記實際滿足條件的點
        actual_indices = [i for i, a in enumerate(actual_spikes) if a]
        if actual_indices:
            actual_times = [timestamps[i] for i in actual_indices]
            actual_prices = [prices[i] for i in actual_indices]
            ax1.scatter(actual_times, actual_prices, color='red', s=100, alpha=0.8, label='Actual Spikes')
        
        ax1.set_title(f'{month}月 ML Prediction vs Actual Spikes')
        ax1.set_ylabel('Price (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 下圖：預測概率
        ax2.plot(timestamps, predictions, 'g-', alpha=0.7, linewidth=1, label='Prediction Probability')
        ax2.axhline(y=0.85, color='orange', linestyle='--', alpha=0.7, label='Threshold (0.85)')
        ax2.set_title(f'{month}月 Prediction Probability')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 格式化x軸
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'ml_prediction_month_{month:02d}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 統計結果 - 使用0.85閾值
        y_true = actual_spikes
        y_pred_85 = [p > 0.85 for p in predictions]
        y_pred_90 = [p > 0.90 for p in predictions]
        y_pred_95 = [p > 0.95 for p in predictions]
        
        print(f"Actual spikes: {sum(y_true)}")
        
        # 測試不同閾值
        for threshold, y_pred, name in [(0.85, y_pred_85, "0.85"), (0.90, y_pred_90, "0.90"), (0.95, y_pred_95, "0.95")]:
            predicted_count = sum(y_pred)
            true_positives = sum(a and p for a, p in zip(y_true, y_pred))
            precision = true_positives / predicted_count if predicted_count > 0 else 0
            recall = true_positives / sum(y_true) if sum(y_true) > 0 else 0
            
            print(f"Threshold {name}: Predicted={predicted_count}, TP={true_positives}, Precision={precision:.1%}, Recall={recall:.1%}")

if __name__ == "__main__":
    print("Collecting training data...")
    training_data, labels = collect_training_data()
    
    print("Training model...")
    model, scaler, feature_names = train_model(training_data, labels)
    
    print("Testing model...")
    test_model(model, scaler, feature_names)
    
    print("Done!")
