#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
import joblib

plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df):
    df['rsi'] = calculate_rsi(df['close'])
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    df['price_change_1h'] = df['close'].pct_change(periods=4)
    df['rsi_change'] = df['rsi'].diff()
    df['price_change_30m'] = df['close'].pct_change(periods=2)
    return df

def create_supervised_dataset(df):
    """創建監督學習數據集"""
    features = []
    labels = []
    crash_indices = []
    
    for i in range(24, len(df) - 24):
        current = df.iloc[i]
        
        # 更嚴格的初步條件
        if (current['price_change_1h'] < -0.003 and  # 1小時跌>0.3%
            current['rsi_change'] < -2 and           # RSI下降>2點
            current['volume_ratio'] > 1.1):          # 成交量>1.1倍
            
            # 檢查未來6小時是否真的下跌
            future_6h = df.iloc[i+1:i+25]
            if len(future_6h) >= 24:
                current_price = current['close']
                lowest_price = future_6h['low'].min()
                drop_pct = (current_price - lowest_price) / current_price
                
                # 特徵向量
                feature_vector = [
                    current['rsi'],
                    current['price_change_1h'],
                    current['price_change_30m'],
                    current['volume_ratio'],
                    current['rsi_change'],
                    current['close'] / current['open'] - 1,  # 當前K線漲跌幅
                ]
                
                features.append(feature_vector)
                labels.append(1 if drop_pct > 0.015 else 0)  # 1.5%閾值
                crash_indices.append(i)
    
    return np.array(features), np.array(labels), crash_indices

def train_crash_model():
    """訓練下跌預測模型"""
    print("Loading training data...")
    df = pd.read_csv('training_data/btc_12months_15m.csv')
    df = add_technical_indicators(df)
    
    print("Creating supervised dataset...")
    X, y, indices = create_supervised_dataset(df)
    
    print(f"Training samples: {len(X)}")
    print(f"Positive samples: {y.sum()} ({y.mean()*100:.1f}%)")
    
    if len(X) < 50:
        print("Not enough samples")
        return None
    
    # 訓練模型
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X, y)
    
    # 保存模型
    joblib.dump(model, 'supervised_crash_model.pkl')
    print("Model saved")
    
    return model, df, indices, y

def visualize_supervised_results():
    """可視化監督學習結果"""
    model, df, indices, labels = train_crash_model()
    
    if model is None:
        return
    
    # 最近3個月數據
    recent_df = df.tail(8760)
    recent_start_idx = len(df) - 8760
    
    # 找出在最近3個月範圍內的訓練樣本
    recent_indices = [i for i in indices if i >= recent_start_idx]
    recent_labels = [labels[indices.index(i)] for i in recent_indices]
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    timestamps = pd.to_datetime(recent_df['timestamp'])
    prices = recent_df['close'].values
    
    # 創建顏色數組
    colors = ['lightgray'] * len(recent_df)
    
    for i, idx in enumerate(recent_indices):
        relative_idx = idx - recent_start_idx
        if recent_labels[i] == 1:
            colors[relative_idx] = 'red'
        else:
            colors[relative_idx] = 'green'
    
    # 分段繪製
    current_color = colors[0]
    start_idx = 0
    
    for i in range(1, len(colors)):
        if colors[i] != current_color or i == len(colors) - 1:
            end_idx = i if i != len(colors) - 1 else i
            
            segment_x = timestamps.iloc[start_idx:end_idx+1]
            segment_y = prices[start_idx:end_idx+1]
            
            if current_color == 'red':
                ax.plot(segment_x, segment_y, color='red', linewidth=2.5, alpha=0.9)
            elif current_color == 'green':
                ax.plot(segment_x, segment_y, color='green', linewidth=2, alpha=0.8)
            else:
                ax.plot(segment_x, segment_y, color='lightgray', linewidth=0.8, alpha=0.6)
            
            start_idx = i
            current_color = colors[i]
    
    ax.set_title('Supervised Crash Detection (Red: >1.5% drop, Green: <1.5% drop - Bottom Detection)', fontsize=14)
    ax.set_ylabel('Price (USDT)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 圖例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lightgray', lw=2, label='Normal'),
        Line2D([0], [0], color='red', lw=2, label='True Crash (>1.5%)'),
        Line2D([0], [0], color='green', lw=2, label='Bottom Detection (<1.5%)')
    ]
    ax.legend(handles=legend_elements, fontsize=11)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('supervised_crash_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 統計
    true_crashes = sum(recent_labels)
    false_signals = len(recent_labels) - true_crashes
    
    print(f"\n=== Supervised Learning Results ===")
    print(f"Total training samples in recent 3 months: {len(recent_labels)}")
    print(f"True crashes (red): {true_crashes}")
    print(f"Bottom detection (green): {false_signals}")
    print(f"Precision for crashes: {true_crashes/len(recent_labels)*100:.1f}%")
    print(f"Bottom detection rate: {false_signals/len(recent_labels)*100:.1f}%")

if __name__ == "__main__":
    visualize_supervised_results()
