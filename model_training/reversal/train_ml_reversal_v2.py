#!/usr/bin/env python3
"""
ML Reversal V2 模型訓練腳本 - 右側入場交易

訓練 MLReversalStrategy 使用的反轉檢測模型
使用 RandomForest 分類器來預測「反轉確認後的入場時機」

🎯 訓練目標：
- 不是預測反轉點本身（事後才知道）
- 而是預測「反轉點後 1-2 根K線」是否適合入場
- 這樣模型學習的是「看到反轉確認信號後的入場時機」

輸入數據：
- 歷史價格數據 (OHLCV)
- 真實反轉點標籤（由 Peak Valley 事後分析得出）
- 右側入場點標籤（反轉點後 1-2 根K線）

輸出模型：
- model_training/output/reversal/reversal_model_v2.pkl

分類：
- A_turn: 上升反轉後的買入入場點
- V_turn: 下降反轉後的賣出入場點
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import sys
import os

# 添加路徑以便導入 utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.extremes import find_local_extremes


def extract_features_with_context(df, entry_idx, reversal_idx):
    """
    提取 [t] 和 [t+1] 或 [t] 和 [t+2] 兩個時間點的特徵組合
    
    Args:
        df: 原始數據 DataFrame（已包含所有技術指標和二階導數）
        entry_idx: 入場點索引 [t+1] 或 [t+2]
        reversal_idx: 反轉點索引 [t]
    
    Returns:
        特徵字典
    """
    if entry_idx < 10 or reversal_idx < 10 or entry_idx >= len(df):
        return None
    
    features = {}
    
    # ============================================================
    # 1. [t] 反轉點的所有特徵
    # ============================================================
    data_t = df.iloc[reversal_idx]
    features.update({
        'close_t': data_t['close'],
        'volume_t': data_t['volume'],
        'rsi_t': data_t['rsi'],
        'macd_t': data_t['macd'],
        'macd_histogram_t': data_t['macd_histogram'],
        'bb_position_t': data_t['bb_position'],
        'volume_ratio_t': data_t['volume_ratio'],
        'price_2nd_t': data_t['price_2nd'],
        'vol_2nd_t': data_t['vol_2nd'],
        'macd_2nd_t': data_t['macd_2nd'],
        'atr_2nd_t': data_t['atr_2nd'],
    })
    
    # ============================================================
    # 2. [t+1] 或 [t+2] 入場點的所有特徵
    # ============================================================
    data_entry = df.iloc[entry_idx]
    features.update({
        'close_entry': data_entry['close'],
        'volume_entry': data_entry['volume'],
        'rsi_entry': data_entry['rsi'],
        'macd_entry': data_entry['macd'],
        'macd_histogram_entry': data_entry['macd_histogram'],
        'bb_position_entry': data_entry['bb_position'],
        'volume_ratio_entry': data_entry['volume_ratio'],
        'price_2nd_entry': data_entry['price_2nd'],
        'vol_2nd_entry': data_entry['vol_2nd'],
        'macd_2nd_entry': data_entry['macd_2nd'],
        'atr_2nd_entry': data_entry['atr_2nd'],
    })
    
    # ============================================================
    # 3. [t] 到 [t+1/t+2] 的變化特徵
    # ============================================================
    features.update({
        'price_change': (data_entry['close'] - data_t['close']) / data_t['close'],
        'volume_change': data_entry['volume'] / data_t['volume'] if data_t['volume'] > 0 else 1.0,
        'rsi_change': data_entry['rsi'] - data_t['rsi'],
        'macd_change': data_entry['macd'] - data_t['macd'],
        'bb_position_change': data_entry['bb_position'] - data_t['bb_position'],
    })
    
    # ============================================================
    # 4. 二階導數符號改變特徵
    # ============================================================
    # 個別符號改變
    price_2nd_changed = int(np.sign(data_t['price_2nd']) != np.sign(data_entry['price_2nd']))
    vol_2nd_changed = int(np.sign(data_t['vol_2nd']) != np.sign(data_entry['vol_2nd']))
    macd_2nd_changed = int(np.sign(data_t['macd_2nd']) != np.sign(data_entry['macd_2nd']))
    atr_2nd_changed = int(np.sign(data_t['atr_2nd']) != np.sign(data_entry['atr_2nd']))
    
    features.update({
        'price_2nd_sign_changed': price_2nd_changed,
        'vol_2nd_sign_changed': vol_2nd_changed,
        'macd_2nd_sign_changed': macd_2nd_changed,
        'atr_2nd_sign_changed': atr_2nd_changed,
        
        # C(4,2) 組合
        'price_vol_confirmation': int(price_2nd_changed and vol_2nd_changed),
        'price_macd_confirmation': int(price_2nd_changed and macd_2nd_changed),
        'price_atr_confirmation': int(price_2nd_changed and atr_2nd_changed),
        'vol_macd_confirmation': int(vol_2nd_changed and macd_2nd_changed),
        'vol_atr_confirmation': int(vol_2nd_changed and atr_2nd_changed),
        'macd_atr_confirmation': int(macd_2nd_changed and atr_2nd_changed),
        
        # C(4,3) 組合
        'price_vol_macd_confirmation': int(price_2nd_changed and vol_2nd_changed and macd_2nd_changed),
        'price_vol_atr_confirmation': int(price_2nd_changed and vol_2nd_changed and atr_2nd_changed),
        'price_macd_atr_confirmation': int(price_2nd_changed and macd_2nd_changed and atr_2nd_changed),
        'vol_macd_atr_confirmation': int(vol_2nd_changed and macd_2nd_changed and atr_2nd_changed),
        
        # C(4,4) 組合
        'quad_confirmation': int(price_2nd_changed and vol_2nd_changed and macd_2nd_changed and atr_2nd_changed),
    })
    
    # ============================================================
    # 5. 動量和波動率特徵（在入場點計算）
    # ============================================================
    if entry_idx >= 10:
        features.update({
            'price_momentum_5': (df.iloc[entry_idx]['close'] - df.iloc[entry_idx - 5]['close']) / df.iloc[entry_idx - 5]['close'],
            'price_momentum_10': (df.iloc[entry_idx]['close'] - df.iloc[entry_idx - 10]['close']) / df.iloc[entry_idx - 10]['close'],
            'volatility_10': df.iloc[entry_idx - 10:entry_idx]['close'].std() / df.iloc[entry_idx - 10:entry_idx]['close'].mean(),
            'volume_trend': np.polyfit(range(10), df.iloc[entry_idx - 10:entry_idx]['volume'].values, 1)[0],
        })
    
    return features


def main(data_file=None):
    """
    主訓練流程
    
    Args:
        data_file: 訓練數據檔案路徑，默認使用最新的 2023-2025 數據
    """
    print("=" * 80)
    print("🤖 ML Reversal V2 模型訓練")
    print("=" * 80)
    
    # 數據來源: 使用 generate_training_data.py 生成的數據
    # 指令: python main.py --generate-data --start 2023-01-01 --end 2025-01-01 --interval 15m --window 16
    
    # 讀取數據
    print("\n📊 讀取訓練數據...")
    
    if data_file is None:
        # 默認使用 2023-2025 兩年份數據
        data_file = "training_data/reversal_training_data_2024-01-01_to_2025-01-01_15m.csv"
    
    print(f"數據檔案: {data_file}")
    print("\n💡 如果檔案不存在，請先執行:")
    print("   python main.py --generate-data --start 2023-01-01 --end 2025-01-01 --interval 15m --window 16")
    
    try:
        df = pd.read_csv(data_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        print(f"\n✅ 成功讀取數據: {len(df)} 行")
        print(f"   時間範圍: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        
        # 檢查必要欄位
        required_columns = ['timestamp', 'close', 'volume', 'reverse']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"\n❌ 缺少必要欄位: {missing_columns}")
            print(f"   可用欄位: {list(df.columns)}")
            return
        
        # 統計反轉點
        reversal_count = df['reverse'].sum()
        reversal_pct = reversal_count / len(df) * 100
        
        print(f"   反轉點數量: {reversal_count} 個 ({reversal_pct:.2f}%)")
        
    except FileNotFoundError as e:
        print(f"\n❌ 找不到數據檔案: {e}")
        print("\n💡 請先生成訓練數據:")
        print("   python main.py --generate-data --start 2023-01-01 --end 2025-01-01 --interval 15m --window 16")
        return
    except Exception as e:
        print(f"\n❌ 讀取數據時發生錯誤: {e}")
        return

    # 檢查技術指標欄位
    print("\n🔧 檢查並計算技術指標...")
    
    required_indicators = ['rsi', 'macd', 'macd_histogram', 'bb_position', 'volume', 'volume_ratio']
    missing_indicators = [col for col in required_indicators if col not in df.columns]
    
    if missing_indicators:
        print(f"   ❌ 缺少技術指標: {missing_indicators}")
        print(f"\n💡 請重新生成訓練數據（已更新 utils/indicators.py）:")
        print("   python main.py --generate-data --start 2023-01-01 --end 2025-01-01 --interval 15m --window 16")
        return
    
    print("   ✅ 數據已包含技術指標")
    
    # 檢查二階導數特徵
    print("\n🔧 檢查二階導數特徵...")
    required_2nd_features = ['price_2nd', 'vol_2nd', 'macd_2nd', 'atr_2nd']
    missing_2nd = [col for col in required_2nd_features if col not in df.columns]
    
    if missing_2nd:
        print(f"   ❌ 缺少二階導數特徵: {missing_2nd}")
        print(f"\n💡 請使用包含二階導數的訓練數據:")
        print("   training_data/reversal_training_data_2024_with_lis_lds.csv")
        return
    
    print("   ✅ 數據已包含二階導數特徵")
    
    # 使用 find_local_extremes 找出候選反轉點（只看過去數據）
    print("\n🎯 尋找候選反轉點...")
    prices = df['close'].values
    extremes, peaks, valleys = find_local_extremes(prices, window=10, prominence_factor=0)
    
    print(f"   找到局部極值: {len(extremes)} 個")
    print(f"   峰值: {len(peaks)} 個")
    print(f"   谷值: {len(valleys)} 個")
    
    # 找出真實反轉點（從訓練數據）
    true_reversal_indices = set(df[df['reverse'] == True].index.tolist())
    print(f"   真實反轉點: {len(true_reversal_indices)} 個")
    
    # 生成訓練樣本
    print("\n📊 生成訓練樣本...")
    print("   包含: 正樣本（入場點）+ 負樣本（非入場點）")
    
    samples = []
    
    # ============================================================
    # 1. 生成正樣本（真實反轉點的入場點）
    # ============================================================
    for reversal_idx in true_reversal_indices:
        if reversal_idx < 10 or reversal_idx >= len(df) - 2:
            continue
        
        # 判斷是 A_turn 還是 V_turn
        price_before = df.iloc[reversal_idx - 5:reversal_idx]['close'].mean()
        price_at = df.iloc[reversal_idx]['close']
        turn_type = 1 if price_at < price_before else 2  # 1=A_turn, 2=V_turn
        
        # [t+1] 入場點
        entry_idx_1 = reversal_idx + 1
        if entry_idx_1 < len(df):
            features_1 = extract_features_with_context(df, entry_idx_1, reversal_idx)
            if features_1 is not None:
                features_1['label'] = turn_type
                features_1['type'] = 'entry'
                features_1['timestamp'] = df.iloc[entry_idx_1]['timestamp']
                samples.append(features_1)
        
        # [t+2] 入場點
        entry_idx_2 = reversal_idx + 2
        if entry_idx_2 < len(df):
            features_2 = extract_features_with_context(df, entry_idx_2, reversal_idx)
            if features_2 is not None:
                features_2['label'] = turn_type
                features_2['type'] = 'entry'
                features_2['timestamp'] = df.iloc[entry_idx_2]['timestamp']
                samples.append(features_2)
    
    num_positive = len(samples)
    print(f"   正樣本（入場點）: {num_positive} 個")
    
    # ============================================================
    # 2. 生成負樣本（非入場點）
    # ============================================================
    # 排除反轉點及其附近的位置
    excluded_indices = set()
    for reversal_idx in true_reversal_indices:
        for offset in range(-2, 3):  # 前後 2 個位置都排除
            excluded_indices.add(reversal_idx + offset)
    
    # 從局部極值中選擇負樣本（假性反轉點）
    negative_candidates = [
        idx for idx in extremes
        if idx not in true_reversal_indices and idx not in excluded_indices
        and idx >= 10 and idx < len(df) - 2
    ]
    
    # 隨機選擇負樣本（與正樣本 1:1 比例）
    num_negative = min(num_positive, len(negative_candidates))
    negative_indices = np.random.choice(negative_candidates, size=num_negative, replace=False)
    
    for neg_idx in negative_indices:
        # 對負樣本，我們檢查 [t+1] 位置
        entry_idx = neg_idx + 1
        if entry_idx < len(df):
            features = extract_features_with_context(df, entry_idx, neg_idx)
            if features is not None:
                features['label'] = 0  # 非入場點
                features['type'] = 'non_entry'
                features['timestamp'] = df.iloc[entry_idx]['timestamp']
                samples.append(features)
    
    print(f"   負樣本（非入場點）: {len(samples) - num_positive} 個")
    print(f"   總樣本數: {len(samples)} 個")
    
    # 轉換為 DataFrame
    reversal_features = pd.DataFrame(samples)

    # 準備訓練數據 - 只使用 Top 20 重要特徵
    feature_columns = [
        'bb_position_change',      # 1. 11.5%
        'price_change',            # 2. 11.4%
        'rsi_change',              # 3. 10.6%
        'bb_position_t',           # 4. 7.5%
        'macd_histogram_t',        # 5. 6.6%
        'rsi_t',                   # 6. 6.3%
        'bb_position_entry',       # 7. 3.9%
        'macd_histogram_entry',    # 8. 3.8%
        'volatility_10',           # 9. 2.9%
        'rsi_entry',               # 10. 2.9%
        'macd_t',                  # 11. 2.7%
        'volume_t',                # 12. 2.4%
        'macd_entry',              # 13. 2.4%
        'macd_2nd_t',              # 14. 2.2%
        'price_2nd_entry',         # 15. 2.2%
        'price_momentum_10',       # 16. 2.1%
        'macd_change',             # 17. 2.0%
        'price_momentum_5',        # 18. 1.7%
        'volume_change',           # 19. 1.4%
        'macd_2nd_entry',          # 20. 1.3%
    ]
    
    print(f"\n✅ 特徵數量: {len(feature_columns)} 個（Top 20 重要特徵）")
    print(f"   Top 5: bb_position_change, price_change, rsi_change, bb_position_t, macd_histogram_t")
    print(f"   累積重要性: 前 10 個佔 68.8%，前 20 個佔 83.2%")

    # 移除缺失值
    clean_features = reversal_features.dropna(subset=feature_columns + ["label"])
    print(f"✅ 清理後特徵: {len(clean_features)} 個")

    if len(clean_features) < 20:
        print("❌ 特徵數量不足，無法訓練模型")
        return

    X = clean_features[feature_columns].values
    y = clean_features["label"].values

    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n📊 數據分割:")
    print(f"   訓練集: {len(X_train)} 個")
    print(f"   測試集: {len(X_test)} 個")
    print(f"\n標籤分佈:")
    print(f"   非入場點 (0): {sum(y == 0)} 個")
    print(f"   A轉入場 (1): {sum(y == 1)} 個")
    print(f"   V轉入場 (2): {sum(y == 2)} 個")

    # 訓練隨機森林模型
    print("\n🤖 訓練隨機森林模型...")
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    rf_model.fit(X_train, y_train)

    # 預測和評估
    y_pred = rf_model.predict(X_test)
    accuracy = rf_model.score(X_test, y_test)

    print(f"\n📈 模型性能:")
    print(f"   準確率: {accuracy:.3f}")
    print(f"\n分類報告:")
    
    # 根據標籤數量決定 target_names
    unique_labels = np.unique(y_test)
    if len(unique_labels) == 3:
        target_names = ["Non-entry", "A_turn", "V_turn"]
    else:
        target_names = [f"Class_{i}" for i in unique_labels]
    
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 特徵重要性
    feature_importance = pd.DataFrame(
        {"feature": feature_columns, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(f"\n🎯 特徵重要性:")
    for _, row in feature_importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")

    # 保存模型和特徵
    import os
    os.makedirs("models/reversal", exist_ok=True)
    
    model_path = "models/reversal/reversal_model_v2.pkl"
    features_path = "models/reversal/reversal_features_v2.csv"
    
    joblib.dump(rf_model, model_path)
    clean_features.to_csv(features_path, index=False)

    print(f"\n✅ 模型已保存: {model_path}")
    print(f"✅ 特徵已保存: {features_path}")
    
    print("\n" + "=" * 80)
    print("✅ 訓練完成！")
    print("=" * 80)
    print("\n💡 下一步:")
    print("   1. 檢查模型性能指標")
    print("   2. 在 strategies/ml/reversal.py 中使用此模型")
    print("   3. 進行回測驗證")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='訓練 ML Reversal V2 模型')
    parser.add_argument('--data', type=str, 
                       help='訓練數據檔案路徑 (默認: training_data/reversal_training_data_2023-01-01_to_2025-01-01_15m.csv)')
    
    args = parser.parse_args()
    
    main(data_file=args.data)
