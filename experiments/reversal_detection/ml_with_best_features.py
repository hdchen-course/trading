#!/usr/bin/env python3
"""
基於 Task 2 分析結果的 ML 訓練

使用從規則分析中發現的最重要特徵：
1. 價格位置特徵 (position_24h, position_48h)
2. RSI 極值
3. BB 位置
4. 波動率加速 (atr_2nd)
5. 成交量特徵
6. 二階導數背離
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb


def engineer_rule_based_features(df):
    """
    基於 Task 2 分析結果的特徵工程
    
    重點特徵：
    1. 價格位置 (最重要！)
    2. RSI 極值
    3. BB 位置
    4. 波動率加速
    5. 成交量
    """
    print("=" * 80)
    print("🔧 基於規則分析的特徵工程")
    print("=" * 80)
    
    features = pd.DataFrame()
    
    # === 1. 價格位置特徵（最重要！）===
    print("\n1️⃣  價格位置特徵（來自 analyze_price_position.py）...")
    
    # 24小時和48小時相對位置
    window_24h = 96
    window_48h = 192
    
    high_24h = df['close'].rolling(window=window_24h, min_periods=1).max()
    low_24h = df['close'].rolling(window=window_24h, min_periods=1).min()
    features['position_24h'] = (df['close'] - low_24h) / (high_24h - low_24h + 1e-10)
    
    high_48h = df['close'].rolling(window=window_48h, min_periods=1).max()
    low_48h = df['close'].rolling(window=window_48h, min_periods=1).min()
    features['position_48h'] = (df['close'] - low_48h) / (high_48h - low_48h + 1e-10)
    
    # 距離高低點的百分比
    features['dist_from_low_24h_pct'] = (df['close'] - low_24h) / (low_24h + 1e-10)
    features['dist_from_high_24h_pct'] = (df['close'] - high_24h) / (high_24h + 1e-10)
    
    # 極端位置標記
    features['extreme_low'] = (features['position_24h'] < 0.2).astype(int)
    features['extreme_high'] = (features['position_24h'] > 0.8).astype(int)
    
    # === 2. RSI 特徵 ===
    print("2️⃣  RSI 特徵...")
    
    if 'rsi' in df.columns:
        features['rsi'] = df['rsi']
        features['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        features['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        features['rsi_extreme_oversold'] = (df['rsi'] < 20).astype(int)
        features['rsi_extreme_overbought'] = (df['rsi'] > 80).astype(int)
    
    # === 3. BB 位置特徵 ===
    print("3️⃣  布林帶位置特徵...")
    
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        bb_range = df['bb_upper'] - df['bb_lower']
        features['bb_position'] = (df['close'] - df['bb_lower']) / (bb_range + 1e-10)
        features['bb_lower_touch'] = (features['bb_position'] < 0.2).astype(int)
        features['bb_upper_touch'] = (features['bb_position'] > 0.8).astype(int)
    
    # === 4. 波動率加速（ATR 二階導數）===
    print("4️⃣  波動率加速特徵...")
    
    if 'atr_2nd' in df.columns:
        features['atr_2nd'] = df['atr_2nd']
        features['volatility_accelerating'] = (df['atr_2nd'] > 0.5).astype(int)
    
    if 'atr' in df.columns:
        features['atr'] = df['atr']
        features['atr_normalized'] = df['atr'] / (df['close'] + 1e-10)
    
    # === 5. 成交量特徵 ===
    print("5️⃣  成交量特徵...")
    
    if 'volume' in df.columns:
        vol_ma = df['volume'].rolling(window=20, min_periods=1).mean()
        features['volume_ratio'] = df['volume'] / (vol_ma + 1e-10)
        features['high_volume'] = (features['volume_ratio'] > 1.5).astype(int)
    
    if 'vol_2nd' in df.columns:
        features['vol_2nd'] = df['vol_2nd']
    
    # === 6. 二階導數背離 ===
    print("6️⃣  二階導數背離特徵...")
    
    if 'price_2nd' in df.columns:
        features['price_2nd'] = df['price_2nd']
    
    if 'macd_2nd' in df.columns:
        features['macd_2nd'] = df['macd_2nd']
    
    # === 7. 組合特徵（基於最佳規則）===
    print("7️⃣  組合特徵（基於最佳規則）...")
    
    # RSI<30 AND BB<0.2 (F1=15.1%)
    if 'rsi' in features.columns and 'bb_position' in features.columns:
        features['rule_oversold_extreme'] = (
            (features['rsi'] < 30) & (features['bb_position'] < 0.2)
        ).astype(int)
    
    # RSI>70 AND BB>0.8 (F1=13.0%)
    if 'rsi' in features.columns and 'bb_position' in features.columns:
        features['rule_overbought_extreme'] = (
            (features['rsi'] > 70) & (features['bb_position'] > 0.8)
        ).astype(int)
    
    # 極端位置 + 波動加速 (F1=15.3%)
    if 'extreme_low' in features.columns and 'volatility_accelerating' in features.columns:
        features['rule_extreme_volatility'] = (
            ((features['extreme_low'] == 1) | (features['extreme_high'] == 1)) &
            (features['volatility_accelerating'] == 1)
        ).astype(int)
    
    # 價跌量增 (F1=6.7%)
    if 'price_2nd' in features.columns and 'vol_2nd' in features.columns:
        features['rule_price_vol_divergence'] = (
            (features['price_2nd'] < -0.5) & (features['vol_2nd'] > 0.5)
        ).astype(int)
    
    print(f"\n✅ 特徵工程完成！")
    print(f"   總特徵數: {len(features.columns)}")
    print(f"   特徵列表: {list(features.columns)[:10]}... (顯示前10個)")
    
    return features


def train_with_best_features(data_path):
    """
    使用最佳特徵訓練 ML 模型
    """
    print("=" * 80)
    print("🚀 基於規則分析的 ML 訓練")
    print("=" * 80)
    
    # 載入數據
    df = pd.read_csv(data_path)
    print(f"\n✅ 數據載入完成: {len(df)} 筆資料, {df['reverse'].sum()} 個反轉點")
    
    # 特徵工程
    features = engineer_rule_based_features(df)
    
    # 準備數據
    X = features
    y = df['reverse'].astype(int)
    
    # 清理數據
    mask = ~(X.isnull().any(axis=1) | np.isinf(X).any(axis=1))
    X = X[mask]
    y = y[mask]
    
    print(f"\n📊 數據清理:")
    print(f"   清理後: {len(X)} 筆")
    print(f"   反轉點: {y.sum()} 個 ({y.sum()/len(y)*100:.2f}%)")
    
    # 時間序列交叉驗證
    print("\n" + "=" * 80)
    print("📊 時間序列交叉驗證")
    print("=" * 80)
    
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    rf_results = []
    xgb_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        print(f"\nFold {fold}:")
        print(f"  訓練集: {len(X_train)} 筆 (反轉點: {y_train.sum()})")
        print(f"  測試集: {len(X_test)} 筆 (反轉點: {y_test.sum()})")
        
        if y_train.sum() < 2:
            print(f"  ⚠️  訓練集反轉點太少，跳過此 fold")
            continue
        
        # === Random Forest ===
        print(f"\n🔧 訓練 RandomForest...")
        
        # SMOTE
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"   SMOTE 重採樣: {len(X_train_balanced)} 筆 (反轉點: {y_train_balanced.sum()})")
        except:
            X_train_balanced, y_train_balanced = X_train, y_train
            print(f"   ⚠️  SMOTE 失敗，使用原始數據")
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train_balanced, y_train_balanced)
        y_pred_rf = rf.predict(X_test)
        
        # 評估
        precision_rf = precision_score(y_test, y_pred_rf, zero_division=0)
        recall_rf = recall_score(y_test, y_pred_rf, zero_division=0)
        f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
        
        print(f"\nFold {fold} RandomForest 結果:")
        print(f"  精確度: {precision_rf*100:.2f}%")
        print(f"  召回率: {recall_rf*100:.2f}%")
        print(f"  F1分數: {f1_rf:.4f}")
        
        if precision_rf > 0 or recall_rf > 0:
            cm = confusion_matrix(y_test, y_pred_rf)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        
        rf_results.append({
            'precision': precision_rf,
            'recall': recall_rf,
            'f1': f1_rf
        })
        
        # === XGBoost ===
        print(f"\n🔧 訓練 XGBoost...")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=len(y_train_balanced[y_train_balanced==0]) / max(len(y_train_balanced[y_train_balanced==1]), 1)
        )
        xgb_model.fit(X_train_balanced, y_train_balanced)
        y_pred_xgb = xgb_model.predict(X_test)
        
        # 評估
        precision_xgb = precision_score(y_test, y_pred_xgb, zero_division=0)
        recall_xgb = recall_score(y_test, y_pred_xgb, zero_division=0)
        f1_xgb = f1_score(y_test, y_pred_xgb, zero_division=0)
        
        print(f"\nFold {fold} XGBoost 結果:")
        print(f"  精確度: {precision_xgb*100:.2f}%")
        print(f"  召回率: {recall_xgb*100:.2f}%")
        print(f"  F1分數: {f1_xgb:.4f}")
        
        if precision_xgb > 0 or recall_xgb > 0:
            cm = confusion_matrix(y_test, y_pred_xgb)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        
        xgb_results.append({
            'precision': precision_xgb,
            'recall': recall_xgb,
            'f1': f1_xgb
        })
    
    # 計算平均結果
    print("\n" + "=" * 80)
    print("📊 平均結果")
    print("=" * 80)
    
    if rf_results:
        rf_avg = {
            'precision': np.mean([r['precision'] for r in rf_results]),
            'recall': np.mean([r['recall'] for r in rf_results]),
            'f1': np.mean([r['f1'] for r in rf_results])
        }
        
        xgb_avg = {
            'precision': np.mean([r['precision'] for r in xgb_results]),
            'recall': np.mean([r['recall'] for r in xgb_results]),
            'f1': np.mean([r['f1'] for r in xgb_results])
        }
        
        print(f"\n📊 RandomForest (基於規則特徵):")
        print(f"  精確度: {rf_avg['precision']*100:.2f}%")
        print(f"  召回率: {rf_avg['recall']*100:.2f}%")
        print(f"  F1分數: {rf_avg['f1']:.4f}")
        
        print(f"\n📊 XGBoost (基於規則特徵):")
        print(f"  精確度: {xgb_avg['precision']*100:.2f}%")
        print(f"  召回率: {xgb_avg['recall']*100:.2f}%")
        print(f"  F1分數: {xgb_avg['f1']:.4f}")
        
        # 特徵重要性
        print("\n" + "=" * 80)
        print("🎯 特徵重要性分析")
        print("=" * 80)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'rf_importance': rf.feature_importances_,
            'xgb_importance': xgb_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('xgb_importance', ascending=False)
        
        print(f"\n🎯 Top 15 重要特徵 (XGBoost):")
        print(feature_importance.head(15).to_string(index=False))
        
        # 與規則方法比較
        print("\n" + "=" * 80)
        print("📊 與規則方法比較")
        print("=" * 80)
        
        print(f"\n方法                              精確度        召回率       F1分數")
        print("-" * 70)
        print(f"最佳規則 (RSI<30 AND BB<0.2)      16.50%     13.90%    0.1510")
        print(f"最佳規則 (極端位置+波動加速)        10.60%     27.50%    0.1530")
        print(f"RandomForest (規則特徵)           {rf_avg['precision']*100:>6.2f}%    {rf_avg['recall']*100:>6.2f}%    {rf_avg['f1']:.4f}")
        print(f"XGBoost (規則特徵)                {xgb_avg['precision']*100:>6.2f}%    {xgb_avg['recall']*100:>6.2f}%    {xgb_avg['f1']:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ 訓練完成！")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='基於規則分析的 ML 訓練')
    parser.add_argument('--data', type=str, 
                        default='training_data/reversal_training_data_2022-01-01_to_2025-09-01_15m.csv',
                        help='訓練數據路徑')
    
    args = parser.parse_args()
    
    train_with_best_features(args.data)
