#!/usr/bin/env python3
"""
優化的特徵工程 - 只使用最重要的特徵

基於特徵重要性分析，選擇最有效的特徵：
1. 價格位置特徵 (position_24h, dist_from_low_24h_pct)
2. 波動性特徵 (atr, volatility)
3. 組合特徵 (rsi_bb_product)
4. 趨勢特徵 (rsi, macd)
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


def engineer_optimized_features(df):
    """
    優化的特徵工程 - 只使用最重要的特徵
    """
    print("=" * 80)
    print("🔧 優化特徵工程")
    print("=" * 80)
    
    features = pd.DataFrame()
    
    # === 1. 核心價格位置特徵 ===
    print("\n1️⃣  價格位置特徵...")
    
    # 24小時相對位置
    high_24h = df['close'].rolling(window=96, min_periods=1).max()
    low_24h = df['close'].rolling(window=96, min_periods=1).min()
    features['position_24h'] = (df['close'] - low_24h) / (high_24h - low_24h + 1e-10)
    
    # 距離低點的百分比距離
    features['dist_from_low_24h_pct'] = (df['close'] - low_24h) / (low_24h + 1e-10)
    
    # 距離高點的百分比距離
    features['dist_from_high_24h_pct'] = (high_24h - df['close']) / (high_24h + 1e-10)
    
    # 48小時相對位置
    high_48h = df['close'].rolling(window=192, min_periods=1).max()
    low_48h = df['close'].rolling(window=192, min_periods=1).min()
    features['position_48h'] = (df['close'] - low_48h) / (high_48h - low_48h + 1e-10)
    
    # === 2. 波動性特徵 ===
    print("2️⃣  波動性特徵...")
    
    # ATR (已經在 df 中)
    if 'atr' in df.columns:
        features['atr'] = df['atr']
    
    # 波動率
    if 'volatility' in df.columns:
        features['volatility'] = df['volatility']
    else:
        features['volatility'] = df['close'].pct_change().rolling(window=20).std()
    
    # === 3. 技術指標 ===
    print("3️⃣  技術指標...")
    
    # RSI
    if 'rsi' in df.columns:
        features['rsi'] = df['rsi']
    
    # MACD
    if 'macd' in df.columns:
        features['macd'] = df['macd']
    
    # 布林帶位置
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        bb_range = df['bb_upper'] - df['bb_lower']
        features['bb_position'] = (df['close'] - df['bb_lower']) / (bb_range + 1e-10)
    
    # === 4. 組合特徵 ===
    print("4️⃣  組合特徵...")
    
    # RSI * BB位置 (重要組合特徵)
    if 'rsi' in features.columns and 'bb_position' in features.columns:
        features['rsi_bb_product'] = features['rsi'] * features['bb_position']
    
    # 價格變化率
    features['price_change_5'] = df['close'].pct_change(5)
    features['price_change_10'] = df['close'].pct_change(10)
    
    # === 5. 二階導數特徵（如果存在）===
    print("5️⃣  二階導數特徵...")
    
    if 'price_2nd' in df.columns:
        features['price_2nd'] = df['price_2nd']
    if 'macd_2nd' in df.columns:
        features['macd_2nd'] = df['macd_2nd']
    if 'atr_2nd' in df.columns:
        features['atr_2nd'] = df['atr_2nd']
    
    print(f"\n✅ 優化特徵工程完成！")
    print(f"   總特徵數: {len(features.columns)}")
    print(f"   特徵列表: {list(features.columns)}")
    
    return features


def train_optimized_models(data_path):
    """
    使用優化特徵訓練模型
    """
    print("=" * 80)
    print("🚀 優化 ML 模型訓練")
    print("=" * 80)
    
    # 載入數據
    df = pd.read_csv(data_path)
    print(f"\n✅ 數據載入完成: {len(df)} 筆資料, {df['reverse'].sum()} 個反轉點")
    
    # 特徵工程
    features = engineer_optimized_features(df)
    
    # 合併特徵和標籤
    X = features
    y = df['reverse'].astype(int)
    
    # 清理數據
    mask = ~(X.isnull().any(axis=1) | np.isinf(X).any(axis=1))
    X = X[mask]
    y = y[mask]
    
    print(f"\n📊 數據清理:")
    print(f"   原始數據: {len(df)} 筆")
    print(f"   清理後: {len(X)} 筆")
    print(f"   反轉點: {y.sum()} 個 ({y.sum()/len(y)*100:.2f}%)")
    
    # 時間序列交叉驗證
    print("\n" + "=" * 80)
    print("📊 時間序列交叉驗證")
    print("=" * 80)
    
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    print(f"\n使用 {n_splits} 折交叉驗證")
    print(f"總數據: {len(X)} 筆\n")
    
    # 儲存結果
    rf_results = []
    xgb_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        print(f"Fold {fold}:")
        print(f"  訓練集: {len(X_train)} 筆 (反轉點: {y_train.sum()})")
        print(f"  測試集: {len(X_test)} 筆 (反轉點: {y_test.sum()})\n")
        
        # === Random Forest ===
        print(f"🔧 訓練 RandomForest (優化特徵)...")
        print(f"   訓練數據: {len(X_train)} 筆")
        print(f"   反轉點: {y_train.sum()} 個 ({y_train.sum()/len(y_train)*100:.2f}%)")
        
        # SMOTE
        if y_train.sum() >= 2:
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                print(f"   使用 SMOTE 處理類別不平衡...")
                print(f"   重採樣後: {len(X_train_balanced)} 筆")
                print(f"   反轉點: {y_train_balanced.sum()} 個 ({y_train_balanced.sum()/len(y_train_balanced)*100:.2f}%)")
            except:
                X_train_balanced, y_train_balanced = X_train, y_train
                print(f"   ⚠️  SMOTE 失敗，使用原始數據")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # 訓練
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train_balanced, y_train_balanced)
        print(f"   ✅ 訓練完成\n")
        
        # 預測
        y_pred_rf = rf.predict(X_test)
        
        # 評估
        precision_rf = precision_score(y_test, y_pred_rf, zero_division=0)
        recall_rf = recall_score(y_test, y_pred_rf, zero_division=0)
        f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        
        print(f"Fold {fold} RandomForest 結果:")
        print(f"  精確度: {precision_rf*100:.2f}%")
        print(f"  召回率: {recall_rf*100:.2f}%")
        print(f"  F1分數: {f1_rf:.4f}")
        if cm_rf.shape == (2, 2):
            tn, fp, fn, tp = cm_rf.ravel()
            print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")
        
        rf_results.append({
            'precision': precision_rf,
            'recall': recall_rf,
            'f1': f1_rf
        })
        
        # === XGBoost ===
        print(f"🔧 訓練 XGBoost (優化特徵)...")
        
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
        print(f"   ✅ 訓練完成\n")
        
        # 預測
        y_pred_xgb = xgb_model.predict(X_test)
        
        # 評估
        precision_xgb = precision_score(y_test, y_pred_xgb, zero_division=0)
        recall_xgb = recall_score(y_test, y_pred_xgb, zero_division=0)
        f1_xgb = f1_score(y_test, y_pred_xgb, zero_division=0)
        cm_xgb = confusion_matrix(y_test, y_pred_xgb)
        
        print(f"Fold {fold} XGBoost 結果:")
        print(f"  精確度: {precision_xgb*100:.2f}%")
        print(f"  召回率: {recall_xgb*100:.2f}%")
        print(f"  F1分數: {f1_xgb:.4f}")
        if cm_xgb.shape == (2, 2):
            tn, fp, fn, tp = cm_xgb.ravel()
            print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")
        
        xgb_results.append({
            'precision': precision_xgb,
            'recall': recall_xgb,
            'f1': f1_xgb
        })
    
    # 計算平均結果
    print("=" * 80)
    print("📊 優化模型平均結果")
    print("=" * 80)
    
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
    
    print(f"\n📊 RandomForest (優化特徵) 平均結果:")
    print(f"  精確度: {rf_avg['precision']*100:.2f}%")
    print(f"  召回率: {rf_avg['recall']*100:.2f}%")
    print(f"  F1分數: {rf_avg['f1']:.4f}")
    
    print(f"\n📊 XGBoost (優化特徵) 平均結果:")
    print(f"  精確度: {xgb_avg['precision']*100:.2f}%")
    print(f"  召回率: {xgb_avg['recall']*100:.2f}%")
    print(f"  F1分數: {xgb_avg['f1']:.4f}")
    
    # 特徵重要性（使用最後一個模型）
    print("\n" + "=" * 80)
    print("🎯 特徵重要性分析")
    print("=" * 80)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'rf_importance': rf.feature_importances_,
        'xgb_importance': xgb_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('xgb_importance', ascending=False)
    
    print(f"\n🎯 Top 10 重要特徵 (XGBoost):")
    print(feature_importance.head(10).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("✅ 優化模型訓練完成！")
    print("=" * 80)
    
    return rf_avg, xgb_avg


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='優化特徵的 ML 模型訓練')
    parser.add_argument('--data', type=str, 
                        default='training_data/reversal_training_data_2022-01-01_to_2025-09-01_15m.csv',
                        help='訓練數據路徑')
    
    args = parser.parse_args()
    
    train_optimized_models(args.data)
