#!/usr/bin/env python3
"""
集成方法 - 結合規則方法和 ML 預測

策略：
1. 使用規則方法（AdaptiveThreeLayer）作為初步篩選
2. 使用 ML 模型對規則方法的候選點進行二次驗證
3. 只有同時通過規則和 ML 驗證的點才被標記為反轉點
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


def load_rule_based_results():
    """
    載入規則方法的結果
    
    這裡我們使用之前分析的最佳規則方法：
    - AdaptiveThreeLayer: F1=0.1167, Precision=8.64%, Recall=17.95%
    """
    # 這是一個示例，實際應該從之前的實驗結果載入
    return {
        'name': 'AdaptiveThreeLayer',
        'precision': 0.0864,
        'recall': 0.1795,
        'f1': 0.1167
    }


def engineer_features_for_ensemble(df):
    """
    為集成方法設計特徵
    
    重點：
    1. 價格位置特徵（最重要）
    2. 波動性特徵
    3. 二階導數特徵
    """
    features = pd.DataFrame()
    
    # 價格位置特徵
    high_24h = df['close'].rolling(window=96, min_periods=1).max()
    low_24h = df['close'].rolling(window=96, min_periods=1).min()
    features['position_24h'] = (df['close'] - low_24h) / (high_24h - low_24h + 1e-10)
    features['dist_from_low_24h_pct'] = (df['close'] - low_24h) / (low_24h + 1e-10)
    
    high_48h = df['close'].rolling(window=192, min_periods=1).max()
    low_48h = df['close'].rolling(window=192, min_periods=1).min()
    features['position_48h'] = (df['close'] - low_48h) / (high_48h - low_48h + 1e-10)
    
    # 波動性特徵
    if 'atr' in df.columns:
        features['atr'] = df['atr']
        features['atr_normalized'] = df['atr'] / df['close']
    
    if 'volatility' in df.columns:
        features['volatility'] = df['volatility']
    
    # 技術指標
    if 'rsi' in df.columns:
        features['rsi'] = df['rsi']
        features['rsi_extreme'] = ((df['rsi'] < 30) | (df['rsi'] > 70)).astype(int)
    
    if 'macd' in df.columns:
        features['macd'] = df['macd']
    
    # 二階導數
    if 'price_2nd' in df.columns:
        features['price_2nd'] = df['price_2nd']
        features['price_2nd_abs'] = np.abs(df['price_2nd'])
    
    if 'macd_2nd' in df.columns:
        features['macd_2nd'] = df['macd_2nd']
    
    if 'atr_2nd' in df.columns:
        features['atr_2nd'] = df['atr_2nd']
    
    # 價格變化
    features['price_change_5'] = df['close'].pct_change(5)
    features['price_change_10'] = df['close'].pct_change(10)
    
    return features


def simulate_rule_based_predictions(y_true, precision=0.0864, recall=0.1795):
    """
    模擬規則方法的預測結果
    
    基於給定的精確度和召回率，生成模擬的規則方法預測
    """
    n_samples = len(y_true)
    n_positives = y_true.sum()
    
    # 計算需要預測為正例的數量
    # recall = TP / (TP + FN)
    # TP = recall * (TP + FN) = recall * n_positives
    tp = int(recall * n_positives)
    
    # precision = TP / (TP + FP)
    # TP + FP = TP / precision
    predicted_positives = int(tp / precision) if precision > 0 else 0
    fp = predicted_positives - tp
    
    # 創建預測數組
    y_pred = np.zeros(n_samples, dtype=int)
    
    # 找出真正的正例位置
    true_positive_indices = np.where(y_true == 1)[0]
    
    # 隨機選擇 tp 個真正的正例
    if len(true_positive_indices) > 0 and tp > 0:
        selected_tp = np.random.choice(true_positive_indices, 
                                       size=min(tp, len(true_positive_indices)), 
                                       replace=False)
        y_pred[selected_tp] = 1
    
    # 隨機選擇 fp 個假正例
    true_negative_indices = np.where(y_true == 0)[0]
    if len(true_negative_indices) > 0 and fp > 0:
        selected_fp = np.random.choice(true_negative_indices, 
                                       size=min(fp, len(true_negative_indices)), 
                                       replace=False)
        y_pred[selected_fp] = 1
    
    return y_pred


def train_ensemble_model(data_path):
    """
    訓練集成模型
    
    策略：
    1. 使用規則方法篩選候選點
    2. 訓練 ML 模型對候選點進行驗證
    3. 只有同時通過兩者的點才被標記為反轉點
    """
    print("=" * 80)
    print("🚀 集成方法訓練")
    print("=" * 80)
    
    # 載入數據
    df = pd.read_csv(data_path)
    print(f"\n✅ 數據載入完成: {len(df)} 筆資料, {df['reverse'].sum()} 個反轉點")
    
    # 特徵工程
    print("\n" + "=" * 80)
    print("🔧 特徵工程")
    print("=" * 80)
    features = engineer_features_for_ensemble(df)
    print(f"\n✅ 特徵工程完成！總特徵數: {len(features.columns)}")
    
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
    print("📊 集成方法評估")
    print("=" * 80)
    
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    ensemble_results = []
    rule_only_results = []
    ml_only_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        print(f"\nFold {fold}:")
        print(f"  訓練集: {len(X_train)} 筆 (反轉點: {y_train.sum()})")
        print(f"  測試集: {len(X_test)} 筆 (反轉點: {y_test.sum()})")
        
        if y_train.sum() < 2:
            print(f"  ⚠️  訓練集反轉點太少，跳過此 fold")
            continue
        
        # === 1. 模擬規則方法預測 ===
        y_pred_rule = simulate_rule_based_predictions(y_test, precision=0.0864, recall=0.1795)
        
        # === 2. 訓練 ML 模型 ===
        print(f"\n🔧 訓練 ML 模型...")
        
        # SMOTE
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"   SMOTE 重採樣: {len(X_train_balanced)} 筆")
        except:
            X_train_balanced, y_train_balanced = X_train, y_train
            print(f"   ⚠️  SMOTE 失敗，使用原始數據")
        
        # 訓練 XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=len(y_train_balanced[y_train_balanced==0]) / max(len(y_train_balanced[y_train_balanced==1]), 1)
        )
        xgb_model.fit(X_train_balanced, y_train_balanced)
        
        # ML 預測
        y_pred_ml = xgb_model.predict(X_test)
        
        # === 3. 集成預測：規則 AND ML ===
        y_pred_ensemble = (y_pred_rule & y_pred_ml).astype(int)
        
        # === 評估 ===
        
        # 規則方法
        precision_rule = precision_score(y_test, y_pred_rule, zero_division=0)
        recall_rule = recall_score(y_test, y_pred_rule, zero_division=0)
        f1_rule = f1_score(y_test, y_pred_rule, zero_division=0)
        
        # ML 方法
        precision_ml = precision_score(y_test, y_pred_ml, zero_division=0)
        recall_ml = recall_score(y_test, y_pred_ml, zero_division=0)
        f1_ml = f1_score(y_test, y_pred_ml, zero_division=0)
        
        # 集成方法
        precision_ensemble = precision_score(y_test, y_pred_ensemble, zero_division=0)
        recall_ensemble = recall_score(y_test, y_pred_ensemble, zero_division=0)
        f1_ensemble = f1_score(y_test, y_pred_ensemble, zero_division=0)
        
        print(f"\n📊 Fold {fold} 結果:")
        print(f"\n  規則方法:")
        print(f"    精確度: {precision_rule*100:.2f}%")
        print(f"    召回率: {recall_rule*100:.2f}%")
        print(f"    F1分數: {f1_rule:.4f}")
        
        print(f"\n  ML 方法:")
        print(f"    精確度: {precision_ml*100:.2f}%")
        print(f"    召回率: {recall_ml*100:.2f}%")
        print(f"    F1分數: {f1_ml:.4f}")
        
        print(f"\n  集成方法 (規則 AND ML):")
        print(f"    精確度: {precision_ensemble*100:.2f}%")
        print(f"    召回率: {recall_ensemble*100:.2f}%")
        print(f"    F1分數: {f1_ensemble:.4f}")
        
        if precision_ensemble > 0 or recall_ensemble > 0:
            cm = confusion_matrix(y_test, y_pred_ensemble)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                print(f"    TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        
        ensemble_results.append({
            'precision': precision_ensemble,
            'recall': recall_ensemble,
            'f1': f1_ensemble
        })
        
        rule_only_results.append({
            'precision': precision_rule,
            'recall': recall_rule,
            'f1': f1_rule
        })
        
        ml_only_results.append({
            'precision': precision_ml,
            'recall': recall_ml,
            'f1': f1_ml
        })
    
    # 計算平均結果
    print("\n" + "=" * 80)
    print("📊 平均結果比較")
    print("=" * 80)
    
    if ensemble_results:
        ensemble_avg = {
            'precision': np.mean([r['precision'] for r in ensemble_results]),
            'recall': np.mean([r['recall'] for r in ensemble_results]),
            'f1': np.mean([r['f1'] for r in ensemble_results])
        }
        
        rule_avg = {
            'precision': np.mean([r['precision'] for r in rule_only_results]),
            'recall': np.mean([r['recall'] for r in rule_only_results]),
            'f1': np.mean([r['f1'] for r in rule_only_results])
        }
        
        ml_avg = {
            'precision': np.mean([r['precision'] for r in ml_only_results]),
            'recall': np.mean([r['recall'] for r in ml_only_results]),
            'f1': np.mean([r['f1'] for r in ml_only_results])
        }
        
        print(f"\n方法                    精確度        召回率       F1分數")
        print("-" * 70)
        print(f"規則方法 (模擬)         {rule_avg['precision']*100:>6.2f}%    {rule_avg['recall']*100:>6.2f}%    {rule_avg['f1']:.4f}")
        print(f"ML 方法                 {ml_avg['precision']*100:>6.2f}%    {ml_avg['recall']*100:>6.2f}%    {ml_avg['f1']:.4f}")
        print(f"集成方法 (規則 AND ML)  {ensemble_avg['precision']*100:>6.2f}%    {ensemble_avg['recall']*100:>6.2f}%    {ensemble_avg['f1']:.4f}")
        
        print("\n" + "=" * 80)
        print("💡 集成方法分析")
        print("=" * 80)
        
        print(f"\n集成方法通過結合規則和 ML：")
        print(f"  ✓ 精確度提升: {(ensemble_avg['precision'] - rule_avg['precision'])*100:+.2f}%")
        print(f"  ✓ 召回率變化: {(ensemble_avg['recall'] - rule_avg['recall'])*100:+.2f}%")
        print(f"  ✓ F1分數變化: {(ensemble_avg['f1'] - rule_avg['f1']):+.4f}")
        
        if ensemble_avg['precision'] > rule_avg['precision']:
            print(f"\n🎯 集成方法成功提升了精確度！")
            print(f"   通過 ML 二次驗證，減少了假陽性")
        
        if ensemble_avg['f1'] > rule_avg['f1']:
            print(f"\n🏆 集成方法的 F1 分數優於單獨使用規則方法！")
    
    print("\n" + "=" * 80)
    print("✅ 集成方法訓練完成！")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='集成方法訓練')
    parser.add_argument('--data', type=str, 
                        default='training_data/reversal_training_data_2022-01-01_to_2025-09-01_15m.csv',
                        help='訓練數據路徑')
    
    args = parser.parse_args()
    
    train_ensemble_model(args.data)
