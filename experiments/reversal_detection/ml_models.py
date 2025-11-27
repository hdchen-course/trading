#!/usr/bin/env python3
"""
ML-Based Reversal Detection

實作機器學習模型：
1. Random Forest
2. XGBoost
3. 時間序列交叉驗證
4. 類別不平衡處理（SMOTE）
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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from experiments.reversal_detection.ml_feature_engineering import ReversalFeatureEngineer


class MLReversalDetector:
    """ML-based 反轉檢測器基類"""
    
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.feature_engineer = ReversalFeatureEngineer()
        self.is_trained = False
    
    def train(self, X_train, y_train, use_smote=True):
        """訓練模型"""
        print(f"\n🔧 訓練 {self.name}...")
        print(f"   訓練數據: {len(X_train)} 筆")
        print(f"   反轉點: {y_train.sum()} 個 ({y_train.sum()/len(y_train)*100:.2f}%)")
        
        if use_smote and y_train.sum() >= 6:  # 至少需要6個正樣本
            print("   使用 SMOTE 處理類別不平衡...")
            # 只使用 SMOTE，不使用 under sampling
            k_neighbors = min(5, y_train.sum()-1)
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=0.3)
            
            try:
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                
                print(f"   重採樣後: {len(X_resampled)} 筆")
                print(f"   反轉點: {y_resampled.sum()} 個 ({y_resampled.sum()/len(y_resampled)*100:.2f}%)")
                
                self.model.fit(X_resampled, y_resampled)
            except Exception as e:
                print(f"   ⚠️  SMOTE 失敗，使用原始數據: {e}")
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        print(f"   ✅ 訓練完成")
    
    def predict(self, X):
        """預測"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """預測機率"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, feature_names):
        """獲取特徵重要性"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return feature_importance
        return None


def time_series_cross_validation(X, y, n_splits=5):
    """時間序列交叉驗證"""
    print("\n" + "=" * 80)
    print("📊 時間序列交叉驗證")
    print("=" * 80)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    print(f"\n使用 {n_splits} 折交叉驗證")
    print(f"總數據: {len(X)} 筆")
    
    fold_info = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        train_reversals = y.iloc[train_idx].sum()
        test_reversals = y.iloc[test_idx].sum()
        
        print(f"\nFold {fold}:")
        print(f"  訓練集: {len(train_idx)} 筆 (反轉點: {train_reversals})")
        print(f"  測試集: {len(test_idx)} 筆 (反轉點: {test_reversals})")
        
        fold_info.append({
            'fold': fold,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_reversals': train_reversals,
            'test_reversals': test_reversals,
            'train_idx': train_idx,
            'test_idx': test_idx
        })
    
    return fold_info


def train_and_evaluate_models(df, labels):
    """訓練並評估所有模型"""
    print("=" * 80)
    print("🤖 ML 模型訓練與評估")
    print("=" * 80)
    
    # 特徵工程
    engineer = ReversalFeatureEngineer()
    features_df = engineer.create_features(df)
    features_clean, labels_clean = engineer.prepare_for_training(features_df, labels)
    
    # 時間序列交叉驗證
    fold_info = time_series_cross_validation(features_clean, labels_clean, n_splits=3)
    
    # 定義模型
    models = [
        MLReversalDetector(
            "RandomForest",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        ),
        MLReversalDetector(
            "XGBoost",
            xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=len(labels_clean) / labels_clean.sum(),
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
        )
    ]
    
    # 對每個模型進行交叉驗證
    results = []
    
    for model_detector in models:
        print("\n" + "=" * 80)
        print(f"🔬 評估 {model_detector.name}")
        print("=" * 80)
        
        fold_results = []
        
        for fold_data in fold_info:
            fold = fold_data['fold']
            train_idx = fold_data['train_idx']
            test_idx = fold_data['test_idx']
            
            X_train = features_clean.iloc[train_idx]
            y_train = labels_clean.iloc[train_idx]
            X_test = features_clean.iloc[test_idx]
            y_test = labels_clean.iloc[test_idx]
            
            # 訓練
            model_detector.train(X_train, y_train, use_smote=True)
            
            # 預測
            y_pred = model_detector.predict(X_test)
            
            # 評估
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            fold_result = {
                'fold': fold,
                'precision': precision * 100,
                'recall': recall * 100,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }
            fold_results.append(fold_result)
            
            print(f"\nFold {fold} 結果:")
            print(f"  精確度: {precision*100:.2f}%")
            print(f"  召回率: {recall*100:.2f}%")
            print(f"  F1分數: {f1:.4f}")
            print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        
        # 計算平均結果
        avg_precision = np.mean([r['precision'] for r in fold_results])
        avg_recall = np.mean([r['recall'] for r in fold_results])
        avg_f1 = np.mean([r['f1'] for r in fold_results])
        
        print(f"\n📊 {model_detector.name} 平均結果:")
        print(f"  精確度: {avg_precision:.2f}%")
        print(f"  召回率: {avg_recall:.2f}%")
        print(f"  F1分數: {avg_f1:.4f}")
        
        results.append({
            'model': model_detector.name,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'fold_results': fold_results
        })
        
        # 特徵重要性（使用最後一個 fold 的模型）
        feature_importance = model_detector.get_feature_importance(features_clean.columns)
        if feature_importance is not None:
            print(f"\n🎯 Top 10 重要特徵:")
            print(feature_importance.head(10).to_string(index=False))
    
    return results, features_clean, labels_clean


def compare_with_rule_based(ml_results):
    """與規則方法比較"""
    print("\n" + "=" * 80)
    print("📊 ML vs 規則方法比較")
    print("=" * 80)
    
    # 規則方法的最佳結果（來自之前的測試）
    rule_based_results = [
        {'name': 'AdaptiveThreeLayer', 'precision': 8.64, 'recall': 17.95, 'f1': 0.1167},
        {'name': 'Conservative48h', 'precision': 5.37, 'recall': 28.21, 'f1': 0.0902}
    ]
    
    print(f"\n{'方法':<25} {'精確度':>10} {'召回率':>10} {'F1分數':>10}")
    print("-" * 70)
    
    # 規則方法
    for result in rule_based_results:
        print(f"{result['name']:<25} {result['precision']:>9.2f}% {result['recall']:>9.2f}% {result['f1']:>9.4f}")
    
    print("-" * 70)
    
    # ML 方法
    for result in ml_results:
        print(f"{result['model']:<25} {result['avg_precision']:>9.2f}% {result['avg_recall']:>9.2f}% {result['avg_f1']:>9.4f}")
    
    # 找出最佳方法
    all_methods = rule_based_results + [
        {'name': r['model'], 'precision': r['avg_precision'], 'recall': r['avg_recall'], 'f1': r['avg_f1']}
        for r in ml_results
    ]
    
    best_f1 = max(all_methods, key=lambda x: x['f1'])
    best_precision = max(all_methods, key=lambda x: x['precision'])
    best_recall = max(all_methods, key=lambda x: x['recall'])
    
    print("\n" + "=" * 80)
    print("🏆 最佳方法")
    print("=" * 80)
    print(f"\n最高 F1: {best_f1['name']} (F1={best_f1['f1']:.4f})")
    print(f"最高精確度: {best_precision['name']} (精確度={best_precision['precision']:.2f}%)")
    print(f"最高召回率: {best_recall['name']} (召回率={best_recall['recall']:.2f}%)")


def main():
    """主函數"""
    print("=" * 80)
    print("🚀 ML-Based 反轉檢測")
    print("=" * 80)
    
    # 載入數據
    df = pd.read_csv('analysis_0911_0913 - data_with_reversal_labels.csv')
    labels = df['reverse']
    
    print(f"\n✅ 數據載入完成: {len(df)} 筆資料, {labels.sum()} 個反轉點")
    
    # 訓練並評估模型
    ml_results, features, labels_clean = train_and_evaluate_models(df, labels)
    
    # 與規則方法比較
    compare_with_rule_based(ml_results)
    
    print("\n" + "=" * 80)
    print("✅ ML 訓練與評估完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
