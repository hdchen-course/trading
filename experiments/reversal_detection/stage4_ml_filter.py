#!/usr/bin/env python3
"""
階段 4: Machine Learning 篩選

使用 ML 模型來篩選候選反轉點
目標：進一步提高精確度
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 重用階段 1-2 的函數
from two_stage_detection import (
    load_data,
    stage1_find_candidates,
    analyze_candidate_features,
    stage3_verify_profit
)


def prepare_ml_data(features_df, df, true_reversals):
    """
    準備 ML 訓練數據
    
    Args:
        features_df: 候選點特徵
        df: 原始數據
        true_reversals: 真實反轉點索引
        
    Returns:
        X: 特徵矩陣
        y: 標籤（1=真實反轉點，0=假陽性）
    """
    print("\n" + "="*80)
    print("準備 ML 訓練數據")
    print("="*80)
    
    # 特徵
    feature_columns = [
        'price_position',
        'volatility',
        'volume_ratio',
        'rsi',
        'trend_strength',
        'left_price_change'
    ]
    
    X = features_df[feature_columns].values
    
    # 標籤：候選點是否是真實反轉點
    candidate_indices = features_df['index'].values
    y = np.isin(candidate_indices, true_reversals).astype(int)
    
    print(f"\n數據統計:")
    print(f"  總樣本: {len(y)}")
    print(f"  正樣本（真實反轉點）: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)")
    print(f"  負樣本（假陽性）: {len(y)-np.sum(y)} ({(len(y)-np.sum(y))/len(y)*100:.1f}%)")
    
    return X, y, feature_columns


def train_ml_model(X, y, feature_columns):
    """
    訓練 ML 模型
    
    使用 RandomForest 分類器
    """
    print("\n" + "="*80)
    print("訓練 ML 模型")
    print("="*80)
    
    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n訓練集: {len(X_train)} 樣本")
    print(f"測試集: {len(X_test)} 樣本")
    
    # 訓練模型
    print(f"\n訓練 RandomForest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',  # 處理不平衡數據
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 評估
    print(f"\n訓練集性能:")
    y_train_pred = model.predict(X_train)
    print(classification_report(y_train, y_train_pred, target_names=['假陽性', '真反轉']))
    
    print(f"\n測試集性能:")
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred, target_names=['假陽性', '真反轉']))
    
    # 特徵重要性
    print(f"\n特徵重要性:")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance)
    
    return model, X_test, y_test


def apply_ml_filter(model, features_df, threshold=0.5):
    """
    應用 ML 模型篩選候選點
    
    Args:
        model: 訓練好的模型
        features_df: 候選點特徵
        threshold: 分類閾值
        
    Returns:
        filtered_indices: 篩選後的索引
        probabilities: 預測概率
    """
    print("\n" + "="*80)
    print("應用 ML 篩選")
    print("="*80)
    
    feature_columns = [
        'price_position',
        'volatility',
        'volume_ratio',
        'rsi',
        'trend_strength',
        'left_price_change'
    ]
    
    X = features_df[feature_columns].values
    
    # 預測概率
    probabilities = model.predict_proba(X)[:, 1]  # 正類概率
    
    # 應用閾值
    predictions = probabilities >= threshold
    
    filtered_features = features_df[predictions].copy()
    filtered_features['probability'] = probabilities[predictions]
    
    print(f"\n篩選結果:")
    print(f"  候選點: {len(features_df)}")
    print(f"  ML 篩選後: {len(filtered_features)}")
    print(f"  篩選率: {len(filtered_features)/len(features_df)*100:.1f}%")
    
    print(f"\n預測概率分佈:")
    print(f"  平均: {np.mean(probabilities):.3f}")
    print(f"  中位數: {np.median(probabilities):.3f}")
    print(f"  最大: {np.max(probabilities):.3f}")
    print(f"  最小: {np.min(probabilities):.3f}")
    
    return filtered_features['index'].values, probabilities


def compare_rule_vs_ml(features_df, df, true_reversals, model):
    """
    對比規則篩選 vs ML 篩選
    """
    print("\n" + "="*80)
    print("對比：規則篩選 vs ML 篩選")
    print("="*80)
    
    # 規則篩選（從階段 2）
    rule1 = (features_df['price_position'] < 0.2) | (features_df['price_position'] > 0.8)
    rule2 = features_df['volatility'] > features_df['volatility'].quantile(0.5)
    rule3 = features_df['volume_ratio'] > 1.0
    rule4 = (features_df['rsi'] < 35) | (features_df['rsi'] > 65)
    rule5 = features_df['trend_strength'] > 0.005
    
    rule_count = (
        rule1.astype(int) +
        rule2.astype(int) +
        rule3.astype(int) +
        rule4.astype(int) +
        rule5.astype(int)
    )
    
    rule_filtered = rule_count >= 3
    rule_indices = features_df[rule_filtered]['index'].values
    
    # ML 篩選
    ml_indices, _ = apply_ml_filter(model, features_df, threshold=0.5)
    
    # 評估
    print(f"\n{'方法':<20} {'信號數':>10} {'精確度':>10} {'召回率':>10}")
    print("-" * 60)
    
    # 規則方法
    rule_tp = np.sum(np.isin(rule_indices, true_reversals))
    rule_precision = rule_tp / len(rule_indices) * 100 if len(rule_indices) > 0 else 0
    rule_recall = rule_tp / len(true_reversals) * 100
    
    print(f"{'規則篩選':<20} {len(rule_indices):>10} {rule_precision:>9.1f}% {rule_recall:>9.1f}%")
    
    # ML 方法
    ml_tp = np.sum(np.isin(ml_indices, true_reversals))
    ml_precision = ml_tp / len(ml_indices) * 100 if len(ml_indices) > 0 else 0
    ml_recall = ml_tp / len(true_reversals) * 100
    
    print(f"{'ML 篩選':<20} {len(ml_indices):>10} {ml_precision:>9.1f}% {ml_recall:>9.1f}%")
    
    # 驗證收益
    print(f"\n規則篩選收益:")
    stage3_verify_profit(df, rule_indices, periods=[1, 2, 4])
    
    print(f"\nML 篩選收益:")
    stage3_verify_profit(df, ml_indices, periods=[1, 2, 4])


def main():
    """主函數"""
    print("="*80)
    print("🤖 階段 4: Machine Learning 篩選")
    print("="*80)
    
    # 載入數據
    df = load_data()
    if df is None:
        return
    
    true_reversals = np.where(df['reverse'].values)[0]
    
    # 階段 1: 找候選點
    candidates, candidate_types, _ = stage1_find_candidates(df, lookback=20)
    
    # 分析特徵
    features_df = analyze_candidate_features(df, candidates, candidate_types)
    
    # 準備 ML 數據
    X, y, feature_columns = prepare_ml_data(features_df, df, true_reversals)
    
    # 訓練模型
    model, X_test, y_test = train_ml_model(X, y, feature_columns)
    
    # 對比規則 vs ML
    compare_rule_vs_ml(features_df, df, true_reversals, model)
    
    print("\n" + "="*80)
    print("💡 結論")
    print("="*80)
    
    print("""
ML 篩選的優勢：
1. 自動學習特徵組合
2. 可以捕捉非線性關係
3. 性能通常優於手工規則

下一步：
1. 嘗試不同的 ML 模型（XGBoost, LightGBM）
2. 調整分類閾值來平衡精確度和召回率
3. 增加更多特徵
4. 使用時間序列交叉驗證
    """)


if __name__ == '__main__':
    main()
