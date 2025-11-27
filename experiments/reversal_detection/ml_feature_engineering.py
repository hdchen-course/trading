#!/usr/bin/env python3
"""
ML Feature Engineering Pipeline

準備用於機器學習的特徵：
1. 技術指標特徵
2. 價格位置特徵
3. 二階導數特徵
4. 背離特徵
5. 時間特徵
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class ReversalFeatureEngineer:
    """反轉檢測特徵工程"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_features(self, df):
        """創建所有特徵"""
        print("=" * 80)
        print("🔧 特徵工程")
        print("=" * 80)
        
        features_df = pd.DataFrame(index=df.index)
        
        # 1. 基礎技術指標
        print("\n1️⃣  基礎技術指標...")
        features_df['rsi'] = df['rsi']
        features_df['bb_position'] = df['bb_position']
        features_df['volume_trend'] = df['volume_trend']
        features_df['macd'] = df['macd']
        features_df['macd_histogram'] = df['macd_histogram']
        features_df['volatility'] = df['volatility']
        features_df['atr'] = df['atr']
        
        # 2. 價格位置特徵
        print("2️⃣  價格位置特徵...")
        features_df = self._add_price_position_features(features_df, df)
        
        # 3. 二階導數特徵（已經是 z-score）
        print("3️⃣  二階導數特徵...")
        features_df['price_2nd'] = df['price_2nd']
        features_df['vol_2nd'] = df['vol_2nd']
        features_df['macd_2nd'] = df['macd_2nd']
        features_df['atr_2nd'] = df['atr_2nd']
        
        # 4. 背離特徵
        print("4️⃣  背離特徵...")
        features_df = self._add_divergence_features(features_df, df)
        
        # 5. 組合特徵
        print("5️⃣  組合特徵...")
        features_df = self._add_combination_features(features_df)
        
        # 6. 統計特徵
        print("6️⃣  統計特徵...")
        features_df = self._add_statistical_features(features_df, df)
        
        self.feature_names = features_df.columns.tolist()
        
        print(f"\n✅ 特徵工程完成！")
        print(f"   總特徵數: {len(self.feature_names)}")
        print(f"   特徵列表: {self.feature_names[:10]}... (顯示前10個)")
        
        return features_df
    
    def _add_price_position_features(self, features_df, df):
        """添加價格位置特徵"""
        # 計算24小時和48小時位置
        window_24h = 96
        window_48h = 192
        
        high_24h = df['high'].rolling(window=window_24h, min_periods=1).max()
        low_24h = df['low'].rolling(window=window_24h, min_periods=1).min()
        features_df['position_24h'] = (df['close'] - low_24h) / (high_24h - low_24h + 1e-10)
        
        high_48h = df['high'].rolling(window=window_48h, min_periods=1).max()
        low_48h = df['low'].rolling(window=window_48h, min_periods=1).min()
        features_df['position_48h'] = (df['close'] - low_48h) / (high_48h - low_48h + 1e-10)
        
        # 距離高低點的百分比
        features_df['dist_from_high_24h_pct'] = (df['close'] - high_24h) / high_24h * 100
        features_df['dist_from_low_24h_pct'] = (df['close'] - low_24h) / low_24h * 100
        
        # 波動範圍
        features_df['range_24h_pct'] = (high_24h - low_24h) / low_24h * 100
        
        return features_df
    
    def _add_divergence_features(self, features_df, df):
        """添加背離特徵"""
        # 價量背離強度
        features_df['price_vol_divergence'] = features_df['price_2nd'] * (-features_df['vol_2nd'])
        
        # MACD-價格背離強度
        features_df['macd_price_divergence'] = features_df['macd_2nd'] * (-features_df['price_2nd'])
        
        # 極強背離標記
        features_df['strong_price_vol_divergence'] = (
            ((features_df['price_2nd'] > 1.5) & (features_df['vol_2nd'] < -1.5)) |
            ((features_df['price_2nd'] < -1.5) & (features_df['vol_2nd'] > 1.5))
        ).astype(int)
        
        return features_df
    
    def _add_combination_features(self, features_df):
        """添加組合特徵"""
        # RSI + BB 組合
        features_df['rsi_bb_product'] = features_df['rsi'] * features_df['bb_position']
        
        # 極端位置標記
        features_df['extreme_position'] = (
            (features_df['position_24h'] < 0.15) | 
            (features_df['position_24h'] > 0.85)
        ).astype(int)
        
        # RSI 極端標記
        features_df['extreme_rsi'] = (
            (features_df['rsi'] < 30) | 
            (features_df['rsi'] > 70)
        ).astype(int)
        
        # 高量標記
        features_df['high_volume'] = (features_df['volume_trend'] > 1.5).astype(int)
        
        # 波動率加速標記
        features_df['volatility_acceleration'] = (features_df['atr_2nd'] > 0.5).astype(int)
        
        return features_df
    
    def _add_statistical_features(self, features_df, df):
        """添加統計特徵"""
        # RSI 的滾動統計
        features_df['rsi_ma_5'] = df['rsi'].rolling(window=5, min_periods=1).mean()
        features_df['rsi_std_5'] = df['rsi'].rolling(window=5, min_periods=1).std()
        
        # 成交量趨勢的滾動統計
        features_df['vol_trend_ma_5'] = features_df['volume_trend'].rolling(window=5, min_periods=1).mean()
        
        # 價格變化率
        features_df['price_change_1'] = df['close'].pct_change(1)
        features_df['price_change_5'] = df['close'].pct_change(5)
        
        return features_df
    
    def prepare_for_training(self, features_df, labels):
        """準備訓練數據"""
        # 移除 NaN
        mask = ~(features_df.isna().any(axis=1) | labels.isna())
        features_clean = features_df[mask].copy()
        labels_clean = labels[mask].copy()
        
        print(f"\n📊 數據清理:")
        print(f"   原始數據: {len(features_df)} 筆")
        print(f"   清理後: {len(features_clean)} 筆")
        print(f"   反轉點: {labels_clean.sum()} 個 ({labels_clean.sum()/len(labels_clean)*100:.2f}%)")
        
        return features_clean, labels_clean
    
    def get_feature_importance_names(self):
        """獲取特徵名稱（用於特徵重要性分析）"""
        return self.feature_names


def analyze_feature_distributions(features_df, labels):
    """分析特徵分佈"""
    print("\n" + "=" * 80)
    print("📊 特徵分佈分析")
    print("=" * 80)
    
    reversal_features = features_df[labels == True]
    non_reversal_features = features_df[labels == False]
    
    print(f"\n{'特徵':<30} {'反轉點均值':>15} {'非反轉點均值':>15} {'差異':>10}")
    print("-" * 80)
    
    significant_features = []
    
    for col in features_df.columns[:15]:  # 顯示前15個特徵
        rev_mean = reversal_features[col].mean()
        non_rev_mean = non_reversal_features[col].mean()
        diff = abs(rev_mean - non_rev_mean)
        
        print(f"{col:<30} {rev_mean:>15.3f} {non_rev_mean:>15.3f} {diff:>10.3f}")
        
        # 記錄差異顯著的特徵
        if diff > 0.1:
            significant_features.append((col, diff))
    
    print(f"\n✅ 發現 {len(significant_features)} 個差異顯著的特徵")
    
    return significant_features


if __name__ == '__main__':
    # 測試特徵工程
    print("=" * 80)
    print("🧪 測試特徵工程 Pipeline")
    print("=" * 80)
    
    # 載入數據
    df = pd.read_csv('analysis_0911_0913 - data_with_reversal_labels.csv')
    labels = df['reverse']
    
    print(f"\n✅ 數據載入完成: {len(df)} 筆資料, {labels.sum()} 個反轉點")
    
    # 創建特徵
    engineer = ReversalFeatureEngineer()
    features_df = engineer.create_features(df)
    
    # 準備訓練數據
    features_clean, labels_clean = engineer.prepare_for_training(features_df, labels)
    
    # 分析特徵分佈
    significant_features = analyze_feature_distributions(features_clean, labels_clean)
    
    print("\n" + "=" * 80)
    print("✅ 特徵工程測試完成！")
    print("=" * 80)
