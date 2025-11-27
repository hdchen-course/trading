#!/usr/bin/env python3
"""
三層反轉檢測器

基於所有分析結果，實作分層檢測策略：
- 第一層：價格位置篩選（區分頂部/底部）
- 第二層：技術指標確認（針對頂部/底部使用不同規則）
- 第三層：波動率加速確認

分別處理頂部反轉和底部反轉
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from experiments.reversal_detection.experiment import ReversalExperiment


class ThreeLayerDetector:
    """三層反轉檢測器"""
    
    def __init__(self, name="ThreeLayer"):
        self.name = name
    
    def detect(self, df):
        """
        三層檢測策略
        
        分別檢測底部反轉和頂部反轉
        """
        # 計算價格位置特徵（如果不存在）
        if 'position_24h' not in df.columns:
            df = self._calculate_price_position(df)
        
        # ========== 底部反轉檢測 ==========
        
        # 第一層：價格位置（接近24小時低點）
        bottom_layer1 = df['position_24h'] < 0.2
        
        # 第二層：技術指標確認（超賣信號）
        bottom_layer2 = (
            (df['rsi'] < 30) &           # RSI 超賣
            (df['bb_position'] < 0.2)    # BB 下軌
        )
        
        # 第三層：波動率加速
        bottom_layer3 = df['atr_2nd'] > 0.5
        
        # 底部反轉信號
        bottom_reversal = bottom_layer1 & bottom_layer2 & bottom_layer3
        
        # ========== 頂部反轉檢測 ==========
        
        # 第一層：價格位置（接近24小時高點）
        top_layer1 = df['position_24h'] > 0.8
        
        # 第二層：技術指標確認（超買信號）
        top_layer2 = (
            (df['rsi'] > 70) &           # RSI 超買
            (df['bb_position'] > 0.8)    # BB 上軌
        )
        
        # 第三層：波動率加速
        top_layer3 = df['atr_2nd'] > 0.5
        
        # 頂部反轉信號
        top_reversal = top_layer1 & top_layer2 & top_layer3
        
        # 合併信號
        signals = bottom_reversal | top_reversal
        
        return signals
    
    def _calculate_price_position(self, df):
        """計算價格位置特徵"""
        window_24h = 96  # 15分鐘 * 96 = 24小時
        
        df['high_24h'] = df['high'].rolling(window=window_24h, min_periods=1).max()
        df['low_24h'] = df['low'].rolling(window=window_24h, min_periods=1).min()
        df['position_24h'] = (df['close'] - df['low_24h']) / (df['high_24h'] - df['low_24h'])
        
        return df


class EnhancedThreeLayerDetector:
    """增強版三層檢測器 - 加入更多條件"""
    
    def __init__(self, name="EnhancedThreeLayer"):
        self.name = name
    
    def detect(self, df):
        """增強版三層檢測"""
        if 'position_24h' not in df.columns:
            df = self._calculate_price_position(df)
        
        # ========== 底部反轉檢測 ==========
        
        # 第一層：價格位置
        bottom_layer1 = df['position_24h'] < 0.2
        
        # 第二層：技術指標（多個條件，滿足任一即可）
        bottom_layer2_a = (df['rsi'] < 30) & (df['bb_position'] < 0.2)  # RSI+BB
        bottom_layer2_b = (df['rsi'] < 25) & (df['volume_trend'] > 1.5)  # 極度超賣+高量
        bottom_layer2_c = (df['bb_position'] < 0.1) & (df['volume_trend'] > 2.0)  # 極度下軌+超高量
        
        bottom_layer2 = bottom_layer2_a | bottom_layer2_b | bottom_layer2_c
        
        # 第三層：波動率或背離
        bottom_layer3_a = df['atr_2nd'] > 0.5  # 波動率加速
        bottom_layer3_b = (df['price_2nd'] < -1.0) & (df['vol_2nd'] > 1.0)  # 價跌量增背離
        
        bottom_layer3 = bottom_layer3_a | bottom_layer3_b
        
        bottom_reversal = bottom_layer1 & bottom_layer2 & bottom_layer3
        
        # ========== 頂部反轉檢測 ==========
        
        # 第一層：價格位置
        top_layer1 = df['position_24h'] > 0.8
        
        # 第二層：技術指標
        top_layer2_a = (df['rsi'] > 70) & (df['bb_position'] > 0.8)  # RSI+BB
        top_layer2_b = (df['rsi'] > 75) & (df['volume_trend'] > 1.5)  # 極度超買+高量
        top_layer2_c = (df['bb_position'] > 0.9) & (df['volume_trend'] > 2.0)  # 極度上軌+超高量
        
        top_layer2 = top_layer2_a | top_layer2_b | top_layer2_c
        
        # 第三層：波動率或背離
        top_layer3_a = df['atr_2nd'] > 0.5
        top_layer3_b = (df['price_2nd'] > 1.0) & (df['vol_2nd'] < -1.0)  # 價漲量縮背離
        
        top_layer3 = top_layer3_a | top_layer3_b
        
        top_reversal = top_layer1 & top_layer2 & top_layer3
        
        signals = bottom_reversal | top_reversal
        
        return signals
    
    def _calculate_price_position(self, df):
        """計算價格位置特徵"""
        window_24h = 96
        
        df['high_24h'] = df['high'].rolling(window=window_24h, min_periods=1).max()
        df['low_24h'] = df['low'].rolling(window=window_24h, min_periods=1).min()
        df['position_24h'] = (df['close'] - df['low_24h']) / (df['high_24h'] - df['low_24h'])
        
        return df


class AdaptiveThreeLayerDetector:
    """自適應三層檢測器 - 根據市場狀態調整閾值"""
    
    def __init__(self, name="AdaptiveThreeLayer"):
        self.name = name
    
    def detect(self, df):
        """自適應三層檢測"""
        if 'position_24h' not in df.columns:
            df = self._calculate_price_position(df)
        
        # 計算市場波動狀態
        volatility_high = df['atr_2nd'] > df['atr_2nd'].quantile(0.75)
        
        # ========== 底部反轉檢測 ==========
        
        # 第一層：根據波動調整位置閾值
        bottom_layer1 = df['position_24h'] < 0.15  # 更嚴格的位置要求
        
        # 第二層：根據波動調整指標閾值
        # 高波動時放寬 RSI 要求，低波動時更嚴格
        bottom_layer2 = (
            ((df['rsi'] < 35) & ~volatility_high) |  # 低波動：RSI < 35
            ((df['rsi'] < 30) & volatility_high)     # 高波動：RSI < 30
        ) & (df['bb_position'] < 0.2)
        
        # 第三層：必須有波動率加速或強背離
        bottom_layer3 = (
            (df['atr_2nd'] > 1.0) |  # 強波動加速
            ((df['price_2nd'] < -1.5) & (df['vol_2nd'] > 1.5))  # 強背離
        )
        
        bottom_reversal = bottom_layer1 & bottom_layer2 & bottom_layer3
        
        # ========== 頂部反轉檢測 ==========
        
        # 第一層
        top_layer1 = df['position_24h'] > 0.85
        
        # 第二層
        top_layer2 = (
            ((df['rsi'] > 65) & ~volatility_high) |
            ((df['rsi'] > 70) & volatility_high)
        ) & (df['bb_position'] > 0.8)
        
        # 第三層
        top_layer3 = (
            (df['atr_2nd'] > 1.0) |
            ((df['price_2nd'] > 1.5) & (df['vol_2nd'] < -1.5))
        )
        
        top_reversal = top_layer1 & top_layer2 & top_layer3
        
        signals = bottom_reversal | top_reversal
        
        return signals
    
    def _calculate_price_position(self, df):
        """計算價格位置特徵"""
        window_24h = 96
        
        df['high_24h'] = df['high'].rolling(window=window_24h, min_periods=1).max()
        df['low_24h'] = df['low'].rolling(window=window_24h, min_periods=1).min()
        df['position_24h'] = (df['close'] - df['low_24h']) / (df['high_24h'] - df['low_24h'])
        
        return df


class Conservative48hDetector:
    """保守的48小時檢測器 - 使用最佳規則"""
    
    def __init__(self, name="Conservative48h"):
        self.name = name
    
    def detect(self, df):
        """
        基於分析中 F1=9.0 的最佳規則：
        48h極端位置 + RSI極端 + 高量
        """
        if 'position_48h' not in df.columns:
            df = self._calculate_price_position(df)
        
        # 48小時極端位置
        extreme_position = (df['position_48h'] < 0.15) | (df['position_48h'] > 0.85)
        
        # RSI 極端
        extreme_rsi = (df['rsi'] < 30) | (df['rsi'] > 70)
        
        # 高量
        high_volume = df['volume_trend'] > 1.5
        
        # 組合信號
        signals = extreme_position & extreme_rsi & high_volume
        
        return signals
    
    def _calculate_price_position(self, df):
        """計算價格位置特徵"""
        window_24h = 96
        window_48h = 192
        
        df['high_24h'] = df['high'].rolling(window=window_24h, min_periods=1).max()
        df['low_24h'] = df['low'].rolling(window=window_24h, min_periods=1).min()
        df['position_24h'] = (df['close'] - df['low_24h']) / (df['high_24h'] - df['low_24h'])
        
        df['high_48h'] = df['high'].rolling(window=window_48h, min_periods=1).max()
        df['low_48h'] = df['low'].rolling(window=window_48h, min_periods=1).min()
        df['position_48h'] = (df['close'] - df['low_48h']) / (df['high_48h'] - df['low_48h'])
        
        return df


def test_all_three_layer_detectors():
    """測試所有三層檢測器"""
    print("=" * 80)
    print("🧪 測試三層反轉檢測器")
    print("=" * 80)
    
    # 創建實驗
    experiment = ReversalExperiment('analysis_0911_0913 - data_with_reversal_labels.csv')
    
    # 測試所有檢測器
    detectors = [
        ThreeLayerDetector(),
        EnhancedThreeLayerDetector(),
        AdaptiveThreeLayerDetector(),
        Conservative48hDetector()
    ]
    
    # 載入數據
    experiment.setup()
    
    print("\n" + "=" * 80)
    print("📊 檢測器性能比較")
    print("=" * 80)
    print(f"\n{'檢測器':<25} {'精確度':>10} {'召回率':>10} {'F1分數':>10} {'信號數':>10}")
    print("-" * 80)
    
    results = []
    for detector in detectors:
        metrics = experiment.test_detection_method(
            detector.detect,
            detector.name,
            ensure_no_lookahead=False  # 分析模式，使用完整數據
        )
        
        result = {
            'detector_name': detector.name,
            'precision': metrics.precision * 100,
            'recall': metrics.recall * 100,
            'f1_score': metrics.f1_score,
            'true_positives': metrics.true_positives,
            'false_positives': metrics.false_positives,
            'false_negatives': metrics.false_negatives,
            'total_signals': metrics.true_positives + metrics.false_positives,
            'total_reversals': metrics.true_positives + metrics.false_negatives
        }
        results.append(result)
        
        print(f"{detector.name:<25} {result['precision']:>9.2f}% {result['recall']:>9.2f}% "
              f"{result['f1_score']:>9.2f} {result['total_signals']:>10}")
    
    # 找出最佳檢測器
    print("\n" + "=" * 80)
    print("🏆 最佳檢測器")
    print("=" * 80)
    
    best_f1 = max(results, key=lambda x: x['f1_score'])
    best_precision = max(results, key=lambda x: x['precision'])
    best_recall = max(results, key=lambda x: x['recall'])
    
    print(f"\n最高 F1 分數: {best_f1['detector_name']} (F1={best_f1['f1_score']:.2f})")
    print(f"最高精確度: {best_precision['detector_name']} (精確度={best_precision['precision']:.2f}%)")
    print(f"最高召回率: {best_recall['detector_name']} (召回率={best_recall['recall']:.2f}%)")
    
    # 詳細分析最佳檢測器
    print("\n" + "=" * 80)
    print(f"📋 最佳檢測器詳細分析: {best_f1['detector_name']}")
    print("=" * 80)
    
    print(f"\n命中反轉點: {best_f1['true_positives']}/{best_f1['total_reversals']}")
    print(f"誤報: {best_f1['false_positives']}")
    print(f"漏報: {best_f1['false_negatives']}")
    print(f"總信號數: {best_f1['total_signals']}")
    
    # 與目標比較
    print("\n" + "=" * 80)
    print("🎯 與目標比較")
    print("=" * 80)
    
    target_precision = 50.0
    target_recall = 70.0
    target_f1 = 0.60
    
    print(f"\n目標: 精確度 ≥ {target_precision}%, 召回率 ≥ {target_recall}%, F1 ≥ {target_f1}")
    print(f"\n{'檢測器':<25} {'精確度達標':>12} {'召回率達標':>12} {'F1達標':>10}")
    print("-" * 70)
    
    for result in results:
        precision_ok = "✅" if result['precision'] >= target_precision else "❌"
        recall_ok = "✅" if result['recall'] >= target_recall else "❌"
        f1_ok = "✅" if result['f1_score'] >= target_f1 else "❌"
        
        print(f"{result['detector_name']:<25} {precision_ok:>12} {recall_ok:>12} {f1_ok:>10}")
    
    print("\n✅ 測試完成！")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    test_all_three_layer_detectors()
