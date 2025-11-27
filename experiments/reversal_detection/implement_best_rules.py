#!/usr/bin/env python3
"""
實作並測試最佳反轉檢測規則

基於 analyze_reversal_scenarios.py 的分析結果，實作以下規則：
1. RSI<30 AND BB<0.1 (F1=15.0, 精確度11.8%, 召回率20.5%)
2. 價跌量增 (召回率33.3%)
3. 高波動+極端位置 (召回率28.2%)
4. 超買+上軌+高量 (精確度6.8%, 召回率15.4%)
5. 極度超買+MACD正 (召回率17.9%)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from experiments.reversal_detection.experiment import ReversalExperiment


class BestRuleDetector:
    """基於分析結果的最佳規則檢測器"""
    
    def __init__(self, name="BestRules"):
        self.name = name
    
    def detect(self, df):
        """
        檢測反轉點
        
        使用多規則投票機制：
        - 每個規則投一票
        - 得票數 >= 2 則判定為反轉點
        """
        # 計算一階導數（用於背離檢測）
        df['price_1st'] = df['close'].diff()
        df['vol_1st'] = df['volume'].diff()
        
        # 初始化投票
        votes = pd.Series(0, index=df.index)
        
        # 規則1: RSI<30 AND BB<0.1 (最高F1分數)
        rule1 = (df['rsi'] < 30) & (df['bb_position'] < 0.1)
        votes += rule1.astype(int)
        
        # 規則2: 價跌量增 (高召回率)
        rule2 = (df['price_1st'] < 0) & (df['vol_1st'] > 0)
        votes += rule2.astype(int)
        
        # 規則3: 高波動+極端位置
        high_vol = df['volatility'] > df['volatility'].quantile(0.75)
        extreme_pos = (df['bb_position'] < 0.1) | (df['bb_position'] > 0.9)
        rule3 = high_vol & extreme_pos
        votes += rule3.astype(int)
        
        # 規則4: 超買+上軌+高量
        rule4 = (df['rsi'] > 70) & (df['bb_position'] > 0.8) & (df['volume_trend'] > 1.5)
        votes += rule4.astype(int)
        
        # 規則5: 極度超買+MACD正
        rule5 = (df['rsi'] > 75) & (df['macd'] > 0)
        votes += rule5.astype(int)
        
        # 規則6: RSI<25 AND BB<0.1 (高精確度)
        rule6 = (df['rsi'] < 25) & (df['bb_position'] < 0.1)
        votes += rule6.astype(int)
        
        # 投票機制：得票數 >= 2
        signals = votes >= 2
        
        return signals


class ConservativeRuleDetector:
    """保守規則檢測器 - 只使用高精確度規則"""
    
    def __init__(self, name="Conservative"):
        self.name = name
    
    def detect(self, df):
        """只使用精確度最高的規則"""
        # 規則1: RSI<25 AND BB<0.1 (精確度14.3%)
        rule1 = (df['rsi'] < 25) & (df['bb_position'] < 0.1)
        
        # 規則2: BB<0.1 AND Vol>2.0 (精確度10.7%)
        rule2 = (df['bb_position'] < 0.1) & (df['volume_trend'] > 2.0)
        
        # 規則3: RSI<30 AND BB<0.1 (精確度11.8%)
        rule3 = (df['rsi'] < 30) & (df['bb_position'] < 0.1)
        
        # 任一規則命中即判定
        signals = rule1 | rule2 | rule3
        
        return signals


class AggressiveRuleDetector:
    """積極規則檢測器 - 追求高召回率"""
    
    def __init__(self, name="Aggressive"):
        self.name = name
    
    def detect(self, df):
        """使用高召回率規則"""
        # 計算一階導數
        df['price_1st'] = df['close'].diff()
        df['vol_1st'] = df['volume'].diff()
        
        # 規則1: 價跌量增 (召回率33.3%)
        rule1 = (df['price_1st'] < 0) & (df['vol_1st'] > 0)
        
        # 規則2: 高波動+極端位置 (召回率28.2%)
        high_vol = df['volatility'] > df['volatility'].quantile(0.75)
        extreme_pos = (df['bb_position'] < 0.1) | (df['bb_position'] > 0.9)
        rule2 = high_vol & extreme_pos
        
        # 規則3: RSI>70 AND Vol>1.5 (召回率25.6%)
        rule3 = (df['rsi'] > 70) & (df['volume_trend'] > 1.5)
        
        # 規則4: 價漲量縮 (召回率23.1%)
        rule4 = (df['price_1st'] > 0) & (df['vol_1st'] < 0)
        
        # 任一規則命中即判定
        signals = rule1 | rule2 | rule3 | rule4
        
        return signals


class ScenarioBasedDetector:
    """場景導向檢測器 - 針對不同場景使用不同規則"""
    
    def __init__(self, name="ScenarioBased"):
        self.name = name
    
    def detect(self, df):
        """根據市場狀態選擇規則"""
        signals = pd.Series(False, index=df.index)
        
        # 場景1: 超賣反彈 (RSI<30, BB<0.3)
        oversold = (df['rsi'] < 30) & (df['bb_position'] < 0.3)
        # 在超賣區，尋找成交量放大的信號
        oversold_signal = oversold & (df['volume_trend'] > 1.2)
        signals |= oversold_signal
        
        # 場景2: 超買回落 (RSI>70, BB>0.7)
        overbought = (df['rsi'] > 70) & (df['bb_position'] > 0.7)
        # 在超買區，尋找成交量放大的信號
        overbought_signal = overbought & (df['volume_trend'] > 1.2)
        signals |= overbought_signal
        
        # 場景3: 高量反轉 (Vol>2.0)
        high_volume = df['volume_trend'] > 2.0
        # 高量配合極端位置
        extreme_pos = (df['bb_position'] < 0.2) | (df['bb_position'] > 0.8)
        high_vol_signal = high_volume & extreme_pos
        signals |= high_vol_signal
        
        # 場景4: 低量反轉 (Vol<0.8)
        # 低量時需要更強的信號
        low_volume = df['volume_trend'] < 0.8
        very_extreme = (df['bb_position'] < 0.1) | (df['bb_position'] > 0.9)
        low_vol_signal = low_volume & very_extreme & ((df['rsi'] < 25) | (df['rsi'] > 75))
        signals |= low_vol_signal
        
        return signals


class HybridDetector:
    """混合檢測器 - 結合多種策略"""
    
    def __init__(self, name="Hybrid"):
        self.name = name
    
    def detect(self, df):
        """混合多種檢測策略"""
        # 計算一階導數
        df['price_1st'] = df['close'].diff()
        df['vol_1st'] = df['volume'].diff()
        
        # 策略1: 高精確度規則（權重3）
        high_precision = (
            ((df['rsi'] < 25) & (df['bb_position'] < 0.1)) |
            ((df['bb_position'] < 0.1) & (df['volume_trend'] > 2.0))
        )
        
        # 策略2: 背離信號（權重2）
        divergence = (
            ((df['price_1st'] < 0) & (df['vol_1st'] > 0)) |
            ((df['price_1st'] > 0) & (df['vol_1st'] < 0))
        )
        
        # 策略3: 極端位置（權重2）
        high_vol = df['volatility'] > df['volatility'].quantile(0.75)
        extreme_pos = (df['bb_position'] < 0.1) | (df['bb_position'] > 0.9)
        extreme_signal = high_vol & extreme_pos
        
        # 策略4: MACD確認（權重1）
        macd_confirm = (
            ((df['rsi'] < 35) & (df['macd'] < 0)) |
            ((df['rsi'] > 65) & (df['macd'] > 0))
        )
        
        # 加權投票
        votes = (
            high_precision.astype(int) * 3 +
            divergence.astype(int) * 2 +
            extreme_signal.astype(int) * 2 +
            macd_confirm.astype(int) * 1
        )
        
        # 得分 >= 4 判定為反轉點
        signals = votes >= 4
        
        return signals


def test_all_detectors():
    """測試所有檢測器"""
    print("=" * 80)
    print("🧪 測試最佳規則檢測器")
    print("=" * 80)
    
    # 創建實驗
    experiment = ReversalExperiment()
    
    # 測試所有檢測器
    detectors = [
        BestRuleDetector(),
        ConservativeRuleDetector(),
        AggressiveRuleDetector(),
        ScenarioBasedDetector(),
        HybridDetector()
    ]
    
    print("\n" + "=" * 80)
    print("📊 檢測器性能比較")
    print("=" * 80)
    print(f"\n{'檢測器':<20} {'精確度':>10} {'召回率':>10} {'F1分數':>10} {'信號數':>10}")
    print("-" * 80)
    
    results = []
    for detector in detectors:
        result = experiment.test_detector(detector)
        results.append(result)
        
        print(f"{detector.name:<20} {result['precision']:>9.2f}% {result['recall']:>9.2f}% "
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
    
    print("\n✅ 測試完成！")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    test_all_detectors()
