#!/usr/bin/env python3
"""
調整增強檢測器參數

目標：找到精確度和召回率的最佳平衡點
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from experiments.reversal_detection.experiment import ReversalExperiment
from experiments.reversal_detection.enhanced_precision_detector import (
    EnhancedPrecisionDetector,
    ConservativeHighPrecisionDetector
)
from detection.optimal_reversal import OptimalReversalDetection


def test_parameter_combinations():
    """測試不同參數組合"""
    
    print("=" * 80)
    print("🔧 調整增強檢測器參數")
    print("=" * 80)
    
    # 載入數據
    data_path = "training_data/reversal_training_data_2022-01-01_to_2025-09-01_15m.csv"
    experiment = ReversalExperiment(data_path)
    experiment.setup()
    
    # 測試不同的參數組合
    test_configs = [
        # 基線
        {
            'name': 'Baseline',
            'detector': OptimalReversalDetection()
        },
        
        # Enhanced Precision - 放寬參數
        {
            'name': 'Enhanced-Relaxed',
            'detector': EnhancedPrecisionDetector(
                volume_confirmation_threshold=1.0,  # 降低成交量要求
                cooldown_periods=4,  # 縮短冷卻期
                divergence_strength_threshold=0.3,  # 降低背離強度要求
                use_dynamic_thresholds=False  # 關閉動態閾值
            )
        },
        
        # Enhanced Precision - 中等參數
        {
            'name': 'Enhanced-Moderate',
            'detector': EnhancedPrecisionDetector(
                volume_confirmation_threshold=1.1,
                cooldown_periods=6,
                divergence_strength_threshold=0.4,
                use_dynamic_thresholds=True
            )
        },
        
        # Enhanced Precision - 原始參數
        {
            'name': 'Enhanced-Strict',
            'detector': EnhancedPrecisionDetector(
                volume_confirmation_threshold=1.2,
                cooldown_periods=8,
                divergence_strength_threshold=0.5,
                use_dynamic_thresholds=True
            )
        },
        
        # Conservative - 放寬參數
        {
            'name': 'Conservative-Relaxed',
            'detector': ConservativeHighPrecisionDetector(
                extreme_threshold=0.20,  # 放寬極端位置
                atr_accel_threshold=0.5,  # 降低波動要求
                volume_threshold=1.2,  # 降低成交量要求
                rsi_extreme_threshold=30  # 放寬 RSI 要求
            )
        },
        
        # Conservative - 中等參數
        {
            'name': 'Conservative-Moderate',
            'detector': ConservativeHighPrecisionDetector(
                extreme_threshold=0.18,
                atr_accel_threshold=0.6,
                volume_threshold=1.3,
                rsi_extreme_threshold=28
            )
        },
        
        # Conservative - 原始參數
        {
            'name': 'Conservative-Strict',
            'detector': ConservativeHighPrecisionDetector(
                extreme_threshold=0.15,
                atr_accel_threshold=0.7,
                volume_threshold=1.5,
                rsi_extreme_threshold=25
            )
        },
        
        # 只用成交量確認
        {
            'name': 'VolumeOnly',
            'detector': EnhancedPrecisionDetector(
                volume_confirmation_threshold=1.2,
                cooldown_periods=8,
                divergence_strength_threshold=0.0,  # 關閉背離過濾
                use_dynamic_thresholds=False
            )
        },
        
        # 只用冷卻期
        {
            'name': 'CooldownOnly',
            'detector': EnhancedPrecisionDetector(
                volume_confirmation_threshold=0.8,  # 關閉成交量過濾
                cooldown_periods=8,
                divergence_strength_threshold=0.0,  # 關閉背離過濾
                use_dynamic_thresholds=False
            )
        },
    ]
    
    print("\n" + "=" * 80)
    print("📊 參數調整結果")
    print("=" * 80)
    print(f"\n{'配置':<25} {'精確度':>10} {'召回率':>10} {'F1分數':>10} {'信號數':>10}")
    print("-" * 80)
    
    results = []
    
    for config in test_configs:
        detector = config['detector']
        name = config['name']
        
        # 檢測
        predictions = detector.detect(experiment.df)
        
        # 評估
        metrics = experiment.evaluator.evaluate(
            predictions,
            experiment.ground_truth,
            name
        )
        
        results.append({
            'name': name,
            'metrics': metrics,
            'signal_count': predictions.sum()
        })
        
        print(f"{name:<25} {metrics.precision:>9.2%} {metrics.recall:>9.2%} "
              f"{metrics.f1_score:>10.4f} {predictions.sum():>10}")
    
    # 找出最佳配置
    print("\n" + "=" * 80)
    print("🏆 最佳配置")
    print("=" * 80)
    
    # 按 F1 分數排序
    results_sorted = sorted(results, key=lambda x: x['metrics'].f1_score, reverse=True)
    
    print("\n按 F1 分數排序（前5名）:")
    for i, result in enumerate(results_sorted[:5], 1):
        m = result['metrics']
        print(f"{i}. {result['name']:<25} F1={m.f1_score:.4f}, "
              f"精確度={m.precision:.2%}, 召回率={m.recall:.2%}")
    
    # 按精確度排序
    results_by_precision = sorted(results, key=lambda x: x['metrics'].precision, reverse=True)
    
    print("\n按精確度排序（前5名）:")
    for i, result in enumerate(results_by_precision[:5], 1):
        m = result['metrics']
        print(f"{i}. {result['name']:<25} 精確度={m.precision:.2%}, "
              f"F1={m.f1_score:.4f}, 召回率={m.recall:.2%}")
    
    # 找出平衡點（精確度 > 10% 且 F1 最高）
    balanced = [r for r in results if r['metrics'].precision > 0.10]
    if balanced:
        balanced_sorted = sorted(balanced, key=lambda x: x['metrics'].f1_score, reverse=True)
        
        print("\n最佳平衡點（精確度 > 10%）:")
        for i, result in enumerate(balanced_sorted[:3], 1):
            m = result['metrics']
            print(f"{i}. {result['name']:<25} F1={m.f1_score:.4f}, "
                  f"精確度={m.precision:.2%}, 召回率={m.recall:.2%}, "
                  f"信號數={result['signal_count']}")
    
    print("\n✅ 參數調整完成！")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    test_parameter_combinations()
