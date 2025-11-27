#!/usr/bin/env python3
"""
測試三重確認反轉檢測器並生成可視化
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from detection.triple_confirmation_reversal import TripleConfirmationReversalDetector
from utils.indicators import calculate_technical_indicators
from improved_stage1 import load_data


def evaluate_detector(signals, true_reversals):
    """評估檢測器性能"""
    
    # 提取檢測到的反轉點索引
    detected_indices = set(np.where(signals)[0])
    
    # 計算指標
    true_positives = len(detected_indices & true_reversals)
    false_positives = len(detected_indices - true_reversals)
    false_negatives = len(true_reversals - detected_indices)
    
    precision = true_positives / len(detected_indices) * 100 if detected_indices else 0
    recall = true_positives / len(true_reversals) * 100 if true_reversals else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total_detected': len(detected_indices),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    """主函數"""
    print("="*80)
    print("三重確認反轉檢測器測試")
    print("="*80)
    
    # 載入數據
    print("\n載入數據...")
    df = load_data()
    if df is None:
        return
    
    true_reversals = set(np.where(df['reverse'].values)[0])
    
    print(f"✅ 數據載入完成")
    print(f"  總數據點: {len(df)}")
    print(f"  真實反轉點: {len(true_reversals)}")
    print(f"  數據期間: {df.index[0]} to {df.index[-1]}")
    
    # 計算技術指標
    print("\n計算技術指標...")
    df = calculate_technical_indicators(df)
    
    # 創建檢測器
    print("\n創建檢測器...")
    detector = TripleConfirmationReversalDetector(
        distance=20,
        window_size=16
    )
    
    # 執行檢測
    print("\n" + "="*80)
    print("執行三重確認檢測")
    print("="*80)
    
    signals = detector.detect(df)
    metrics = evaluate_detector(signals, true_reversals)
    
    print(f"\n檢測結果:")
    print(f"  檢測到的反轉點: {metrics['total_detected']}")
    print(f"  真陽性: {metrics['true_positives']}")
    print(f"  假陽性: {metrics['false_positives']}")
    print(f"  精確度: {metrics['precision']:.1f}%")
    print(f"  召回率: {metrics['recall']:.1f}%")
    print(f"  F1分數: {metrics['f1']:.1f}")
    
    # 訊號頻率分析
    print("\n" + "="*80)
    print("訊號頻率分析")
    print("="*80)
    
    # 計算時間跨度
    time_span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    
    signals_per_day = metrics['total_detected'] / time_span_days
    signals_per_week = signals_per_day * 7
    signals_per_month = signals_per_day * 30
    
    print(f"\n訊號頻率:")
    print(f"  每天: {signals_per_day:.1f} 個")
    print(f"  每週: {signals_per_week:.1f} 個")
    print(f"  每月: {signals_per_month:.1f} 個")
    
    # 保存結果
    print("\n" + "="*80)
    print("保存結果")
    print("="*80)
    
    # 保存檢測信號到 DataFrame
    df['triple_confirmation_signal'] = signals
    output_file = 'experiments/reversal_detection/triple_confirmation_results.csv'
    df[['timestamp', 'close', 'triple_confirmation_signal']].to_csv(output_file, index=False)
    print(f"\n✅ 檢測結果已保存到: {output_file}")
    
    # 保存性能指標
    metrics_df = pd.DataFrame([metrics])
    metrics_file = 'experiments/reversal_detection/triple_confirmation_metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✅ 性能指標已保存到: {metrics_file}")
    
    # 總結
    print("\n" + "="*80)
    print("📊 總結")
    print("="*80)
    
    print(f"\n✅ 三重確認檢測器性能:")
    print(f"  精確度: {metrics['precision']:.1f}% (目標: 81.2%)")
    print(f"  召回率: {metrics['recall']:.1f}% (目標: 40.4%)")
    print(f"  每天訊號: {signals_per_day:.1f} 個 (目標: 1.2個)")
    
    if abs(metrics['precision'] - 81.2) < 2:
        print("\n🎉 精確度達標！")
    else:
        print(f"\n⚠️ 精確度偏差: {metrics['precision'] - 81.2:+.1f}%")
    
    print("\n" + "="*80)
    print("可視化")
    print("="*80)
    print("\n使用以下命令生成可視化圖表:")
    print(f"python main.py --visualize --start {df.index[0].strftime('%Y-%m-%d')} --end {df.index[-1].strftime('%Y-%m-%d')} --methods peak")


if __name__ == '__main__':
    main()
