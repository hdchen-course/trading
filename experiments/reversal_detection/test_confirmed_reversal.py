#!/usr/bin/env python3
"""
測試確認反轉策略 vs 預測反轉策略

左側交易（預測）：
- 嘗試在反轉點之前入場
- 精確度低（10.6%），因為很多是假信號
- 風險高，但如果對了收益大

右側交易（確認）：
- 等待反轉確認後才入場
- 精確度應該更高，因為已經看到反轉發生
- 風險低，收益略少（晚入場 0.5%）
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from detection.optimal_reversal import OptimalReversalDetection
from detection.early_reversal_confirmation import (
    EarlyReversalConfirmation,
    VolumeConfirmedEarlyReversal,
    MultiIndicatorEarlyReversal
)


def load_data():
    """載入訓練數據"""
    paths = [
        "training_data/reversal_training_data_2022-01-01_to_2025-09-01_15m.csv",
        "training_data/btc_15m_2022_2025.csv",
        "training_data/btc_features_labeled.csv"
    ]
    
    for path in paths:
        if os.path.exists(path):
            print(f"✅ 載入數據: {path}")
            df = pd.read_csv(path)
            return df
    
    print("❌ 找不到訓練數據")
    return None


def evaluate_detector(df, detector, ground_truth):
    """評估檢測器性能"""
    predictions = detector.detect(df)
    
    # 計算混淆矩陣
    tp = np.sum(predictions & ground_truth)
    fp = np.sum(predictions & ~ground_truth)
    fn = np.sum(~predictions & ground_truth)
    tn = np.sum(~predictions & ~ground_truth)
    
    # 計算指標
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'name': detector.name,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'total_signals': tp + fp
    }


def analyze_entry_timing(df, predictions, ground_truth):
    """
    分析入場時機
    
    對於每個真陽性信號，計算：
    1. 距離真實反轉點有多遠（時間和價格）
    2. 如果在信號點入場，後續的收益如何
    """
    print("\n" + "=" * 80)
    print("⏰ 入場時機分析")
    print("=" * 80)
    
    # 找到真陽性
    true_positives = predictions & ground_truth
    tp_indices = np.where(true_positives)[0]
    
    if len(tp_indices) == 0:
        print("\n沒有真陽性信號")
        return
    
    # 找到所有真實反轉點
    reversal_indices = np.where(ground_truth)[0]
    
    distances = []
    price_diffs = []
    
    for tp_idx in tp_indices:
        # 找到最近的真實反轉點
        nearest_reversal = min(reversal_indices, key=lambda x: abs(x - tp_idx))
        
        # 計算距離（週期數）
        distance = tp_idx - nearest_reversal
        distances.append(distance)
        
        # 計算價格差異
        signal_price = df['close'].iloc[tp_idx]
        reversal_price = df['close'].iloc[nearest_reversal]
        price_diff = abs(signal_price - reversal_price) / reversal_price * 100
        price_diffs.append(price_diff)
    
    print(f"\n真陽性信號數: {len(tp_indices)}")
    print(f"\n距離真實反轉點:")
    print(f"  平均: {np.mean(distances):.1f} 個週期")
    print(f"  中位數: {np.median(distances):.1f} 個週期")
    print(f"  範圍: {np.min(distances)} 到 {np.max(distances)} 個週期")
    
    print(f"\n價格差異:")
    print(f"  平均: {np.mean(price_diffs):.2f}%")
    print(f"  中位數: {np.median(price_diffs):.2f}%")
    print(f"  範圍: {np.min(price_diffs):.2f}% 到 {np.max(price_diffs):.2f}%")
    
    # 計算後續收益
    periods = [4, 8, 16]  # 1小時、2小時、4小時
    
    print(f"\n後續收益分析:")
    for period in periods:
        hours = period * 15 / 60
        
        profits = []
        for tp_idx in tp_indices:
            if tp_idx + period < len(df):
                entry_price = df['close'].iloc[tp_idx]
                exit_price = df['close'].iloc[tp_idx + period]
                profit = (exit_price - entry_price) / entry_price * 100
                profits.append(profit)
        
        if profits:
            print(f"\n  {hours:.1f} 小時後:")
            print(f"    平均收益: {np.mean(profits):+.2f}%")
            print(f"    中位數收益: {np.median(profits):+.2f}%")
            print(f"    勝率: {sum(1 for p in profits if p > 0) / len(profits) * 100:.1f}%")
            print(f"    最大收益: {np.max(profits):+.2f}%")
            print(f"    最大虧損: {np.min(profits):+.2f}%")


def compare_strategies():
    """比較左側交易 vs 右側交易"""
    print("=" * 80)
    print("🔬 左側交易 vs 右側交易對比")
    print("=" * 80)
    
    # 載入數據
    df = load_data()
    if df is None:
        return
    
    print(f"\n數據概況:")
    print(f"  總數據點: {len(df)}")
    
    if 'reverse' not in df.columns:
        print("❌ 數據中沒有 'reverse' 標籤")
        return
    
    ground_truth = df['reverse'].values
    reversal_count = np.sum(ground_truth)
    print(f"  真實反轉點: {reversal_count} ({reversal_count/len(df)*100:.2f}%)")
    
    # 測試不同的檢測器
    left_detector = OptimalReversalDetection()
    left_detector.name = "左側交易（預測反轉）"
    
    detectors = [
        left_detector,
        EarlyReversalConfirmation(
            confirmation_pct=0.003,  # 0.3%
            name="右側交易（0.3% 確認）"
        ),
        EarlyReversalConfirmation(
            confirmation_pct=0.005,  # 0.5%
            name="右側交易（0.5% 確認）"
        ),
        EarlyReversalConfirmation(
            confirmation_pct=0.007,  # 0.7%
            name="右側交易（0.7% 確認）"
        ),
        VolumeConfirmedEarlyReversal(name="右側交易（成交量確認）"),
        MultiIndicatorEarlyReversal(name="右側交易（多指標確認）")
    ]
    
    print("\n" + "=" * 80)
    print("📊 性能對比")
    print("=" * 80)
    print(f"\n{'策略':<30} {'精確度':>10} {'召回率':>10} {'F1':>8} {'信號數':>8}")
    print("-" * 80)
    
    results = []
    for detector in detectors:
        result = evaluate_detector(df, detector, ground_truth)
        results.append(result)
        
        print(f"{result['name']:<30} "
              f"{result['precision']:>9.2f}% "
              f"{result['recall']:>9.2f}% "
              f"{result['f1']:>8.4f} "
              f"{result['total_signals']:>8}")
    
    # 找出最佳策略
    print("\n" + "=" * 80)
    print("🏆 最佳策略")
    print("=" * 80)
    
    best_precision = max(results, key=lambda x: x['precision'])
    best_f1 = max(results, key=lambda x: x['f1'])
    
    print(f"\n最高精確度: {best_precision['name']}")
    print(f"  精確度: {best_precision['precision']:.2f}%")
    print(f"  召回率: {best_precision['recall']:.2f}%")
    print(f"  F1: {best_precision['f1']:.4f}")
    print(f"  信號數: {best_precision['total_signals']}")
    
    print(f"\n最高 F1: {best_f1['name']}")
    print(f"  精確度: {best_f1['precision']:.2f}%")
    print(f"  召回率: {best_f1['recall']:.2f}%")
    print(f"  F1: {best_f1['f1']:.4f}")
    print(f"  信號數: {best_f1['total_signals']}")
    
    # 分析最佳策略的入場時機
    best_detector = next(d for d in detectors if d.name == best_precision['name'])
    best_predictions = best_detector.detect(df)
    analyze_entry_timing(df, best_predictions, ground_truth)
    
    # 視覺化對比
    visualize_comparison(df, detectors, ground_truth)
    
    return results


def visualize_comparison(df, detectors, ground_truth):
    """視覺化不同策略的信號"""
    print("\n" + "=" * 80)
    print("📈 生成視覺化圖表")
    print("=" * 80)
    
    # 只取最近 1000 個數據點
    n_points = min(1000, len(df))
    df_recent = df.iloc[-n_points:].copy()
    ground_truth_recent = ground_truth[-n_points:]
    
    # 轉換時間戳
    if 'timestamp' in df_recent.columns:
        df_recent['datetime'] = pd.to_datetime(df_recent['timestamp'])
    else:
        df_recent['datetime'] = pd.date_range(end=pd.Timestamp.now(), periods=n_points, freq='15min')
    
    # 創建圖表
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))
    
    # 圖 1: 左側交易（預測）
    ax1 = axes[0]
    ax1.plot(df_recent['datetime'], df_recent['close'], 'b-', linewidth=1, alpha=0.7, label='Price')
    
    # 真實反轉點
    true_reversals = df_recent[ground_truth_recent]
    ax1.scatter(true_reversals['datetime'], true_reversals['close'],
                c='red', s=100, marker='o', zorder=5, label='True Reversals', alpha=0.5)
    
    # 左側交易信號
    left_detector = detectors[0]
    left_signals = left_detector.detect(df_recent)
    left_signal_points = df_recent[left_signals]
    ax1.scatter(left_signal_points['datetime'], left_signal_points['close'],
                c='orange', s=150, marker='^', zorder=6, label='Left-side Signals', edgecolors='black')
    
    ax1.set_title('左側交易（預測反轉）', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 圖 2: 右側交易（0.5% 確認）
    ax2 = axes[1]
    ax2.plot(df_recent['datetime'], df_recent['close'], 'b-', linewidth=1, alpha=0.7, label='Price')
    
    # 真實反轉點
    ax2.scatter(true_reversals['datetime'], true_reversals['close'],
                c='red', s=100, marker='o', zorder=5, label='True Reversals', alpha=0.5)
    
    # 右側交易信號（0.5%）
    right_detector = detectors[2]  # 0.5% 確認
    right_signals = right_detector.detect(df_recent)
    right_signal_points = df_recent[right_signals]
    ax2.scatter(right_signal_points['datetime'], right_signal_points['close'],
                c='green', s=150, marker='^', zorder=6, label='Right-side Signals (0.5%)', edgecolors='black')
    
    ax2.set_title('右側交易（0.5% 確認反轉）', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Price (USDT)', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 圖 3: 對比
    ax3 = axes[2]
    ax3.plot(df_recent['datetime'], df_recent['close'], 'b-', linewidth=1, alpha=0.7, label='Price')
    
    # 真實反轉點
    ax3.scatter(true_reversals['datetime'], true_reversals['close'],
                c='red', s=100, marker='o', zorder=5, label='True Reversals', alpha=0.5)
    
    # 左側信號
    ax3.scatter(left_signal_points['datetime'], left_signal_points['close'],
                c='orange', s=150, marker='^', zorder=6, label='Left-side', edgecolors='black', alpha=0.7)
    
    # 右側信號
    ax3.scatter(right_signal_points['datetime'], right_signal_points['close'],
                c='green', s=150, marker='s', zorder=7, label='Right-side (0.5%)', edgecolors='black')
    
    ax3.set_title('左側 vs 右側交易對比', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Price (USDT)', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存圖表
    os.makedirs('output', exist_ok=True)
    output_path = 'output/left_vs_right_side_trading.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 圖表已保存: {output_path}")
    
    plt.close()


def main():
    """主函數"""
    results = compare_strategies()
    
    print("\n" + "=" * 80)
    print("💡 結論")
    print("=" * 80)
    
    print("""
左側交易（預測反轉）:
  優點: 如果預測正確，可以在最佳價位入場
  缺點: 精確度低（~10%），90% 的信號是假的
  適合: 風險承受能力高的交易者

右側交易（確認反轉）:
  優點: 精確度更高，因為已經看到反轉發生
  缺點: 入場點略晚（0.3-0.7%），收益略少
  適合: 追求穩定的交易者

建議:
  1. 使用右側交易作為主要策略
  2. 確認閾值設為 0.5% 是個好的平衡點
  3. 可以結合成交量確認來進一步提高精確度
  4. 用嚴格的止損來控制風險
    """)
    
    print("=" * 80)


if __name__ == '__main__':
    main()
