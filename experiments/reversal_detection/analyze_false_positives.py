#!/usr/bin/env python3
"""
分析假陽性：為什麼 90% 的信號都是錯的？

找出假陽性的共同特徵，然後針對性地過濾
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from detection.optimal_reversal import OptimalReversalDetection


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


def analyze_false_positives(df):
    """分析假陽性的特徵"""
    print("\n" + "=" * 80)
    print("🔍 分析假陽性")
    print("=" * 80)
    
    # 使用最佳檢測器
    detector = OptimalReversalDetection()
    predictions = detector.detect(df)
    
    # 獲取真實標籤
    if 'reverse' not in df.columns:
        print("❌ 數據中沒有 'reverse' 標籤")
        return
    
    ground_truth = df['reverse'].values
    
    # 分類結果
    true_positives = predictions & ground_truth
    false_positives = predictions & ~ground_truth
    false_negatives = ~predictions & ground_truth
    true_negatives = ~predictions & ~ground_truth
    
    # 統計
    tp_count = np.sum(true_positives)
    fp_count = np.sum(false_positives)
    fn_count = np.sum(false_negatives)
    tn_count = np.sum(true_negatives)
    
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    
    print(f"\n📊 基礎統計:")
    print(f"  真陽性 (TP): {tp_count}")
    print(f"  假陽性 (FP): {fp_count} ⚠️")
    print(f"  假陰性 (FN): {fn_count}")
    print(f"  真陰性 (TN): {tn_count}")
    print(f"\n  精確度: {precision:.2%}")
    print(f"  召回率: {recall:.2%}")
    
    # 提取假陽性數據
    fp_df = df[false_positives].copy()
    tp_df = df[true_positives].copy()
    
    print(f"\n🎯 假陽性 vs 真陽性對比:")
    print(f"  假陽性數量: {len(fp_df)}")
    print(f"  真陽性數量: {len(tp_df)}")
    
    # 比較特徵
    features_to_compare = [
        'rsi', 'macd', 'volume_trend', 'volatility',
        'bb_position', 'atr'
    ]
    
    print(f"\n📈 特徵對比:")
    print(f"{'特徵':<20} {'假陽性均值':>15} {'真陽性均值':>15} {'差異':>10}")
    print("-" * 70)
    
    insights = []
    
    for feature in features_to_compare:
        if feature in df.columns:
            fp_mean = fp_df[feature].mean()
            tp_mean = tp_df[feature].mean()
            diff = abs(fp_mean - tp_mean)
            diff_pct = diff / tp_mean * 100 if tp_mean != 0 else 0
            
            print(f"{feature:<20} {fp_mean:>15.2f} {tp_mean:>15.2f} {diff_pct:>9.1f}%")
            
            # 記錄顯著差異
            if diff_pct > 20:
                insights.append({
                    'feature': feature,
                    'fp_mean': fp_mean,
                    'tp_mean': tp_mean,
                    'diff_pct': diff_pct
                })
    
    # 關鍵洞察
    print(f"\n💡 關鍵洞察:")
    if insights:
        for insight in sorted(insights, key=lambda x: x['diff_pct'], reverse=True):
            print(f"\n  {insight['feature']}:")
            print(f"    假陽性: {insight['fp_mean']:.2f}")
            print(f"    真陽性: {insight['tp_mean']:.2f}")
            print(f"    差異: {insight['diff_pct']:.1f}%")
            
            # 給出建議
            if insight['feature'] == 'volume_trend':
                if insight['fp_mean'] < insight['tp_mean']:
                    print(f"    💡 建議: 要求更高的成交量（> {insight['tp_mean']:.2f}）")
            elif insight['feature'] == 'rsi':
                print(f"    💡 建議: 調整 RSI 閾值")
            elif insight['feature'] == 'volatility':
                if insight['fp_mean'] < insight['tp_mean']:
                    print(f"    💡 建議: 要求更高的波動（> {insight['tp_mean']:.2f}）")
    else:
        print("  沒有發現顯著差異的特徵")
    
    # 分析假陽性發生的時間模式
    print(f"\n⏰ 時間模式分析:")
    if 'timestamp' in df.columns:
        fp_df['datetime'] = pd.to_datetime(fp_df['timestamp'])
        tp_df['datetime'] = pd.to_datetime(tp_df['timestamp'])
        
        # 按小時統計
        fp_df['hour'] = fp_df['datetime'].dt.hour
        tp_df['hour'] = tp_df['datetime'].dt.hour
        
        fp_by_hour = fp_df['hour'].value_counts().sort_index()
        tp_by_hour = tp_df['hour'].value_counts().sort_index()
        
        print(f"\n  假陽性最多的時段:")
        for hour in fp_by_hour.head(3).index:
            count = fp_by_hour[hour]
            print(f"    {hour:02d}:00 - {count} 次")
    
    # 價格變化分析
    print(f"\n📉 信號後的價格變化:")
    
    # 計算信號後 N 個週期的價格變化
    periods = [4, 8, 16]  # 1小時、2小時、4小時（15分鐘 * N）
    
    for period in periods:
        hours = period * 15 / 60
        
        # 假陽性的價格變化
        fp_indices = np.where(false_positives)[0]
        fp_price_changes = []
        
        for idx in fp_indices:
            if idx + period < len(df):
                price_change = (df['close'].iloc[idx + period] - df['close'].iloc[idx]) / df['close'].iloc[idx] * 100
                fp_price_changes.append(price_change)
        
        # 真陽性的價格變化
        tp_indices = np.where(true_positives)[0]
        tp_price_changes = []
        
        for idx in tp_indices:
            if idx + period < len(df):
                price_change = (df['close'].iloc[idx + period] - df['close'].iloc[idx]) / df['close'].iloc[idx] * 100
                tp_price_changes.append(price_change)
        
        if fp_price_changes and tp_price_changes:
            fp_avg = np.mean(fp_price_changes)
            tp_avg = np.mean(tp_price_changes)
            
            print(f"\n  {hours:.1f} 小時後:")
            print(f"    假陽性平均變化: {fp_avg:+.2f}%")
            print(f"    真陽性平均變化: {tp_avg:+.2f}%")
            
            if abs(fp_avg) < 0.5:
                print(f"    💡 假陽性後價格幾乎不動 → 可能是盤整期的噪音")
    
    return fp_df, tp_df, insights


def suggest_improvements(insights):
    """根據分析結果提出改進建議"""
    print(f"\n" + "=" * 80)
    print("🎯 改進建議")
    print("=" * 80)
    
    print(f"\n基於分析，以下是最實用的改進方向：")
    
    print(f"\n1️⃣  簡單過濾器（立即可用）:")
    print(f"   - 如果假陽性的成交量明顯較低 → 加入最低成交量要求")
    print(f"   - 如果假陽性的波動較低 → 提高波動閾值")
    print(f"   - 如果假陽性集中在某些時段 → 避開這些時段")
    
    print(f"\n2️⃣  價格確認（等待確認）:")
    print(f"   - 不要立即交易，等待 1-2 個週期")
    print(f"   - 確認價格真的開始反轉")
    print(f"   - 這會降低召回率，但大幅提高精確度")
    
    print(f"\n3️⃣  組合信號（多重確認）:")
    print(f"   - 基礎信號 + RSI 極端 + 高成交量")
    print(f"   - 只在 3 個條件都滿足時才交易")
    
    print(f"\n4️⃣  止損策略（接受誤報）:")
    print(f"   - 精確度 10% 意味著 9 次虧損 1 次盈利")
    print(f"   - 用嚴格的止損控制虧損")
    print(f"   - 用大的盈虧比（3:1 或更高）來彌補")


def main():
    """主函數"""
    print("=" * 80)
    print("🔬 假陽性分析")
    print("=" * 80)
    
    # 載入數據
    df = load_data()
    if df is None:
        return
    
    print(f"\n數據概況:")
    print(f"  總數據點: {len(df)}")
    if 'reverse' in df.columns:
        print(f"  反轉點: {df['reverse'].sum()} ({df['reverse'].sum()/len(df)*100:.2f}%)")
    
    # 分析假陽性
    fp_df, tp_df, insights = analyze_false_positives(df)
    
    # 提出改進建議
    suggest_improvements(insights)
    
    print(f"\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
