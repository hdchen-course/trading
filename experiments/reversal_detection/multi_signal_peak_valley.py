#!/usr/bin/env python3
"""
多信號 Peak Valley 分析

目標：
對 macd_2nd, price_2nd, atr_2nd, vol_2nd 這四個 z-score 特徵
應用 peak_valley 方法，看看它們的局部極值是否與真實反轉點相關

假設：
- 如果這些指標的局部極值也出現在真實反轉點附近
- 那麼可以用它們來進一步篩選候選點
- 多個信號的交叉確認可能提高精確度
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from improved_stage1 import load_data


def find_extremes_for_signal(values, distance=20, prominence=None):
    """
    對任意信號應用 peak_valley 檢測
    
    Args:
        values: 信號值（如 macd_2nd, price_2nd 等）
        distance: 極值之間的最小距離
        prominence: 突出度閾值
    
    Returns:
        extremes: 所有極值的索引
        peaks: 波峰索引
        valleys: 波谷索引
    """
    # 動態計算 prominence
    if prominence is None:
        window_size = min(500, len(values) // 4)
        if window_size < 100:
            window_size = len(values)
        
        rolling_std = pd.Series(values).rolling(window=window_size, min_periods=1).std()
        prominence = rolling_std.median() * 0.5
    
    # 找波谷
    valleys, _ = find_peaks(-values, distance=distance, prominence=prominence)
    
    # 找波峰
    peaks, _ = find_peaks(values, distance=distance, prominence=prominence)
    
    # 合併並排序
    extremes = np.sort(np.concatenate([peaks, valleys]))
    
    return extremes, peaks, valleys


def analyze_signal_extremes(df, signal_name, true_reversals, distance=20):
    """
    分析單個信號的極值與真實反轉點的關係
    
    Args:
        df: DataFrame
        signal_name: 信號名稱（如 'macd_2nd'）
        true_reversals: 真實反轉點索引集合
        distance: 極值之間的最小距離
    """
    print(f"\n{'='*80}")
    print(f"分析信號: {signal_name}")
    print(f"{'='*80}")
    
    if signal_name not in df.columns:
        print(f"⚠️  信號 {signal_name} 不存在於數據中")
        return None
    
    # 獲取信號值
    values = df[signal_name].values
    
    # 檢查是否有 NaN
    nan_count = np.isnan(values).sum()
    if nan_count > 0:
        print(f"⚠️  發現 {nan_count} 個 NaN 值，將填充為 0")
        values = np.nan_to_num(values, nan=0.0)
    
    # 找出信號的局部極值
    extremes, peaks, valleys = find_extremes_for_signal(values, distance=distance)
    
    print(f"\n信號統計:")
    print(f"  局部極值總數: {len(extremes)}")
    print(f"  波峰: {len(peaks)}")
    print(f"  波谷: {len(valleys)}")
    
    # 分析與真實反轉點的重疊
    # 定義「接近」：在 ±2 根K線範圍內
    tolerance = 2
    
    matched_reversals = set()
    matched_extremes = set()
    
    for rev_idx in true_reversals:
        for ext_idx in extremes:
            if abs(rev_idx - ext_idx) <= tolerance:
                matched_reversals.add(rev_idx)
                matched_extremes.add(ext_idx)
    
    # 計算指標
    reversal_coverage = len(matched_reversals) / len(true_reversals) * 100 if len(true_reversals) > 0 else 0
    extreme_precision = len(matched_extremes) / len(extremes) * 100 if len(extremes) > 0 else 0
    
    print(f"\n與真實反轉點的關係:")
    print(f"  匹配的反轉點: {len(matched_reversals)} / {len(true_reversals)} ({reversal_coverage:.1f}%)")
    print(f"  匹配的極值: {len(matched_extremes)} / {len(extremes)} ({extreme_precision:.1f}%)")
    
    # 評估作為過濾器的潛力
    if reversal_coverage >= 70 and extreme_precision >= 50:
        print(f"\n✅ 高潛力過濾器！")
        print(f"   - 覆蓋率 >= 70%: 不會漏掉太多真實反轉點")
        print(f"   - 精確度 >= 50%: 能有效過濾假陽性")
    elif reversal_coverage >= 70:
        print(f"\n⚠️  中等潛力過濾器")
        print(f"   - 覆蓋率足夠，但精確度較低")
        print(f"   - 可作為輔助信號使用")
    else:
        print(f"\n❌ 低潛力過濾器")
        print(f"   - 覆蓋率不足，會漏掉太多真實反轉點")
    
    return {
        'signal_name': signal_name,
        'extremes_count': len(extremes),
        'peaks_count': len(peaks),
        'valleys_count': len(valleys),
        'matched_reversals': len(matched_reversals),
        'matched_extremes': len(matched_extremes),
        'reversal_coverage': reversal_coverage,
        'extreme_precision': extreme_precision,
        'extremes': extremes,
        'peaks': peaks,
        'valleys': valleys
    }


def analyze_multi_signal_combination(df, signal_results, true_reversals, price_extremes):
    """
    分析多個信號的組合效果
    
    策略：
    1. 價格極值（必須）
    2. + 其他信號的極值（可選組合）
    """
    print(f"\n{'='*80}")
    print(f"多信號組合分析")
    print(f"{'='*80}")
    
    # 過濾掉無效的信號
    valid_signals = [s for s in signal_results if s is not None and s['reversal_coverage'] >= 50]
    
    if not valid_signals:
        print("\n⚠️  沒有找到有效的輔助信號")
        return
    
    print(f"\n有效的輔助信號 (覆蓋率 >= 50%):")
    for sig in valid_signals:
        print(f"  - {sig['signal_name']}: 覆蓋率 {sig['reversal_coverage']:.1f}%, 精確度 {sig['extreme_precision']:.1f}%")
    
    # 測試不同的組合
    tolerance = 2
    
    print(f"\n{'='*80}")
    print(f"組合測試結果")
    print(f"{'='*80}")
    
    # 基準：只用價格
    print(f"\n基準 - 只用價格極值:")
    price_matched = 0
    for rev_idx in true_reversals:
        for ext_idx in price_extremes:
            if abs(rev_idx - ext_idx) <= tolerance:
                price_matched += 1
                break
    
    price_coverage = price_matched / len(true_reversals) * 100
    price_precision = price_matched / len(price_extremes) * 100
    
    print(f"  覆蓋率: {price_coverage:.1f}%")
    print(f"  精確度: {price_precision:.1f}%")
    print(f"  候選點: {len(price_extremes)}")
    
    # 測試每個信號的 AND 組合
    for sig in valid_signals:
        print(f"\n組合 - 價格 AND {sig['signal_name']}:")
        
        # 找出同時滿足兩個條件的點
        combined_extremes = []
        
        for price_ext in price_extremes:
            # 檢查附近是否有該信號的極值
            has_signal = False
            for sig_ext in sig['extremes']:
                if abs(price_ext - sig_ext) <= tolerance:
                    has_signal = True
                    break
            
            if has_signal:
                combined_extremes.append(price_ext)
        
        # 計算指標
        matched = 0
        for rev_idx in true_reversals:
            for ext_idx in combined_extremes:
                if abs(rev_idx - ext_idx) <= tolerance:
                    matched += 1
                    break
        
        coverage = matched / len(true_reversals) * 100 if len(true_reversals) > 0 else 0
        precision = matched / len(combined_extremes) * 100 if len(combined_extremes) > 0 else 0
        reduction = (len(price_extremes) - len(combined_extremes)) / len(price_extremes) * 100
        
        print(f"  覆蓋率: {coverage:.1f}% (變化: {coverage - price_coverage:+.1f}%)")
        print(f"  精確度: {precision:.1f}% (變化: {precision - price_precision:+.1f}%)")
        print(f"  候選點: {len(combined_extremes)} (減少: {reduction:.1f}%)")
        
        # 評估
        if coverage >= 90 and precision > price_precision:
            print(f"  ✅ 推薦組合！保持高覆蓋率且提升精確度")
        elif coverage >= 80 and reduction >= 20:
            print(f"  ⚠️  可考慮組合，但會損失一些覆蓋率")
        else:
            print(f"  ❌ 不推薦，覆蓋率損失太大")
    
    # 測試 OR 組合（擴大覆蓋範圍）
    print(f"\n{'='*80}")
    print(f"OR 組合測試（擴大覆蓋範圍）")
    print(f"{'='*80}")
    
    for sig in valid_signals:
        print(f"\n組合 - 價格 OR {sig['signal_name']}:")
        
        # 合併兩個信號的極值
        combined_extremes = np.unique(np.concatenate([price_extremes, sig['extremes']]))
        
        # 計算指標
        matched = 0
        for rev_idx in true_reversals:
            for ext_idx in combined_extremes:
                if abs(rev_idx - ext_idx) <= tolerance:
                    matched += 1
                    break
        
        coverage = matched / len(true_reversals) * 100 if len(true_reversals) > 0 else 0
        precision = matched / len(combined_extremes) * 100 if len(combined_extremes) > 0 else 0
        increase = (len(combined_extremes) - len(price_extremes)) / len(price_extremes) * 100
        
        print(f"  覆蓋率: {coverage:.1f}% (變化: {coverage - price_coverage:+.1f}%)")
        print(f"  精確度: {precision:.1f}% (變化: {precision - price_precision:+.1f}%)")
        print(f"  候選點: {len(combined_extremes)} (增加: {increase:.1f}%)")
        
        if coverage > price_coverage and precision >= price_precision * 0.9:
            print(f"  ✅ 推薦組合！提升覆蓋率且精確度損失小")


def main():
    """主函數"""
    print("="*80)
    print("🔬 多信號 Peak Valley 分析")
    print("="*80)
    
    # 載入數據
    df = load_data()
    if df is None:
        return
    
    true_reversals = set(np.where(df['reverse'].values)[0])
    
    print(f"\n數據概況:")
    print(f"  總數據點: {len(df)}")
    print(f"  真實反轉點: {len(true_reversals)}")
    
    # 首先分析價格（作為基準）
    print(f"\n{'='*80}")
    print(f"基準 - 價格 (close) 的 Peak Valley")
    print(f"{'='*80}")
    
    price_extremes, price_peaks, price_valleys = find_extremes_for_signal(
        df['close'].values, distance=20
    )
    
    print(f"  局部極值總數: {len(price_extremes)}")
    
    # 分析四個 z-score 信號
    signals = ['macd_2nd', 'price_2nd', 'atr_2nd', 'vol_2nd']
    
    signal_results = []
    for signal_name in signals:
        result = analyze_signal_extremes(df, signal_name, true_reversals, distance=20)
        signal_results.append(result)
    
    # 分析多信號組合
    analyze_multi_signal_combination(df, signal_results, true_reversals, price_extremes)
    
    # 總結
    print(f"\n{'='*80}")
    print(f"💡 總結與建議")
    print(f"{'='*80}")
    
    print("""
基於分析結果：

1. 單一信號評估：
   - 查看每個信號的覆蓋率和精確度
   - 覆蓋率 >= 70% 且精確度 >= 50% 的信號值得使用

2. AND 組合（提高精確度）：
   - 要求價格極值 AND 其他信號極值同時出現
   - 可以過濾掉一些假陽性
   - 但要注意不要損失太多覆蓋率

3. OR 組合（提高覆蓋率）：
   - 價格極值 OR 其他信號極值
   - 可以捕捉更多反轉點
   - 但可能增加假陽性

4. 實際應用建議：
   - 如果某個信號的 AND 組合能保持 90%+ 覆蓋率且提升精確度
     → 可以在階段1就使用
   - 如果覆蓋率損失較大
     → 留到階段2作為過濾特徵使用
    """)


if __name__ == '__main__':
    main()
