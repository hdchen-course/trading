#!/usr/bin/env python3
"""
分析反轉點的多種場景和規律

檢查以下模式：
1. 價量背離 (price vs volume divergence)
2. 二階導數組合 (price_2nd, vol_2nd, macd_2nd, atr_2nd)
3. 技術指標極值組合
4. 多種規則組合來捕捉不同類型的反轉點
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np


def load_data(data_path='analysis_0911_0913 - data_with_reversal_labels.csv'):
    """載入數據"""
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def calculate_derivatives(df):
    """計算真實的二階導數（如果原始數據中都是0）"""
    # 計算一階導數
    df['price_1st'] = df['close'].diff()
    df['vol_1st'] = df['volume'].diff()
    df['macd_1st'] = df['macd'].diff()
    df['atr_1st'] = df['atr'].diff()
    
    # 計算二階導數
    df['price_2nd_calc'] = df['price_1st'].diff()
    df['vol_2nd_calc'] = df['vol_1st'].diff()
    df['macd_2nd_calc'] = df['macd_1st'].diff()
    df['atr_2nd_calc'] = df['atr_1st'].diff()
    
    return df


def analyze_divergence_patterns(df):
    """分析背離模式"""
    print("=" * 80)
    print("📊 背離模式分析")
    print("=" * 80)
    
    # 先計算一階導數（如果不存在）
    if 'price_1st' not in df.columns:
        df = calculate_derivatives(df)
    
    reversal_df = df[df['reverse'] == True].copy()
    
    # 定義背離條件
    divergence_patterns = []
    
    # 1. 價量背離 - 價格上漲但成交量下降
    pattern1 = (df['price_1st'] > 0) & (df['vol_1st'] < 0)
    divergence_patterns.append(("價漲量縮", pattern1))
    
    # 2. 價量背離 - 價格下跌但成交量上升
    pattern2 = (df['price_1st'] < 0) & (df['vol_1st'] > 0)
    divergence_patterns.append(("價跌量增", pattern2))
    
    # 3. MACD背離 - MACD上升但價格下跌
    pattern3 = (df['macd_1st'] > 0) & (df['price_1st'] < 0)
    divergence_patterns.append(("MACD漲價跌", pattern3))
    
    # 4. MACD背離 - MACD下降但價格上漲
    pattern4 = (df['macd_1st'] < 0) & (df['price_1st'] > 0)
    divergence_patterns.append(("MACD跌價漲", pattern4))
    
    # 5. 波動率背離 - ATR上升但價格變化小
    pattern5 = (df['atr_1st'] > 0) & (df['price_1st'].abs() < df['price_1st'].abs().median())
    divergence_patterns.append(("波動增價穩", pattern5))
    
    print(f"\n{'模式':<20} {'反轉點命中':>12} {'總信號':>10} {'精確度':>10} {'召回率':>10}")
    print("-" * 80)
    
    total_reversals = df['reverse'].sum()
    results = []
    
    for pattern_name, condition in divergence_patterns:
        hits = (condition & df['reverse']).sum()
        total_signals = condition.sum()
        precision = hits / total_signals * 100 if total_signals > 0 else 0
        recall = hits / total_reversals * 100 if total_reversals > 0 else 0
        
        print(f"{pattern_name:<20} {hits:>12} {total_signals:>10} {precision:>9.1f}% {recall:>9.1f}%")
        results.append((pattern_name, hits, total_signals, precision, recall))
    
    return results, df


def analyze_second_derivative_patterns(df):
    """分析二階導數組合模式"""
    print("\n" + "=" * 80)
    print("📈 二階導數組合模式分析")
    print("=" * 80)
    
    reversal_df = df[df['reverse'] == True].copy()
    total_reversals = df['reverse'].sum()
    
    # 檢查是否需要重新計算二階導數
    if df['price_2nd'].abs().sum() == 0:
        print("\n⚠️  原始數據中二階導數為0，使用計算值...")
        df = calculate_derivatives(df)
        price_2nd_col = 'price_2nd_calc'
        vol_2nd_col = 'vol_2nd_calc'
        macd_2nd_col = 'macd_2nd_calc'
        atr_2nd_col = 'atr_2nd_calc'
    else:
        price_2nd_col = 'price_2nd'
        vol_2nd_col = 'vol_2nd'
        macd_2nd_col = 'macd_2nd'
        atr_2nd_col = 'atr_2nd'
    
    # 定義二階導數組合模式
    second_deriv_patterns = []
    
    # 1. 價格加速上漲 (price_2nd > 0)
    pattern1 = df[price_2nd_col] > 0
    second_deriv_patterns.append(("價格加速上漲", pattern1))
    
    # 2. 價格加速下跌 (price_2nd < 0)
    pattern2 = df[price_2nd_col] < 0
    second_deriv_patterns.append(("價格加速下跌", pattern2))
    
    # 3. 成交量加速增加
    pattern3 = df[vol_2nd_col] > 0
    second_deriv_patterns.append(("成交量加速增", pattern3))
    
    # 4. 成交量加速減少
    pattern4 = df[vol_2nd_col] < 0
    second_deriv_patterns.append(("成交量加速減", pattern4))
    
    # 5. 價格加速 + 成交量減速 (可能的頂部)
    pattern5 = (df[price_2nd_col] > 0) & (df[vol_2nd_col] < 0)
    second_deriv_patterns.append(("價加速量減速", pattern5))
    
    # 6. 價格減速 + 成交量加速 (可能的底部)
    pattern6 = (df[price_2nd_col] < 0) & (df[vol_2nd_col] > 0)
    second_deriv_patterns.append(("價減速量加速", pattern6))
    
    # 7. MACD加速轉正
    pattern7 = df[macd_2nd_col] > 0
    second_deriv_patterns.append(("MACD加速轉正", pattern7))
    
    # 8. MACD加速轉負
    pattern8 = df[macd_2nd_col] < 0
    second_deriv_patterns.append(("MACD加速轉負", pattern8))
    
    # 9. 波動率加速上升
    pattern9 = df[atr_2nd_col] > 0
    second_deriv_patterns.append(("波動率加速升", pattern9))
    
    # 10. 三重加速 (price, vol, macd 都加速)
    pattern10 = (df[price_2nd_col] > 0) & (df[vol_2nd_col] > 0) & (df[macd_2nd_col] > 0)
    second_deriv_patterns.append(("三重加速", pattern10))
    
    print(f"\n{'模式':<20} {'反轉點命中':>12} {'總信號':>10} {'精確度':>10} {'召回率':>10}")
    print("-" * 80)
    
    results = []
    for pattern_name, condition in second_deriv_patterns:
        hits = (condition & df['reverse']).sum()
        total_signals = condition.sum()
        precision = hits / total_signals * 100 if total_signals > 0 else 0
        recall = hits / total_reversals * 100 if total_reversals > 0 else 0
        
        print(f"{pattern_name:<20} {hits:>12} {total_signals:>10} {precision:>9.1f}% {recall:>9.1f}%")
        results.append((pattern_name, hits, total_signals, precision, recall))
    
    return results, df


def analyze_extreme_combinations(df):
    """分析極值組合模式"""
    print("\n" + "=" * 80)
    print("🎯 技術指標極值組合分析")
    print("=" * 80)
    
    total_reversals = df['reverse'].sum()
    
    # 定義極值組合
    extreme_patterns = []
    
    # 1. 超賣區 + 下軌 + 高量
    pattern1 = (df['rsi'] < 30) & (df['bb_position'] < 0.2) & (df['volume_trend'] > 1.5)
    extreme_patterns.append(("超賣+下軌+高量", pattern1))
    
    # 2. 超買區 + 上軌 + 高量
    pattern2 = (df['rsi'] > 70) & (df['bb_position'] > 0.8) & (df['volume_trend'] > 1.5)
    extreme_patterns.append(("超買+上軌+高量", pattern2))
    
    # 3. 極度超賣 + MACD負值
    pattern3 = (df['rsi'] < 25) & (df['macd'] < 0)
    extreme_patterns.append(("極度超賣+MACD負", pattern3))
    
    # 4. 極度超買 + MACD正值
    pattern4 = (df['rsi'] > 75) & (df['macd'] > 0)
    extreme_patterns.append(("極度超買+MACD正", pattern4))
    
    # 5. BB下軌 + MACD直方圖轉正
    pattern5 = (df['bb_position'] < 0.2) & (df['macd_histogram'] > 0)
    extreme_patterns.append(("下軌+MACD轉正", pattern5))
    
    # 6. BB上軌 + MACD直方圖轉負
    pattern6 = (df['bb_position'] > 0.8) & (df['macd_histogram'] < 0)
    extreme_patterns.append(("上軌+MACD轉負", pattern6))
    
    # 7. 低波動 + 超賣 + 低量
    pattern7 = (df['volatility'] < df['volatility'].quantile(0.25)) & (df['rsi'] < 35) & (df['volume_trend'] < 0.8)
    extreme_patterns.append(("低波動+超賣+低量", pattern7))
    
    # 8. 高波動 + 極端位置
    pattern8 = (df['volatility'] > df['volatility'].quantile(0.75)) & ((df['bb_position'] < 0.1) | (df['bb_position'] > 0.9))
    extreme_patterns.append(("高波動+極端位置", pattern8))
    
    # 9. RSI背離 + BB中軌突破
    pattern9 = (df['rsi'].between(45, 55)) & (df['bb_position'].between(0.45, 0.55)) & (df['volume_trend'] > 1.3)
    extreme_patterns.append(("RSI中性+BB中軌+高量", pattern9))
    
    # 10. 價格低於SAR + RSI超賣
    pattern10 = (df['price_above_sar_ratio'] < 0.3) & (df['rsi'] < 35)
    extreme_patterns.append(("價低於SAR+超賣", pattern10))
    
    print(f"\n{'模式':<25} {'反轉點命中':>12} {'總信號':>10} {'精確度':>10} {'召回率':>10}")
    print("-" * 80)
    
    results = []
    for pattern_name, condition in extreme_patterns:
        hits = (condition & df['reverse']).sum()
        total_signals = condition.sum()
        precision = hits / total_signals * 100 if total_signals > 0 else 0
        recall = hits / total_reversals * 100 if total_reversals > 0 else 0
        
        print(f"{pattern_name:<25} {hits:>12} {total_signals:>10} {precision:>9.1f}% {recall:>9.1f}%")
        results.append((pattern_name, hits, total_signals, precision, recall))
    
    return results


def find_best_rule_combinations(df):
    """找出最佳規則組合"""
    print("\n" + "=" * 80)
    print("🏆 最佳規則組合搜索")
    print("=" * 80)
    
    total_reversals = df['reverse'].sum()
    
    # 收集所有有效的規則（精確度 > 5% 或召回率 > 5%）
    all_rules = []
    
    # 基礎規則
    basic_rules = [
        ("RSI<30", df['rsi'] < 30),
        ("RSI<25", df['rsi'] < 25),
        ("RSI>70", df['rsi'] > 70),
        ("BB<0.2", df['bb_position'] < 0.2),
        ("BB<0.1", df['bb_position'] < 0.1),
        ("BB>0.8", df['bb_position'] > 0.8),
        ("Vol>1.5", df['volume_trend'] > 1.5),
        ("Vol>2.0", df['volume_trend'] > 2.0),
        ("MACD<0", df['macd'] < 0),
        ("MACD>0", df['macd'] > 0),
        ("MACD_hist>0", df['macd_histogram'] > 0),
    ]
    
    # 測試所有兩兩組合
    print("\n尋找最佳兩規則組合...")
    best_combinations = []
    
    for i, (name1, cond1) in enumerate(basic_rules):
        for j, (name2, cond2) in enumerate(basic_rules[i+1:], i+1):
            combined_cond = cond1 & cond2
            hits = (combined_cond & df['reverse']).sum()
            total_signals = combined_cond.sum()
            
            if total_signals > 0:
                precision = hits / total_signals * 100
                recall = hits / total_reversals * 100
                
                # 只保留有意義的組合
                if precision > 10 or recall > 10:
                    best_combinations.append({
                        'name': f"{name1} AND {name2}",
                        'hits': hits,
                        'signals': total_signals,
                        'precision': precision,
                        'recall': recall,
                        'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    })
    
    # 按 F1 分數排序
    best_combinations.sort(key=lambda x: x['f1'], reverse=True)
    
    print(f"\n{'規則組合':<40} {'命中':>6} {'信號':>6} {'精確度':>8} {'召回率':>8} {'F1':>8}")
    print("-" * 80)
    
    for combo in best_combinations[:15]:  # 顯示前15個
        print(f"{combo['name']:<40} {combo['hits']:>6} {combo['signals']:>6} "
              f"{combo['precision']:>7.1f}% {combo['recall']:>7.1f}% {combo['f1']:>7.1f}")
    
    return best_combinations


def analyze_reversal_scenarios(df):
    """分析反轉點的不同場景"""
    print("\n" + "=" * 80)
    print("🔍 反轉點場景分類")
    print("=" * 80)
    
    reversal_df = df[df['reverse'] == True].copy()
    
    # 根據 RSI 和 BB position 分類
    scenarios = []
    
    # 場景1: 超賣反彈
    scenario1 = reversal_df[(reversal_df['rsi'] < 30) & (reversal_df['bb_position'] < 0.3)]
    scenarios.append(("超賣反彈", scenario1))
    
    # 場景2: 超買回落
    scenario2 = reversal_df[(reversal_df['rsi'] > 70) & (reversal_df['bb_position'] > 0.7)]
    scenarios.append(("超買回落", scenario2))
    
    # 場景3: 中性區突破
    scenario3 = reversal_df[(reversal_df['rsi'].between(40, 60)) & (reversal_df['bb_position'].between(0.4, 0.6))]
    scenarios.append(("中性區突破", scenario3))
    
    # 場景4: 高量反轉
    scenario4 = reversal_df[reversal_df['volume_trend'] > 1.5]
    scenarios.append(("高量反轉", scenario4))
    
    # 場景5: 低量反轉
    scenario5 = reversal_df[reversal_df['volume_trend'] < 0.8]
    scenarios.append(("低量反轉", scenario5))
    
    print(f"\n{'場景':<20} {'數量':>8} {'佔比':>10} {'平均RSI':>10} {'平均BB':>10} {'平均Vol':>10}")
    print("-" * 80)
    
    for scenario_name, scenario_df in scenarios:
        if len(scenario_df) > 0:
            count = len(scenario_df)
            percentage = count / len(reversal_df) * 100
            avg_rsi = scenario_df['rsi'].mean()
            avg_bb = scenario_df['bb_position'].mean()
            avg_vol = scenario_df['volume_trend'].mean()
            
            print(f"{scenario_name:<20} {count:>8} {percentage:>9.1f}% {avg_rsi:>10.2f} {avg_bb:>10.3f} {avg_vol:>10.2f}")
            
            # 顯示該場景的案例
            if count <= 3:
                print(f"\n  案例:")
                for idx, row in scenario_df.iterrows():
                    print(f"    {row['timestamp']}: RSI={row['rsi']:.1f}, BB={row['bb_position']:.2f}, Vol={row['volume_trend']:.1f}")
                print()


def main(data_path='analysis_0911_0913 - data_with_reversal_labels.csv'):
    """主函數"""
    print("=" * 80)
    print("🚀 反轉點場景與規律分析")
    print("=" * 80)
    
    # 載入數據
    df = load_data(data_path)
    print(f"\n✅ 數據載入完成: {len(df)} 筆資料, {df['reverse'].sum()} 個反轉點")
    
    # 1. 分析背離模式
    divergence_results, df = analyze_divergence_patterns(df)
    
    # 2. 分析二階導數模式
    second_deriv_results, df = analyze_second_derivative_patterns(df)
    
    # 3. 分析極值組合
    extreme_results = analyze_extreme_combinations(df)
    
    # 4. 找出最佳規則組合
    best_combinations = find_best_rule_combinations(df)
    
    # 5. 分析反轉場景
    analyze_reversal_scenarios(df)
    
    # 總結
    print("\n" + "=" * 80)
    print("📋 分析總結")
    print("=" * 80)
    
    print("\n💡 建議實作的規則（基於分析結果）:")
    print("\n1. 從最佳組合中選擇 F1 分數最高的前5個規則")
    print("2. 針對不同場景（超賣、超買、中性）設計專門規則")
    print("3. 結合背離模式和二階導數信號")
    print("4. 使用多規則投票機制提高準確度")
    
    print("\n✅ 分析完成！")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='反轉點場景與規律分析')
    parser.add_argument('--data', type=str, 
                        default='analysis_0911_0913 - data_with_reversal_labels.csv',
                        help='Path to training data CSV')
    
    args = parser.parse_args()
    
    main(args.data)
