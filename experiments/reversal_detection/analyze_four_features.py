"""
分析四個二階導數特徵的所有組合
Features: Price_2nd, Vol_2nd, MACD_2nd, ATR_2nd
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from pathlib import Path
from improved_stage1 import load_data as load_base_data, find_local_extremes
from analyze_price_2nd import calculate_price_2nd
from analyze_vol_2nd import calculate_vol_2nd
from analyze_macd_2nd import calculate_macd_2nd_zscore

def calculate_atr_2nd_zscore(df, window=200):
    """計算 ATR 二階導數的 z-score"""
    # 計算 ATR (Average True Range)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # 第一個值沒有前一個close
    
    # ATR (14期移動平均)
    atr = pd.Series(tr).rolling(window=14, min_periods=1).mean().values
    
    # 計算 ATR 的一階導數
    atr_1st = np.diff(atr, prepend=atr[0])
    
    # 計算 ATR 的二階導數
    atr_2nd = np.diff(atr_1st, prepend=atr_1st[0])
    
    # 計算 z-score
    atr_2nd_series = pd.Series(atr_2nd)
    rolling_mean = atr_2nd_series.rolling(window=window, min_periods=1).mean()
    rolling_std = atr_2nd_series.rolling(window=window, min_periods=1).std()
    
    atr_2nd_zscore = (atr_2nd_series - rolling_mean) / (rolling_std + 1e-10)
    
    return atr_2nd, atr_2nd_zscore.values

def check_sign_change(current_val, reversal_val):
    """檢查符號是否改變"""
    if pd.isna(current_val) or pd.isna(reversal_val):
        return False
    return np.sign(current_val) != np.sign(reversal_val)

def test_single_features(features):
    """測試單一特徵"""
    print("\n" + "="*80)
    print("1. 單一特徵測試")
    print("="*80)
    
    total_true = len(features[features['is_true'] == True])
    results = []
    
    feature_names = ['price_sign_changed', 'vol_sign_changed', 'macd_sign_changed', 'atr_sign_changed']
    display_names = ['Price_2nd', 'Vol_2nd', 'MACD_2nd', 'ATR_2nd']
    
    for feat_col, display_name in zip(feature_names, display_names):
        rule = features[features[feat_col] == True]
        rule_true = len(rule[rule['is_true'] == True])
        
        precision = rule_true / len(rule) * 100 if len(rule) > 0 else 0
        recall = rule_true / total_true * 100 if total_true > 0 else 0
        
        results.append({
            'Feature': display_name,
            'Candidates': len(rule),
            'Precision': precision,
            'Recall': recall
        })
        
        print(f"\n{display_name}:")
        print(f"  候選點: {len(rule)}")
        print(f"  精確度: {precision:.1f}%")
        print(f"  覆蓋率: {recall:.1f}%")
    
    return pd.DataFrame(results)

def test_two_and_combinations(features):
    """測試兩兩 AND 組合"""
    print("\n" + "="*80)
    print("2. 兩兩 AND 組合")
    print("="*80)
    
    total_true = len(features[features['is_true'] == True])
    results = []
    
    feature_cols = ['price_sign_changed', 'vol_sign_changed', 'macd_sign_changed', 'atr_sign_changed']
    display_names = ['Price', 'Vol', 'MACD', 'ATR']
    
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            f1_col, f2_col = feature_cols[i], feature_cols[j]
            f1_name, f2_name = display_names[i], display_names[j]
            
            rule = features[(features[f1_col] == True) & (features[f2_col] == True)]
            rule_true = len(rule[rule['is_true'] == True])
            
            precision = rule_true / len(rule) * 100 if len(rule) > 0 else 0
            recall = rule_true / total_true * 100 if total_true > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'Combination': f"{f1_name} AND {f2_name}",
                'Candidates': len(rule),
                'Precision': precision,
                'Recall': recall,
                'F1': f1_score
            })
            
            print(f"\n{f1_name} AND {f2_name}:")
            print(f"  候選點: {len(rule)}")
            print(f"  精確度: {precision:.1f}%")
            print(f"  覆蓋率: {recall:.1f}%")
            print(f"  F1分數: {f1_score:.1f}")
    
    return pd.DataFrame(results)

def test_three_and_combinations(features):
    """測試三個 AND 組合"""
    print("\n" + "="*80)
    print("3. 三個 AND 組合")
    print("="*80)
    
    total_true = len(features[features['is_true'] == True])
    results = []
    
    feature_cols = ['price_sign_changed', 'vol_sign_changed', 'macd_sign_changed', 'atr_sign_changed']
    display_names = ['Price', 'Vol', 'MACD', 'ATR']
    
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            for k in range(j+1, len(feature_cols)):
                f1_col, f2_col, f3_col = feature_cols[i], feature_cols[j], feature_cols[k]
                f1_name, f2_name, f3_name = display_names[i], display_names[j], display_names[k]
                
                rule = features[(features[f1_col] == True) & 
                               (features[f2_col] == True) & 
                               (features[f3_col] == True)]
                rule_true = len(rule[rule['is_true'] == True])
                
                precision = rule_true / len(rule) * 100 if len(rule) > 0 else 0
                recall = rule_true / total_true * 100 if total_true > 0 else 0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                results.append({
                    'Combination': f"{f1_name} AND {f2_name} AND {f3_name}",
                    'Candidates': len(rule),
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1_score
                })
                
                print(f"\n{f1_name} AND {f2_name} AND {f3_name}:")
                print(f"  候選點: {len(rule)}")
                print(f"  精確度: {precision:.1f}%")
                print(f"  覆蓋率: {recall:.1f}%")
                print(f"  F1分數: {f1_score:.1f}")
    
    return pd.DataFrame(results)

def test_four_and_combination(features):
    """測試四個 AND 組合"""
    print("\n" + "="*80)
    print("4. 四個 AND 組合")
    print("="*80)
    
    total_true = len(features[features['is_true'] == True])
    
    rule = features[(features['price_sign_changed'] == True) & 
                   (features['vol_sign_changed'] == True) & 
                   (features['macd_sign_changed'] == True) &
                   (features['atr_sign_changed'] == True)]
    rule_true = len(rule[rule['is_true'] == True])
    
    precision = rule_true / len(rule) * 100 if len(rule) > 0 else 0
    recall = rule_true / total_true * 100 if total_true > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPrice AND Vol AND MACD AND ATR:")
    print(f"  候選點: {len(rule)}")
    print(f"  精確度: {precision:.1f}%")
    print(f"  覆蓋率: {recall:.1f}%")
    print(f"  F1分數: {f1_score:.1f}")
    
    return pd.DataFrame([{
        'Combination': 'Price AND Vol AND MACD AND ATR',
        'Candidates': len(rule),
        'Precision': precision,
        'Recall': recall,
        'F1': f1_score
    }])

def test_best_combinations_with_atr(features):
    """測試之前最佳組合加上 ATR"""
    print("\n" + "="*80)
    print("5. 最佳組合 + ATR")
    print("="*80)
    
    total_true = len(features[features['is_true'] == True])
    results = []
    
    # 測試 1: Price AND MACD (之前最佳) vs Price AND MACD AND ATR
    print("\n比較 1: Price AND MACD vs Price AND MACD AND ATR")
    
    # Price AND MACD
    rule_pm = features[(features['price_sign_changed'] == True) & 
                       (features['macd_sign_changed'] == True)]
    rule_pm_true = len(rule_pm[rule_pm['is_true'] == True])
    precision_pm = rule_pm_true / len(rule_pm) * 100 if len(rule_pm) > 0 else 0
    recall_pm = rule_pm_true / total_true * 100 if total_true > 0 else 0
    
    print(f"\nPrice AND MACD:")
    print(f"  候選點: {len(rule_pm)}")
    print(f"  精確度: {precision_pm:.1f}%")
    print(f"  覆蓋率: {recall_pm:.1f}%")
    
    # Price AND MACD AND ATR
    rule_pma = features[(features['price_sign_changed'] == True) & 
                        (features['macd_sign_changed'] == True) &
                        (features['atr_sign_changed'] == True)]
    rule_pma_true = len(rule_pma[rule_pma['is_true'] == True])
    precision_pma = rule_pma_true / len(rule_pma) * 100 if len(rule_pma) > 0 else 0
    recall_pma = rule_pma_true / total_true * 100 if total_true > 0 else 0
    
    print(f"\nPrice AND MACD AND ATR:")
    print(f"  候選點: {len(rule_pma)}")
    print(f"  精確度: {precision_pma:.1f}%")
    print(f"  覆蓋率: {recall_pma:.1f}%")
    print(f"  精確度變化: {precision_pma - precision_pm:+.1f}%")
    print(f"  覆蓋率變化: {recall_pma - recall_pm:+.1f}%")
    
    results.append({
        'Combination': 'Price AND MACD',
        'Candidates': len(rule_pm),
        'Precision': precision_pm,
        'Recall': recall_pm
    })
    
    results.append({
        'Combination': 'Price AND MACD AND ATR',
        'Candidates': len(rule_pma),
        'Precision': precision_pma,
        'Recall': recall_pma
    })
    
    # 測試 2: Price AND Vol AND MACD (最高精確度) vs 加上 ATR
    print("\n" + "-"*80)
    print("比較 2: Price AND Vol AND MACD vs 加上 ATR")
    
    # Price AND Vol AND MACD
    rule_pvm = features[(features['price_sign_changed'] == True) & 
                        (features['vol_sign_changed'] == True) &
                        (features['macd_sign_changed'] == True)]
    rule_pvm_true = len(rule_pvm[rule_pvm['is_true'] == True])
    precision_pvm = rule_pvm_true / len(rule_pvm) * 100 if len(rule_pvm) > 0 else 0
    recall_pvm = rule_pvm_true / total_true * 100 if total_true > 0 else 0
    
    print(f"\nPrice AND Vol AND MACD:")
    print(f"  候選點: {len(rule_pvm)}")
    print(f"  精確度: {precision_pvm:.1f}%")
    print(f"  覆蓋率: {recall_pvm:.1f}%")
    
    # 四個全部 AND
    rule_all = features[(features['price_sign_changed'] == True) & 
                       (features['vol_sign_changed'] == True) &
                       (features['macd_sign_changed'] == True) &
                       (features['atr_sign_changed'] == True)]
    rule_all_true = len(rule_all[rule_all['is_true'] == True])
    precision_all = rule_all_true / len(rule_all) * 100 if len(rule_all) > 0 else 0
    recall_all = rule_all_true / total_true * 100 if total_true > 0 else 0
    
    print(f"\nPrice AND Vol AND MACD AND ATR:")
    print(f"  候選點: {len(rule_all)}")
    print(f"  精確度: {precision_all:.1f}%")
    print(f"  覆蓋率: {recall_all:.1f}%")
    print(f"  精確度變化: {precision_all - precision_pvm:+.1f}%")
    print(f"  覆蓋率變化: {recall_all - recall_pvm:+.1f}%")
    
    results.append({
        'Combination': 'Price AND Vol AND MACD',
        'Candidates': len(rule_pvm),
        'Precision': precision_pvm,
        'Recall': recall_pvm
    })
    
    results.append({
        'Combination': 'Price AND Vol AND MACD AND ATR',
        'Candidates': len(rule_all),
        'Precision': precision_all,
        'Recall': recall_all
    })
    
    return pd.DataFrame(results)

def extract_all_features(df, candidates, true_reversals):
    """提取所有四個二階導數特徵"""
    print("\n計算所有二階導數特徵...")
    
    # 計算 price_2nd
    price_2nd_raw, price_2nd_smooth = calculate_price_2nd(df, window=3)
    
    # 計算 vol_2nd
    vol_2nd_raw, vol_2nd_smooth = calculate_vol_2nd(df, window=3)
    
    # 計算 MACD_2nd z-score
    macd_2nd_raw, macd_2nd_zscore, macd_hist = calculate_macd_2nd_zscore(df, window=200)
    
    # 計算 ATR_2nd z-score
    atr_2nd_raw, atr_2nd_zscore = calculate_atr_2nd_zscore(df, window=200)
    
    # 添加到 DataFrame
    df['price_2nd_smooth'] = price_2nd_smooth
    df['vol_2nd_smooth'] = vol_2nd_smooth
    df['macd_2nd_zscore'] = macd_2nd_zscore
    df['atr_2nd_zscore'] = atr_2nd_zscore
    
    features = []
    
    for cand_idx in candidates:
        # 找出對應的反轉點
        reversal_idx = None
        for offset in [1, 2]:
            check_idx = cand_idx - offset
            if check_idx >= 0:
                reversal_idx = check_idx
                break
        
        if reversal_idx is None or reversal_idx < 5:
            continue
        
        # Price_2nd
        rev_price_2nd = df['price_2nd_smooth'].iloc[reversal_idx]
        curr_price_2nd = df['price_2nd_smooth'].iloc[cand_idx]
        price_sign_changed = (np.sign(rev_price_2nd) != np.sign(curr_price_2nd))
        
        # Vol_2nd
        rev_vol_2nd = df['vol_2nd_smooth'].iloc[reversal_idx]
        curr_vol_2nd = df['vol_2nd_smooth'].iloc[cand_idx]
        vol_sign_changed = (np.sign(rev_vol_2nd) != np.sign(curr_vol_2nd))
        
        # MACD_2nd
        rev_macd_2nd = df['macd_2nd_zscore'].iloc[reversal_idx]
        curr_macd_2nd = df['macd_2nd_zscore'].iloc[cand_idx]
        macd_sign_changed = (np.sign(rev_macd_2nd) != np.sign(curr_macd_2nd))
        
        # ATR_2nd
        rev_atr_2nd = df['atr_2nd_zscore'].iloc[reversal_idx]
        curr_atr_2nd = df['atr_2nd_zscore'].iloc[cand_idx]
        atr_sign_changed = (np.sign(rev_atr_2nd) != np.sign(curr_atr_2nd))
        
        # 判斷是否為真陽性
        is_true = reversal_idx in true_reversals
        
        features.append({
            'reversal_idx': reversal_idx,
            'is_true': is_true,
            'price_sign_changed': price_sign_changed,
            'vol_sign_changed': vol_sign_changed,
            'macd_sign_changed': macd_sign_changed,
            'atr_sign_changed': atr_sign_changed,
        })
    
    return pd.DataFrame(features)


def main():
    print("載入數據...")
    df = load_base_data()
    if df is None:
        return
    
    true_reversals = set(np.where(df['reverse'].values)[0])
    
    print(f"\n數據概況:")
    print(f"總數據點: {len(df)}")
    print(f"真實反轉點: {len(true_reversals)}")
    print(f"基準精確度: {len(true_reversals) / len(df) * 100:.1f}%")
    
    # 生成候選點
    print("\n生成候選點...")
    prices = df['close'].values
    extremes, peaks, valleys = find_local_extremes(prices, distance=20)
    
    candidates = []
    for i in range(20, len(df)):
        for offset in [1, 2]:
            check_idx = i - offset
            if check_idx in extremes:
                if i not in candidates:
                    candidates.append(i)
                break
    
    print(f"  候選點數量: {len(candidates)}")
    
    # 提取所有特徵
    features = extract_all_features(df, candidates, true_reversals)
    print(f"  提取了 {len(features)} 個候選點的特徵")
    
    # 執行所有測試
    results_single = test_single_features(features)
    results_two_and = test_two_and_combinations(features)
    results_three_and = test_three_and_combinations(features)
    results_four_and = test_four_and_combination(features)
    results_best = test_best_combinations_with_atr(features)
    
    # 保存結果
    print("\n" + "="*80)
    print("保存結果...")
    
    output_dir = Path('experiments/reversal_detection')
    
    results_single.to_csv(output_dir / 'four_features_single.csv', index=False)
    results_two_and.to_csv(output_dir / 'four_features_two_and.csv', index=False)
    results_three_and.to_csv(output_dir / 'four_features_three_and.csv', index=False)
    results_four_and.to_csv(output_dir / 'four_features_four_and.csv', index=False)
    results_best.to_csv(output_dir / 'four_features_best_comparison.csv', index=False)
    
    print("✅ 所有結果已保存")
    
    # 總結
    print("\n" + "="*80)
    print("📊 關鍵發現總結")
    print("="*80)
    
    # 找出最佳單一特徵
    best_single = results_single.loc[results_single['Precision'].idxmax()]
    print(f"\n最佳單一特徵: {best_single['Feature']}")
    print(f"  精確度: {best_single['Precision']:.1f}%")
    print(f"  覆蓋率: {best_single['Recall']:.1f}%")
    
    # 找出最佳兩個組合
    best_two = results_two_and.loc[results_two_and['Precision'].idxmax()]
    print(f"\n最佳兩個 AND 組合: {best_two['Combination']}")
    print(f"  精確度: {best_two['Precision']:.1f}%")
    print(f"  覆蓋率: {best_two['Recall']:.1f}%")
    
    # 找出最佳三個組合
    best_three = results_three_and.loc[results_three_and['Precision'].idxmax()]
    print(f"\n最佳三個 AND 組合: {best_three['Combination']}")
    print(f"  精確度: {best_three['Precision']:.1f}%")
    print(f"  覆蓋率: {best_three['Recall']:.1f}%")
    
    # 四個組合
    print(f"\n四個 AND 組合: {results_four_and['Combination'].iloc[0]}")
    print(f"  精確度: {results_four_and['Precision'].iloc[0]:.1f}%")
    print(f"  覆蓋率: {results_four_and['Recall'].iloc[0]:.1f}%")

if __name__ == '__main__':
    main()
