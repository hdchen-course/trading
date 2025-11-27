#!/usr/bin/env python3
"""
測試 LSTM 模型在反轉檢測上的表現

檢查現有的 Bidirectional LSTM 模型是否比規則方法更好
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def test_lstm_detection():
    """
    測試 LSTM 模型的反轉檢測表現
    """
    print("=" * 80)
    print("🧪 測試 LSTM 模型表現")
    print("=" * 80)
    
    # 載入數據
    data_path = 'training_data/reversal_training_data_2022-01-01_to_2025-09-01_15m.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ 數據文件不存在: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"\n✅ 數據載入完成: {len(df)} 筆資料")
    print(f"   反轉點: {df['reverse'].sum()} 個 ({df['reverse'].sum()/len(df)*100:.2f}%)")
    
    # 檢查 LSTM 模型是否存在
    model_path = 'models/lstm/bidirectional_lstm_model_fixed.keras'
    scaler_path = 'models/lstm/scaler.pkl'
    
    if not os.path.exists(model_path):
        print(f"\n❌ LSTM 模型不存在: {model_path}")
        print("   請先訓練 LSTM 模型")
        return
    
    if not os.path.exists(scaler_path):
        print(f"\n❌ Scaler 不存在: {scaler_path}")
        return
    
    print(f"\n✅ LSTM 模型存在")
    
    # 嘗試使用 LSTM 檢測
    try:
        from detection import LSTMDetection
        
        print("\n🔧 初始化 LSTM 檢測器...")
        detector = LSTMDetection()
        
        print("🔧 運行 LSTM 檢測...")
        signals = detector.detect(df)
        
        # 評估結果
        y_true = df['reverse'].values
        y_pred = signals
        
        # 計算指標
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        
        print("\n" + "=" * 80)
        print("📊 LSTM 模型結果")
        print("=" * 80)
        print(f"\n精確度 (Precision): {precision*100:.2f}%")
        print(f"召回率 (Recall):    {recall*100:.2f}%")
        print(f"F1 分數:           {f1:.4f} ({f1*100:.2f}%)")
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print(f"\n混淆矩陣:")
            print(f"  真陽性 (TP): {tp}")
            print(f"  假陽性 (FP): {fp}")
            print(f"  假陰性 (FN): {fn}")
            print(f"  真陰性 (TN): {tn}")
            
            if tp + fp > 0:
                print(f"\n信號統計:")
                print(f"  總信號數: {tp + fp}")
                print(f"  正確信號: {tp} ({tp/(tp+fp)*100:.1f}%)")
                print(f"  錯誤信號: {fp} ({fp/(tp+fp)*100:.1f}%)")
        
        # 與規則方法比較
        print("\n" + "=" * 80)
        print("📊 與規則方法比較")
        print("=" * 80)
        
        print("\n規則方法 (極端位置 + 波動加速):")
        print(f"  F1 分數: 15.3%")
        print(f"  精確度:  10.6%")
        print(f"  召回率:  27.5%")
        
        print(f"\nLSTM 模型:")
        print(f"  F1 分數: {f1*100:.1f}%")
        print(f"  精確度:  {precision*100:.1f}%")
        print(f"  召回率:  {recall*100:.1f}%")
        
        # 判斷哪個更好
        rule_f1 = 0.153
        if f1 > rule_f1:
            improvement = (f1 - rule_f1) / rule_f1 * 100
            print(f"\n🏆 LSTM 模型勝出！")
            print(f"   F1 分數提升: {improvement:.1f}%")
        elif f1 > rule_f1 * 0.9:
            print(f"\n⚖️  LSTM 模型表現接近規則方法")
            print(f"   差距: {abs(f1 - rule_f1)*100:.1f}%")
        else:
            decline = (rule_f1 - f1) / rule_f1 * 100
            print(f"\n❌ 規則方法仍然更好")
            print(f"   LSTM F1 分數低: {decline:.1f}%")
        
        # 結論
        print("\n" + "=" * 80)
        print("💡 結論")
        print("=" * 80)
        
        if f1 > rule_f1:
            print("\n✅ 建議使用 LSTM 模型作為主要檢測方法")
            print("   - 更高的 F1 分數")
            print("   - 可以學習複雜的時序模式")
        else:
            print("\n建議:")
            if f1 < 0.05:
                print("   ❌ LSTM 模型表現不佳，建議:")
                print("      1. 重新訓練模型（使用更多數據）")
                print("      2. 調整模型架構")
                print("      3. 優化特徵工程")
            else:
                print("   ⚖️  可以考慮 LSTM 作為輔助方法:")
                print("      1. 與規則方法組合使用")
                print("      2. 用於信號確認")
                print("      3. 繼續優化 LSTM 模型")
        
    except Exception as e:
        print(f"\n❌ LSTM 檢測失敗: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n可能的原因:")
        print("  1. 模型文件損壞")
        print("  2. 數據格式不匹配")
        print("  3. 缺少必要的特徵欄位")


if __name__ == '__main__':
    test_lstm_detection()
