# 模型訓練目錄

這個目錄包含所有用於訓練機器學習模型的腳本和數據。

## 目錄結構

```
model_training/
├── lstm/              # LSTM 深度學習模型訓練
├── xgboost/           # XGBoost 梯度提升模型訓練
├── crash/             # 崩盤預測模型訓練
├── reversal/          # 反轉點檢測模型訓練
├── trend/             # 趨勢預測模型訓練
├── bollinger/         # 布林帶策略模型訓練
├── pattern/           # 價格模式識別訓練
└── strategy/          # 交易策略模型訓練
```

## 各模組說明

### LSTM (lstm/)
深度學習模型，用於時間序列預測
- `bidirectional_lstm_fixed.py` - 雙向 LSTM 模型
- `unidirectional_lstm.py` - 單向 LSTM 模型
- `generate_lstm_data.py` - 生成 LSTM 訓練數據
- `train_70percent_model.py` - 訓練高準確率模型
- `train_comprehensive_talib.py` - 使用 TA-Lib 指標訓練

**輸出模型**: `model_training/output/lstm/`

### XGBoost (xgboost/)
梯度提升樹模型，用於反轉點預測
- `prepare_data.py` - 準備訓練數據
- `train_xgboost.py` - 訓練 XGBoost 模型

**輸出模型**: `model_training/output/xgboost/`

### Crash (crash/)
崩盤預測模型，預測價格大幅下跌
- `create_balanced_4h_1h_model.py` - 4小時+1小時平衡模型
- `create_balanced_8h_1h_model.py` - 8小時+1小時平衡模型
- `create_balanced_crash_model.py` - 平衡崩盤預測模型
- `create_crash_prediction_dataset.py` - 創建崩盤預測數據集
- `train_combined_crash_model.py` - 組合崩盤模型
- `train_crash_detection_filtered.py` - 過濾式崩盤檢測

**輸出模型**: `model_training/output/crash/`

### Reversal (reversal/)
反轉點檢測模型
- `train_reversal_model.py` - 基礎反轉模型
- `train_reversal_model_v2.py` - 改進版反轉模型

**輸出模型**: `model_training/output/reversal/`

### Trend (trend/)
趨勢預測模型
- `train_trend_model.py` - 趨勢方向預測
- `train_separate_models.py` - 上升/下降分離模型

**輸出模型**: `model_training/output/trend/`

### Bollinger (bollinger/)
布林帶策略模型
- `collect_bb_patterns.py` - 收集布林帶模式
- `train_bb_stochrsi_model.py` - 布林帶 + StochRSI 模型
- `retrain_bb_model.py` - 重新訓練布林帶模型
- `retrain_bb_model_30_15.py` - 30/15 參數模型
- `retrain_bb_model_40_15.py` - 40/15 參數模型
- `retrain_bb_simple.py` - 簡化布林帶模型

**輸出模型**: `model_training/output/bollinger/`

### Pattern (pattern/)
價格模式識別
- `collect_specific_patterns.py` - 收集特定價格模式

**輸出模型**: `model_training/output/pattern/`

### Strategy (strategy/)
交易策略模型
- `retrain_optimized_strategy.py` - 優化策略模型
- `retrain_original_strategy.py` - 原始策略模型

**輸出模型**: `model_training/output/strategy/`

## 使用方式

### 1. 訓練新模型

```bash
# LSTM 模型
cd model_training/lstm
python bidirectional_lstm_fixed.py

# XGBoost 模型
cd model_training/xgboost
python prepare_data.py
python train_xgboost.py

# 崩盤預測模型
cd model_training/crash
python create_balanced_8h_1h_model.py
```

### 2. 模型輸出位置

所有訓練好的模型會自動保存到 `model_training/output/` 目錄的對應子目錄：
- LSTM 模型 → `model_training/output/lstm/`
- XGBoost 模型 → `model_training/output/xgboost/`
- 崩盤預測模型 → `model_training/output/crash/`
- 等等...

**重要**: 新訓練的模型先保存在 `output/` 中測試，確認效果後再複製到 `models/` 使用

### 3. 訓練數據

訓練數據存放在各模組的 `data/` 子目錄：
- `lstm/data/` - LSTM 訓練數據
- `xgboost/data/` - XGBoost 訓練數據

## 注意事項

1. **數據準備**: 訓練前確保有足夠的歷史數據
2. **計算資源**: LSTM 訓練需要較多計算資源，建議使用 GPU
3. **模型評估**: 訓練後檢查模型在驗證集上的表現
4. **版本管理**: 重要模型建議保存多個版本
5. **輸出路徑**: 所有腳本已更新為輸出到 `model_training/output/` 目錄
6. **模型部署**: 測試確認後才複製到 `models/` 目錄，避免影響正在運行的系統

## 相關目錄

- `model_training/output/` - 新訓練模型的輸出位置（測試用）
- `models/` - 生產環境使用的模型（回測和監控用）
- `experiments/` - 實驗性代碼和分析
- `data/` - 原始數據和數據提供者
- `output/` - 分析結果和圖表輸出

## 🔄 模型部署流程

```bash
# 1. 訓練模型
cd model_training/lstm
python bidirectional_lstm_fixed.py
# → 輸出到 model_training/output/lstm/

# 2. 測試模型
cd ../../experiments/testing
python test_ml_reversal.py

# 3. 確認效果後部署
cp model_training/output/lstm/bidirectional_lstm_model_fixed.keras \
   models/lstm/bidirectional_lstm_model_fixed.keras
```

詳細說明請參考 `model_training/output/README.md`
