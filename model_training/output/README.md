# 模型訓練輸出目錄

這個目錄用於存放新訓練的模型，在確認模型效果後再複製到 `models/` 目錄使用。

## 📁 目錄結構

```
output/
├── lstm/        # LSTM 模型輸出
├── xgboost/     # XGBoost 模型輸出
├── crash/       # 崩盤預測模型輸出
├── reversal/    # 反轉模型輸出
├── trend/       # 趨勢模型輸出
├── bollinger/   # 布林帶模型輸出
├── pattern/     # 模式識別模型輸出
└── strategy/    # 策略模型輸出
```

## 🔄 工作流程

### 1. 訓練模型
```bash
cd model_training/lstm
python bidirectional_lstm_fixed.py
# 模型保存到: model_training/output/lstm/bidirectional_lstm_model_fixed.keras
```

### 2. 測試模型
在 `experiments/testing/` 中測試新訓練的模型：
```bash
cd experiments/testing
python test_ml_reversal.py  # 使用 output/ 中的模型測試
```

### 3. 評估效果
- 檢查準確率、召回率等指標
- 進行回測驗證
- 與現有模型比較

### 4. 部署模型
確認效果良好後，複製到 `models/` 目錄：
```bash
# 備份舊模型
cp models/lstm/bidirectional_lstm_model_fixed.keras \
   models/lstm/bidirectional_lstm_model_fixed.keras.backup

# 部署新模型
cp model_training/output/lstm/bidirectional_lstm_model_fixed.keras \
   models/lstm/
```

## ⚠️ 重要提醒

1. **不要直接覆蓋 models/ 中的模型**
   - `models/` 目錄中的模型正在被回測和監控系統使用
   - 先在 `output/` 中訓練和測試

2. **保留訓練記錄**
   - 記錄訓練參數、數據集、準確率等信息
   - 建議創建訓練日誌文件

3. **版本管理**
   - 重要模型建議加上日期或版本號
   - 例如：`lstm_model_20251125.keras`

4. **清理舊文件**
   - 定期清理 `output/` 中不需要的模型
   - 保持目錄整潔

## 📝 模型部署檢查清單

- [ ] 模型訓練完成，無錯誤
- [ ] 在測試集上驗證準確率
- [ ] 進行回測，檢查實際效果
- [ ] 與現有模型比較性能
- [ ] 備份現有模型
- [ ] 複製新模型到 `models/`
- [ ] 更新相關文檔
- [ ] 測試系統是否正常運行

## 🎯 最佳實踐

### 訓練日誌範例
創建 `training_log.md` 記錄訓練信息：
```markdown
# 訓練記錄

## 2025-11-25 - LSTM 模型訓練

**模型**: bidirectional_lstm_model_fixed.keras
**數據集**: 2024-01-01 to 2025-11-01
**訓練參數**:
- Epochs: 50
- Batch size: 32
- Learning rate: 0.001

**結果**:
- 訓練準確率: 75.3%
- 驗證準確率: 72.1%
- 測試準確率: 71.8%

**決定**: 準確率提升 3%，部署到生產環境
```

### 模型命名建議
```
# 包含日期
lstm_model_20251125.keras

# 包含版本號
xgboost_reversal_v2.pkl

# 包含準確率
crash_model_75pct.pkl

# 包含特殊標記
trend_model_experimental.pkl
```
