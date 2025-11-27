# 三重確認反轉檢測器使用指南

## 📊 檢測器說明

**TripleConfirmationReversalDetector** 使用兩階段檢測：
- **Stage 1**: 局部極值檢測（高召回率）
- **Stage 2**: Price_2nd AND Vol_2nd AND MACD_2nd 過濾（高精確度 81.2%）

## 🚀 使用方法

### 1. 可視化檢測結果

```bash
# 只顯示三重確認檢測
python main.py --visualize --start 2024-09-01 --end 2024-09-10 --methods triple

# 與其他方法對比
python main.py --visualize --start 2024-09-01 --end 2024-09-10 --methods triple peak macd

# 指定輸出文件
python main.py --visualize --start 2024-09-01 --end 2024-09-10 --methods triple --output triple_signals.png
```

### 2. 運行性能測試

```bash
# 在完整數據集上測試
python experiments/reversal_detection/test_triple_confirmation.py
```

## 📈 預期性能

根據歷史數據分析（2022-01-01 to 2025-09-01）：

| 指標 | 數值 |
|------|------|
| **精確度** | 81.2% |
| **覆蓋率** | 40.4% |
| **每天訊號** | 1.2 個 |
| **每週訊號** | 8.4 個 |
| **每月訊號** | 36 個 |

## 🎯 適用場景

### ✅ 適合使用的情況

- 追求高勝率交易（81.2%）
- 不想頻繁交易
- 可以接受每天1-2個訊號
- 風險厭惡型交易者

### ⚠️ 不適合的情況

- 需要高頻交易
- 不能錯過任何機會
- 需要覆蓋所有反轉點

## 🔧 參數配置

```python
from detection import TripleConfirmationReversalDetector

# 創建檢測器
detector = TripleConfirmationReversalDetector(
    distance=20,      # 局部極值的最小距離
    window_size=16    # 滾動窗口大小（用於計算指標）
)

# 檢測反轉點
signals = detector.detect(df)  # 返回布爾數組
```

## 📊 與其他方法對比

| 方法 | 精確度 | 覆蓋率 | 每天訊號 |
|------|--------|--------|---------|
| **Triple Confirmation** | **81.2%** | 40.4% | 1.2 |
| Peak Valley | ~45% | ~90% | 3-4 |
| MACD Reversal | ~60% | ~70% | 2-3 |

## 🔍 檢測原理

### Stage 1: 局部極值檢測
使用 `scipy.signal.find_peaks` 找出價格的峰值和谷值

### Stage 2: 三重確認過濾
檢查以下三個條件是否**同時**滿足：

1. **Price_2nd 符號改變**: 價格二階導數的符號改變
2. **Vol_2nd 符號改變**: 成交量二階導數的符號改變  
3. **MACD_2nd 符號改變**: MACD 二階導數的符號改變

只有三個條件都滿足時，才確認為反轉點。

## 📝 輸出格式

檢測器返回一個布爾數組，與輸入 DataFrame 長度相同：
- `True`: 該點為檢測到的反轉點
- `False`: 該點不是反轉點

```python
signals = detector.detect(df)
# signals 是 numpy.ndarray，dtype=bool

# 獲取反轉點的索引
reversal_indices = np.where(signals)[0]

# 獲取反轉點的時間和價格
reversal_times = df.iloc[reversal_indices]['timestamp']
reversal_prices = df.iloc[reversal_indices]['close']
```

## 🎨 可視化顏色

在圖表中，三重確認檢測的信號點顯示為：
- **顏色**: 紅色 (red)
- **標記**: 默認圓點

## 📚 相關文檔

- `ALL_COMBINATIONS_ANALYSIS_SUMMARY.md` - 三特徵組合分析
- `FOUR_FEATURES_ANALYSIS_SUMMARY.md` - 四特徵組合分析（包含 ATR）
- `PRICE_2ND_ANALYSIS_SUMMARY.md` - Price_2nd 特徵分析
- `VOL_2ND_ANALYSIS_SUMMARY.md` - Vol_2nd 特徵分析

## 💡 使用建議

1. **首次使用**: 先在小範圍時間段測試（如1週）
2. **參數調整**: 如果訊號太少，可以降低 `distance` 參數
3. **結合其他指標**: 可以與 RSI、布林帶等指標結合使用
4. **回測驗證**: 在實際交易前，務必進行充分的回測

## ⚙️ 技術細節

### 依賴的工具函數

- `utils.indicators.calculate_technical_indicators()` - 計算基礎技術指標
- `utils.indicators.analyze_cumulative_metrics()` - 計算二階導數和 z-score
- `scipy.signal.find_peaks()` - 尋找局部極值

### 計算流程

1. 計算技術指標（MACD, ATR, EMA 等）
2. 使用滾動窗口計算二階導數
3. 對二階導數進行 z-score 標準化
4. 尋找局部極值點
5. 對每個極值點檢查三重確認條件
6. 返回通過過濾的反轉點

## 🐛 故障排除

### 問題：沒有檢測到任何信號

**可能原因**:
- 時間範圍太短
- 數據質量問題
- 參數設置過於嚴格

**解決方案**:
- 擴大時間範圍（至少1週）
- 檢查數據是否完整
- 降低 `distance` 參數

### 問題：訊號太多

**可能原因**:
- `distance` 參數太小
- 數據波動性太大

**解決方案**:
- 增加 `distance` 參數（如 30 或 40）
- 考慮使用更長的時間框架

## 📞 支持

如有問題或建議，請查看：
- 實驗腳本：`experiments/reversal_detection/`
- 測試腳本：`experiments/reversal_detection/test_triple_confirmation.py`
