# 視覺化改進 - 智能價格標註

## 問題

在 `Last 1 Month Detail` 圖表中，當信號數量很多時（例如103個），只標註前10個信號的價格，導致用戶困惑為什麼其他信號沒有標籤。

## 解決方案

實現智能標註策略，根據信號數量自動調整標註方式：

### 標註策略

| 信號數量 | 標註策略 | 說明 |
|---------|---------|------|
| ≤ 20 | 全部標註 | 信號少，可以全部顯示 |
| 21-50 | 每隔一個標註 | 中等數量，標註一半避免擁擠 |
| > 50 | 只標註極值 | 信號多，只標註最高和最低各5個 |

### 實現細節

```python
# 智能標註價格：根據信號數量調整標註策略
n_signals = len(signal_times)
if n_signals <= 20:
    # 少於20個信號：全部標註
    for time, price in zip(signal_times, signal_prices):
        ax3.annotate(f'${price:.0f}', ...)
elif n_signals <= 50:
    # 20-50個信號：每隔一個標註
    for i, (time, price) in enumerate(zip(signal_times, signal_prices)):
        if i % 2 == 0:
            ax3.annotate(f'${price:.0f}', ...)
else:
    # 超過50個信號：只標註極值點（最高和最低的各5個）
    prices_array = np.array(signal_prices)
    top_5_idx = np.argsort(prices_array)[-5:]
    bottom_5_idx = np.argsort(prices_array)[:5]
    important_idx = set(top_5_idx) | set(bottom_5_idx)
    
    for i, (time, price) in enumerate(zip(signal_times, signal_prices)):
        if i in important_idx:
            ax3.annotate(f'${price:.0f}', ...)
```

### 圖表標題改進

圖表標題現在會顯示使用的標註策略：

- `Last 1 Month Detail (15 signals, all signals labeled)`
- `Last 1 Month Detail (35 signals, every 2nd signal labeled)`
- `Last 1 Month Detail (103 signals, top/bottom 10 signals labeled)`

## 使用範例

```bash
# 測試少量信號（全部標註）
python main.py --visualize --start 2025-09-01 --end 2025-09-10 --methods triple

# 測試中等信號（每隔一個標註）
python main.py --visualize --start 2025-09-01 --end 2025-09-20 --methods triple

# 測試大量信號（只標註極值）
python main.py --visualize --start 2024-01-01 --end 2024-12-31 --methods triple
```

## 效果

### 改進前
- ❌ 只標註前10個信號
- ❌ 用戶不知道為什麼其他信號沒有標籤
- ❌ 固定策略不適應不同情況

### 改進後
- ✅ 根據信號數量智能調整
- ✅ 圖表標題說明標註策略
- ✅ 少量信號全部標註，大量信號標註重點
- ✅ 避免圖表過於擁擠

## 其他改進建議

如果未來需要進一步改進，可以考慮：

1. **交互式標註**: 使用 plotly 等交互式圖表庫，鼠標懸停顯示價格
2. **顏色編碼**: 用顏色深淺表示價格高低
3. **分組標註**: 將相近的信號分組，只標註組代表
4. **可配置**: 允許用戶通過參數控制標註策略

## 相關文件

- `scripts/visualize.py`: 視覺化主程式
- `main.py`: 命令行入口
- `detection/triple_confirmation_reversal.py`: 三重確認檢測器
