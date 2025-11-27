# 策略權重優化工作流程

## 概述

這個工作流程幫助你系統化地優化交易策略權重，並分析信號強度與盈虧的關係。

## 完整流程

### 步驟 1: 初始回測（180天）
測試每個策略的個別表現，獲取：
- 交易次數
- 總報酬率
- 勝率
- 詳細交易記錄

### 步驟 2: 計算新權重
根據步驟1的表現，使用以下方法之一計算新權重：
- `balanced`: 平衡勝率(60%)和報酬率(40%) **[推薦]**
- `win_rate`: 只看勝率
- `return`: 只看報酬率

### 步驟 3: 使用新權重回測
使用優化後的權重再次回測180天，驗證改進效果

### 步驟 4: 分析信號強度
分析信號強度與盈虧的關係：
- 信號強度分布（盈虧對比）
- 信號強度 vs 盈虧散點圖
- 各信號強度區間的勝率
- 各信號強度區間的平均報酬率

### 步驟 5: 生成優化建議
基於分析結果生成具體建議：
- 建議的信號強度閾值
- 應該避免的信號強度區間
- 預期改進效果

## 使用方法

### 完整流程（推薦）

```bash
# 執行完整的5步驟優化流程
python experiments/strategy_optimization/optimize_weights_workflow.py
```

這會自動執行所有步驟，並在 `output/` 目錄生成以下文件：

**步驟1輸出：**
- `step1_initial_backtest_{COIN}_{timestamp}.csv` - 每個策略的表現
- `backtest_{SYMBOL}_{STRATEGY}_{timestamp}.png` - 每個策略的回測圖表
- `trades_{SYMBOL}_{STRATEGY}_{timestamp}.csv` - 每個策略的交易記錄

**步驟2輸出：**
- `step2_new_weights_{COIN}_{timestamp}.csv` - 新計算的權重

**步驟3輸出：**
- `step3_optimized_backtest_{timestamp}.csv` - 優化後的回測結果
- `backtest_{SYMBOL}_Combined_{timestamp}.png` - 綜合策略圖表
- `trades_{SYMBOL}_Combined_{timestamp}.csv` - 綜合策略交易記錄（含信號強度）

**步驟4輸出：**
- `step4_signal_strength_analysis_{COIN}_{timestamp}.png` - 信號強度分析圖表
- `step4_signal_strength_details_{COIN}_{timestamp}.csv` - 詳細數據

**步驟5輸出：**
- `step5_recommendations_{timestamp}.csv` - 優化建議

## 輸出文件說明

### 信號強度分析圖表
包含4個子圖：
1. **信號強度分布** - 對比獲利和虧損交易的信號強度分布
2. **信號強度 vs 盈虧** - 散點圖顯示信號強度與盈虧的關係
3. **各區間勝率** - 不同信號強度區間的勝率
4. **各區間平均報酬** - 不同信號強度區間的平均報酬率

### 交易記錄CSV欄位

| 欄位 | 說明 |
|------|------|
| type | 交易類型（long/short） |
| entry_time | 進場時間 |
| exit_time | 出場時間 |
| entry_price | 進場價格 |
| exit_price | 出場價格 |
| pnl | 絕對盈虧（美元） |
| return_pct | 價格變動百分比 |
| leveraged_return_pct | 槓桿後報酬率 |
| exit_reason | 出場原因 |
| duration_hours | 持倉時間（小時） |
| capital_before | 交易前資金 |
| capital_after | 交易後資金 |
| **signal_strength** | **進場信號強度** ⭐ |

## 分析信號強度

### 如何解讀信號強度

信號強度 = 所有策略的加權分數總和的絕對值

例如：
- 策略A（權重0.3）給出買入信號(+1) → 貢獻 0.3
- 策略B（權重0.2）給出買入信號(+1) → 貢獻 0.2
- 策略C（權重0.5）沒有信號(0) → 貢獻 0.0
- **總信號強度 = 0.3 + 0.2 = 0.5**

### 典型模式

**模式1: 低信號強度虧損多**
```
虧損交易平均信號強度: 0.25
獲利交易平均信號強度: 0.65
→ 建議: 過濾掉信號強度 < 0.45 的交易
```

**模式2: 特定區間表現差**
```
信號強度 0.2-0.4: 勝率 30%
信號強度 0.6-0.8: 勝率 65%
→ 建議: 只接受信號強度 > 0.5 的交易
```

**模式3: 極端信號也不好**
```
信號強度 0.8-1.0: 勝率 45%（過度自信）
信號強度 0.4-0.6: 勝率 60%（適中最好）
→ 建議: 避免極端信號，保持在中等強度
```

## 應用優化結果

### 方法1: 更新配置文件

根據步驟2的輸出，手動更新 `config/coin_configs.py` 中的權重：

```python
# 原始權重
ReversalStrategy(
    StrategyConfig(
        name="Reversal A/V Turn",
        params={},
        weight=0.25,  # 舊權重
    )
),

# 更新為新權重（從 step2_new_weights_*.csv 獲取）
ReversalStrategy(
    StrategyConfig(
        name="Reversal A/V Turn",
        params={},
        weight=0.35,  # 新權重
    )
),
```

### 方法2: 添加信號強度過濾

在策略中添加信號強度過濾邏輯（未來功能）：

```python
# 在 trading/backtest.py 中添加
MIN_SIGNAL_STRENGTH = 0.45  # 從步驟5的建議獲取

if signal == 1 and position is None:
    current_strength = signal_strength.iloc[i]
    if current_strength >= MIN_SIGNAL_STRENGTH:
        # 進場
        position = {...}
```

## 注意事項

1. **數據充足性**: 180天數據確保有足夠的交易樣本
2. **過度擬合**: 不要過度優化，保留一定的泛化能力
3. **市場變化**: 定期重新執行優化（建議每季度一次）
4. **交易成本**: 考慮手續費對小信號的影響
5. **風險管理**: 信號強度過濾可能減少交易機會

## 故障排除

### 問題1: 沒有 signal_strength 欄位
**原因**: 使用舊版本的 backtest.py
**解決**: 確保已更新 `trading/backtest.py` 和 `strategies/base.py`

### 問題2: 某些策略沒有交易
**原因**: 策略參數不適合當前市場
**解決**: 調整策略參數或移除該策略

### 問題3: 所有策略表現都很差
**原因**: 市場環境不適合當前策略組合
**解決**: 考慮添加新策略或調整時間範圍

## 進階用法

### 自定義權重計算方法

編輯 `optimize_weights_workflow.py` 中的 `step2_calculate_new_weights` 函數：

```python
# 添加自定義方法
elif method == 'custom':
    scores = []
    for s in valid_strategies:
        # 自定義評分邏輯
        custom_score = (
            s['win_rate'] * 0.4 +
            max(0, s['total_return']) * 0.3 +
            min(1.0, s['trades'] / 100) * 0.3  # 交易頻率
        )
        scores.append(custom_score)
```

### 分析特定時間段

修改 `main()` 函數中的天數參數：

```python
# 只分析最近30天
strategy_performance = step1_initial_backtest(days=30)
```

## 相關文件

- `trading/backtest.py` - 回測引擎（已添加信號強度記錄）
- `strategies/base.py` - 策略基類（已添加信號強度計算）
- `config/coin_configs.py` - 策略配置文件
- `experiments/reversal_detection/` - 反轉檢測實驗

## 更新日誌

- 2024-01-XX: 初始版本，完整5步驟工作流程
- 2024-01-XX: 添加信號強度記錄和分析功能
