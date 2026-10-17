# 加密貨幣順勢訊號系統(誠實重建版)

> 分支 `rebuild/honest-core`。這是把舊系統(回測灌水、前視偏誤、ML 亂燉)**打掉重建**的可信版本。
> 目標:**不騙自己** — 正期望值、回撤可控、樣本外仍為正、純規則可解釋、老機器也能跑。

## 為什麼重建

舊系統(`main.py`、`detection/`、`strategies/`、`trading/`)的回測數字**不可信**,主因:

1. **前視偏誤**:`detection/triple_confirmation_reversal.py` 用「未來 1~2 根」確認反轉卻把訊號標在極值當根 → 回測精準買底賣頂、實盤無法重現。該策略占舊 BTC 權重 28.6%。
2. **方向 bug**:`MLReversalV7Strategy` 回傳 0/1/2 而非 -1/0/1,做空被當成買進。
3. **ML 作弊**:V6/V7 用隨機 `train_test_split` 而非時間序列切分 → 訓練分數虛高、實盤破功。

上述舊檔案**保留但視為 legacy、不可信**,新系統完全不依賴它們。

## 新系統(可信核心)

| 模組 | 作用 |
|---|---|
| `engine/backtester.py` | 誠實回測:**下一根 open 成交**、盤中 high/low 判停損停利、**ATR 移動停損**、**每筆固定風險控管**、完整績效(回撤/期望值/獲利因子/Sharpe) |
| `engine/metrics.py` | 績效指標(不只勝率,看期望值/賺賠比/回撤) |
| `engine/data.py` | 資料載入 / 多週期聚合 / 指標(**純 pandas,無 TA-Lib**) |
| `engine/fetch_binance.py` | 幣安 USDT 永續 K 線抓取(**純標準庫**,老 macOS 可跑),快取於 `data/cache/` |
| `signals/library.py` | 可解釋訊號:Donchian 突破、EMA 交叉、EMA ribbon(90/120/240/300/420) |
| `engine/strategy.py` | 生產策略層:輸出「現在該不該進、方向、建議初始停損」 |
| `engine/notify.py` | Telegram 通知(讀環境變數,未設定時安全略過) |
| `scripts/research_signals.py` | 單訊號樣本內/外研究 |
| `scripts/research_portfolio.py` | 多幣種 × 多訊號 組合研究 |
| `scripts/live_signals.py` | 當前訊號狀態儀表板 |
| `scripts/run_live.py` | 即時監控入口(抓最新資料→找觸發→推 Telegram) |

依賴:可攜核心只需 `pandas numpy`(抓資料/通知只用標準庫 `urllib`)。**不需要 TA-Lib、不需要 torch/xgboost。**

## 快速開始

```bash
# 1. 用專案 venv(Python 3.12)
./venv/bin/python -m pip install -q pytest     # 首次需要

# 2. 跑單元測試(驗證引擎正確性 — 使用者不需看懂程式,只看綠燈)
./venv/bin/python -m pytest tests/ -q

# 3. 抓多幣種資料(快取到 data/cache/)
./venv/bin/python -I engine/fetch_binance.py BTCUSDT ETHUSDT SOLUSDT ...

# 4. 研究:單訊號 / 多幣種組合
./venv/bin/python scripts/research_signals.py
./venv/bin/python scripts/research_portfolio.py

# 5. 即時訊號(可選 Telegram)
export TELEGRAM_BOT_TOKEN="..."; export TELEGRAM_CHAT_ID="..."   # 可選
./venv/bin/python scripts/run_live.py
```

## 誠實的績效期望(真實資料、扣手續費+滑價、含樣本外)

10 幣 × 3 順勢訊號組合,回撤控制在 ~18% 預算內:

| 期間 | 年化 | 最大回撤 | Sharpe |
|---|---|---|---|
| 全期(6.8 年) | +15% | 14% | 1.01 |
| 樣本外(近 2 年) | +5% | 18% | 0.39 |

**合理期望:年化 5~15%、回撤 15~18%、約 1/3 的週為正。**
**「每週 +2%」用誠實方法做不到** — 要衝那種報酬只能把槓桿開到回撤爆掉(就是舊系統「會賠死」的路)。
順勢策略有好年也有盤整的壞年,不保證每週/每月為正。

## 策略邏輯(可解釋)

- **方向**:高週期(4h/8h/12h)順勢 — EMA 50/200 交叉、Donchian 20 突破、EMA ribbon 排列。
- **出場**:ATR 移動停損(讓獲利奔跑、回撤追蹤出場),初始硬停損 6%。
- **部位**:每筆只賭固定比例權益(風險基礎 sizing),直接控制回撤。
- **分散**:同邏輯套多幣種,降低單幣雜訊、提高 Sharpe。
