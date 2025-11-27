import pandas as pd
import numpy as np
from crypto_strategies import MLReversalStrategy, StrategyConfig

# 讀取數據
print("📊 讀取BTC數據...")
df = pd.read_csv("training_data/btc_12months_15m.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 取最近3個月數據測試
recent_data = df.tail(8640)  # 3個月 * 30天 * 24小時 * 4 (15分鐘)
print(f"測試數據: {len(recent_data)} 行")

# 創建策略配置
config = StrategyConfig(name="ML_Reversal", params={}, weight=1.0)

# 初始化策略
strategy = MLReversalStrategy(config)

if strategy.model is None:
    print("❌ 模型未載入，無法測試")
    exit()

# 生成信號
print("🤖 生成ML轉折信號...")
signals = strategy.generate_signals(recent_data)

# 分析信號
buy_signals = signals[signals == 1]
sell_signals = signals[signals == -1]

print(f"\n📈 信號統計:")
print(f"買入信號: {len(buy_signals)} 個")
print(f"賣出信號: {len(sell_signals)} 個")

# 顯示最近的信號
if len(buy_signals) > 0:
    print(f"\n🟢 最近5個買入信號:")
    for idx in buy_signals.tail(5).index:
        timestamp = recent_data.loc[idx, "timestamp"]
        price = recent_data.loc[idx, "close"]
        print(f"  {timestamp}: {price:.2f}")

if len(sell_signals) > 0:
    print(f"\n🔴 最近5個賣出信號:")
    for idx in sell_signals.tail(5).index:
        timestamp = recent_data.loc[idx, "timestamp"]
        price = recent_data.loc[idx, "close"]
        print(f"  {timestamp}: {price:.2f}")

# 簡單回測
print(f"\n💰 簡單回測 (最近100個信號):")
all_signals = signals[signals != 0].tail(100)
if len(all_signals) > 0:
    trades = []
    position = None

    for idx in all_signals.index:
        signal = signals.loc[idx]
        price = recent_data.loc[idx, "close"]
        timestamp = recent_data.loc[idx, "timestamp"]

        if signal == 1 and position is None:  # 買入
            position = {"entry_price": price, "entry_time": timestamp}
        elif signal == -1 and position is not None:  # 賣出
            profit = (price - position["entry_price"]) / position[
                "entry_price"
            ]
            trades.append(profit)
            position = None

    if len(trades) > 0:
        total_return = sum(trades)
        win_rate = len([t for t in trades if t > 0]) / len(trades)
        avg_return = total_return / len(trades)

        print(f"交易次數: {len(trades)}")
        print(f"總收益: {total_return:.2%}")
        print(f"勝率: {win_rate:.1%}")
        print(f"平均收益: {avg_return:.2%}")
    else:
        print("沒有完整的交易對")
else:
    print("沒有足夠的信號進行回測")
