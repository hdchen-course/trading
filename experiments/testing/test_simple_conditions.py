import pandas as pd
import numpy as np
import talib

# 讀取數據
df = pd.read_csv("training_data/btc_12months_15m.csv")
print(f"數據長度: {len(df)}")

# 計算布林通道 (32, 2.4)
bb_upper, bb_middle, bb_lower = talib.BBANDS(
    df["close"], timeperiod=32, nbdevup=2.4, nbdevdn=2.4
)

# 計算成交量條件 (1.2倍)
volume_ma = df["volume"].rolling(20).mean()

# 條件檢測
bb_condition = (df["close"] <= bb_lower) | (df["close"] >= bb_upper)
volume_condition = df["volume"] >= (volume_ma * 1.2)

# 組合條件
combined_signal = bb_condition & volume_condition
signal_count = combined_signal.sum()

print(f"布林通道條件 (32, 2.4): {bb_condition.sum()} 個點")
print(f"成交量條件 (1.2倍): {volume_condition.sum()} 個點")
print(f"組合條件: {signal_count} 個點")
print(f"組合條件比例: {(signal_count/len(df)*100):.2f}%")
