import pandas as pd

# 读取数据
df = pd.read_csv('btc_15min_data.csv')
print("Available columns:")
print(df.columns.tolist())
print(f"\nTotal columns: {len(df.columns)}")
print(f"Data shape: {df.shape}")
