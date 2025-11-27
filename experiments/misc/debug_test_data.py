#!/usr/bin/env python3
"""檢查測試數據"""

# 做多情境1測試數據
print("=== 做多情境1測試數據檢查 ===")
prices = (
    [100] * 50
    + [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]  # 漸漲到118
    + [118] * 20
)

print("關鍵價格點:")
for i, price in enumerate(prices[50:70]):
    real_i = i + 50
    profit_pct = (price - 100) / 100 * 100
    print(f"  {real_i}: {price} ({profit_pct:.1f}%獲利)")

print(f"\n預期動態調整:")
print(f"  4%獲利時(104): 停損移至102, 停利變112")
print(f"  6%獲利時(106): 停損移至103, 停利變118")
print(f"  最終118應觸發停利")

print("\n=== 做空情境3測試數據檢查 ===")
prices = (
    [100] * 50
    + [100, 98, 96, 94, 92, 90, 88, 86, 84, 82]  # 漸跌到82
    + [82] * 20
)

print("關鍵價格點:")
for i, price in enumerate(prices[50:70]):
    real_i = i + 50
    profit_pct = (100 - price) / 100 * 100
    print(f"  {real_i}: {price} ({profit_pct:.1f}%獲利)")

print(f"\n預期動態調整:")
print(f"  4%獲利時(96): 停損移至98, 停利變88")
print(f"  6%獲利時(94): 停損移至97, 停利變82")
print(f"  最終82應觸發停利")
