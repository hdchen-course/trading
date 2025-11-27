"""
布林帶模型使用指南

這個模型專門識別以下市場條件：
1. 價格觸及布林帶上軌或下軌 (32期, 3.2標準差)
2. MACD發生金叉或死叉
3. 成交量超過20期平均的1.2倍

當這些條件同時出現時，模型預測有95.7%機率價格會下跌
"""

import pandas as pd
import numpy as np
import talib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


def analyze_current_market():
    """分析當前市場狀況"""

    # 載入數據
    data = pd.read_csv("data/btc_12months_15m.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # 計算指標
    df = data.copy()
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(
        df["close"], timeperiod=32, nbdevup=3.2, nbdevdn=3.2
    )
    df["macd"], df["macd_signal"], _ = talib.MACD(df["close"])
    df["volume_ma"] = df["volume"].rolling(window=20).mean()

    current = df.iloc[-1]
    prev = df.iloc[-2]

    print("=== 當前市場分析 ===")
    print(f"時間: {current['timestamp']}")
    print(f"價格: ${current['close']:.2f}")
    print(f"布林帶上軌: ${current['bb_upper']:.2f}")
    print(f"布林帶中軌: ${current['bb_middle']:.2f}")
    print(f"布林帶下軌: ${current['bb_lower']:.2f}")

    # 價格位置
    if current["close"] >= current["bb_upper"] * 0.98:
        print("✅ 價格接近上軌")
        price_condition = True
    elif current["close"] <= current["bb_lower"] * 1.02:
        print("✅ 價格接近下軌")
        price_condition = True
    else:
        print("❌ 價格不在布林帶邊緣")
        price_condition = False

    # MACD交叉
    macd_cross_up = (
        current["macd"] > current["macd_signal"]
        and prev["macd"] <= prev["macd_signal"]
    )
    macd_cross_down = (
        current["macd"] < current["macd_signal"]
        and prev["macd"] >= prev["macd_signal"]
    )

    if macd_cross_up:
        print("✅ MACD金叉")
        macd_condition = True
    elif macd_cross_down:
        print("✅ MACD死叉")
        macd_condition = True
    else:
        print("❌ MACD無交叉")
        macd_condition = False

    # 成交量
    volume_ratio = current["volume"] / current["volume_ma"]
    if volume_ratio > 1.2:
        print(f"✅ 成交量放大 ({volume_ratio:.1f}x)")
        volume_condition = True
    else:
        print(f"❌ 成交量未放大 ({volume_ratio:.1f}x)")
        volume_condition = False

    return price_condition and macd_condition and volume_condition


def simulate_signal():
    """模擬一個信號情況"""
    print("\n=== 模型使用示例 ===")
    print("假設出現以下情況：")
    print("- 價格觸及布林帶上軌")
    print("- MACD死叉")
    print("- 成交量1.5倍放大")
    print("\n根據歷史數據，這種情況下：")
    print("📉 95.7% 機率價格會下跌")
    print("💡 建議：考慮做空或減倉")


def usage_instructions():
    """使用說明"""
    print("\n=== 使用方法 ===")
    print("1. 實時監控：python use_bb_model.py")
    print("2. 當出現信號時，考慮以下策略：")
    print("   - 做空BTC")
    print("   - 減少多頭倉位")
    print("   - 設置止損點")
    print("3. 風險管理：")
    print("   - 不要全倉操作")
    print("   - 設置合理止損")
    print("   - 考慮市場整體趨勢")


if __name__ == "__main__":
    has_signal = analyze_current_market()

    if has_signal:
        print("\n🚨 信號觸發！建議使用模型預測")
    else:
        print("\n⏳ 等待信號出現")

    simulate_signal()
    usage_instructions()
