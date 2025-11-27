#!/usr/bin/env python3
"""
測試精確布林帶模型
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def calculate_indicators(df):
    """計算技術指標"""
    # 布林帶
    df["bb_middle"] = df["close"].rolling(32).mean()
    bb_std = df["close"].rolling(32).std()
    df["bb_upper"] = df["bb_middle"] + (bb_std * 3.2)
    df["bb_lower"] = df["bb_middle"] - (bb_std * 3.2)

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    # 成交量移動平均
    df["volume_ma"] = df["volume"].rolling(20).mean()

    return df


def get_current_signal(df):
    """獲取當前市場信號"""
    latest = df.iloc[-1]

    # 檢查布林帶位置
    close_price = latest["close"]
    bb_upper = latest["bb_upper"]
    bb_lower = latest["bb_lower"]

    # 檢查MACD
    macd = latest["macd"]
    macd_signal = latest["macd_signal"]

    # 檢查成交量
    volume = latest["volume"]
    volume_ma = latest["volume_ma"]

    print(f"當前價格: ${close_price:,.2f}")
    print(f"布林帶上軌: ${bb_upper:,.2f}")
    print(f"布林帶下軌: ${bb_lower:,.2f}")
    print(f"MACD: {macd:.4f}")
    print(f"MACD信號線: {macd_signal:.4f}")
    print(f"成交量: {volume:,.0f}")
    print(f"成交量均線: {volume_ma:,.0f}")

    # 判斷信號條件
    at_upper = close_price >= bb_upper * 0.999  # 接近上軌
    at_lower = close_price <= bb_lower * 1.001  # 接近下軌
    macd_cross_down = macd < macd_signal  # MACD死叉
    macd_cross_up = macd > macd_signal  # MACD金叉
    high_volume = volume > volume_ma * 1.2  # 高成交量

    print(f"\n信號條件檢查:")
    print(f"接近上軌: {at_upper}")
    print(f"接近下軌: {at_lower}")
    print(f"MACD死叉: {macd_cross_down}")
    print(f"MACD金叉: {macd_cross_up}")
    print(f"高成交量: {high_volume}")

    # 檢查精確信號
    upper_signal = at_upper and macd_cross_down and high_volume
    lower_signal = at_lower and macd_cross_up and high_volume

    if upper_signal:
        return "上軌死叉", True
    elif lower_signal:
        return "下軌金叉", True
    else:
        return "無信號", False


def prepare_features(df, index):
    """準備模型特徵"""
    if index < 10:
        return None

    # 提取前10根K線的OHLCV
    features = []
    for i in range(index - 9, index + 1):
        features.extend(
            [
                df.iloc[i]["open"],
                df.iloc[i]["high"],
                df.iloc[i]["low"],
                df.iloc[i]["close"],
                df.iloc[i]["volume"],
            ]
        )

    return np.array(features).reshape(1, 10, 5)


def main():
    print("=== 精確布林帶模型測試 ===\n")

    # 載入數據
    try:
        data = pd.read_csv("data/btc_12months_15m.csv")
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        print(f"載入數據: {len(data)} 條記錄")
        print(
            f"時間範圍: {data['timestamp'].min()} 到 {data['timestamp'].max()}\n"
        )
    except FileNotFoundError:
        print("錯誤: 找不到數據文件 data/btc_12months_15m.csv")
        return

    # 計算指標
    data = calculate_indicators(data)

    # 載入模型
    try:
        model = load_model("bb_precise_model.keras")
        print("成功載入精確模型\n")
    except FileNotFoundError:
        print("錯誤: 找不到模型文件 bb_precise_model.keras")
        return

    # 獲取當前信號
    signal_type, has_signal = get_current_signal(data)

    print(f"\n=== 當前市場狀態 ===")
    print(f"信號類型: {signal_type}")
    print(f"符合精確條件: {'是' if has_signal else '否'}")

    if has_signal:
        # 準備特徵進行預測
        features = prepare_features(data, len(data) - 1)
        if features is not None:
            # 標準化特徵
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(
                features.reshape(-1, 50)
            ).reshape(1, 10, 5)

            # 預測
            prediction = model.predict(features_scaled, verbose=0)[0][0]

            print(f"\n=== 模型預測 ===")
            print(f"預測值: {prediction:.4f}")
            print(f"預測結果: {'上漲' if prediction > 0.5 else '下跌'}")
            print(f"信心度: {max(prediction, 1-prediction)*100:.1f}%")

            # 根據歷史表現給出建議
            if signal_type == "上軌死叉":
                print(f"\n=== 交易建議 ===")
                print("歷史統計: 上軌死叉信號96.4%概率預示下跌")
                if prediction < 0.5:
                    print("✅ 模型預測與歷史統計一致 - 建議考慮做空")
                else:
                    print("⚠️ 模型預測與歷史統計不一致 - 謹慎操作")
            elif signal_type == "下軌金叉":
                print(f"\n=== 交易建議 ===")
                print("歷史統計: 下軌金叉信號仍有96.4%概率預示下跌")
                if prediction < 0.5:
                    print("✅ 模型預測與歷史統計一致 - 建議謹慎做空")
                else:
                    print("⚠️ 模型預測上漲 - 可能是反轉信號，謹慎做多")
    else:
        print("\n當前市場不符合精確信號條件，建議等待")


if __name__ == "__main__":
    main()
