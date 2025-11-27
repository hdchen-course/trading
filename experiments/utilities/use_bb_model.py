import pandas as pd
import numpy as np
import talib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")


def check_bb_signal(data, model_path="bb_pattern_model.keras"):
    """檢查當前是否有布林帶信號"""

    # 載入模型
    model = load_model(model_path)

    # 確保有足夠的數據
    if len(data) < 50:
        return None, "數據不足"

    # 計算指標
    df = data.copy()
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(
        df["close"], timeperiod=32, nbdevup=3.2, nbdevdn=3.2
    )
    df["macd"], df["macd_signal"], _ = talib.MACD(df["close"])
    df["volume_ma"] = df["volume"].rolling(window=20).mean()

    # 檢查最新數據點
    current = df.iloc[-1]
    prev = df.iloc[-2]

    # 檢查條件
    price_at_top = current["close"] >= current["bb_upper"] * 0.98
    price_at_bottom = current["close"] <= current["bb_lower"] * 1.02

    # MACD交叉
    macd_cross_up = (
        current["macd"] > current["macd_signal"]
        and prev["macd"] <= prev["macd_signal"]
    )
    macd_cross_down = (
        current["macd"] < current["macd_signal"]
        and prev["macd"] >= prev["macd_signal"]
    )

    # 成交量條件
    volume_spike = current["volume"] > current["volume_ma"] * 1.2

    # 檢查是否符合條件
    if (
        volume_spike
        and (macd_cross_up or macd_cross_down)
        and (price_at_top or price_at_bottom)
    ):

        # 準備模型輸入數據（最近10根K線）
        pattern_data = []
        for i in range(-10, 0):
            candle = df.iloc[i]
            pattern_data.extend(
                [
                    candle["open"],
                    candle["high"],
                    candle["low"],
                    candle["close"],
                    candle["volume"],
                ]
            )

        # 重塑和標準化
        X = np.array(pattern_data).reshape(1, 10, 5)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X.reshape(-1, 50)).reshape(1, 10, 5)

        # 預測
        prediction = model.predict(X_scaled, verbose=0)[0][0]

        # 解釋結果
        signal_type = "上軌" if price_at_top else "下軌"
        macd_type = "金叉" if macd_cross_up else "死叉"

        return {
            "signal": True,
            "prediction": prediction,
            "probability_down": 1 - prediction,  # 模型預測下跌的機率
            "signal_type": signal_type,
            "macd_type": macd_type,
            "current_price": current["close"],
            "bb_upper": current["bb_upper"],
            "bb_lower": current["bb_lower"],
            "volume_ratio": current["volume"] / current["volume_ma"],
        }

    return {"signal": False}, "未符合信號條件"


def get_trading_recommendation(signal_result):
    """根據信號給出交易建議"""
    if not signal_result.get("signal"):
        return "無信號"

    prob_down = signal_result["probability_down"]

    if prob_down > 0.8:
        return f"強烈建議做空/減倉 (下跌機率: {prob_down:.1%})"
    elif prob_down > 0.6:
        return f"建議做空/減倉 (下跌機率: {prob_down:.1%})"
    else:
        return f"謹慎觀望 (下跌機率: {prob_down:.1%})"


if __name__ == "__main__":
    # 載入最新數據
    data = pd.read_csv("data/btc_12months_15m.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # 檢查信號
    result, message = check_bb_signal(data)

    if result.get("signal"):
        print("🚨 布林帶信號觸發！")
        print(f"位置: {result['signal_type']}")
        print(f"MACD: {result['macd_type']}")
        print(f"當前價格: ${result['current_price']:.2f}")
        print(f"成交量倍數: {result['volume_ratio']:.1f}x")
        print(f"下跌機率: {result['probability_down']:.1%}")
        print(f"交易建議: {get_trading_recommendation(result)}")
    else:
        print("目前無信號")
        print(f"原因: {message}")
