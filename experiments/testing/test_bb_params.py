import pandas as pd
import numpy as np
import talib
from binance.client import Client
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


def get_binance_data(symbol="BTCUSDT", interval="1h", days=30):
    client = Client()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    klines = client.get_historical_klines(
        symbol,
        interval,
        start_time.strftime("%Y-%m-%d"),
        end_time.strftime("%Y-%m-%d"),
    )

    df = pd.DataFrame(
        klines,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    return df[["open", "high", "low", "close", "volume"]].copy()


def detect_reversal_patterns(df):
    patterns = {}

    # HAMMER
    patterns["HAMMER"] = talib.CDLHAMMER(
        df["open"], df["high"], df["low"], df["close"]
    )

    # SHOOTING STAR
    patterns["SHOOTINGSTAR"] = talib.CDLSHOOTINGSTAR(
        df["open"], df["high"], df["low"], df["close"]
    )

    # DOJI
    patterns["DOJI"] = talib.CDLDOJI(
        df["open"], df["high"], df["low"], df["close"]
    )

    # ENGULFING
    patterns["ENGULFING"] = talib.CDLENGULFING(
        df["open"], df["high"], df["low"], df["close"]
    )

    # HARAMI
    patterns["HARAMI"] = talib.CDLHARAMI(
        df["open"], df["high"], df["low"], df["close"]
    )

    # 合併所有型態，強度>=85
    reversal_signal = np.zeros(len(df), dtype=bool)
    for pattern_name, pattern_values in patterns.items():
        pattern_condition = np.abs(pattern_values) >= 85
        reversal_signal = reversal_signal | pattern_condition

    return reversal_signal


def test_bb_parameters(df, param_combinations):
    results = []

    # 計算成交量條件 (1.2倍)
    volume_ma = df["volume"].rolling(20).mean()
    volume_condition = df["volume"] >= (volume_ma * 1.2)

    # 反轉型態條件
    reversal_condition = detect_reversal_patterns(df)

    for period, std_dev in param_combinations:
        # 計算布林帶
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            df["close"], timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
        )

        # 布林帶邊緣條件
        bb_condition = (df["close"] <= bb_lower) | (df["close"] >= bb_upper)

        # 組合所有條件
        combined_signal = bb_condition & volume_condition & reversal_condition
        signal_count = combined_signal.sum()

        results.append(
            {
                "period": period,
                "std_dev": std_dev,
                "signal_count": signal_count,
                "percentage": (signal_count / len(df)) * 100,
            }
        )

        print(
            f"期間: {period}, 標準差: {std_dev}, 信號數: {signal_count}, 比例: {(signal_count/len(df)*100):.2f}%"
        )

    return results


# 測試參數組合
param_combinations = [
    (20, 1.5),
    (20, 2.0),
    (20, 2.5),
    (30, 1.5),
    (30, 2.0),
    (30, 2.5),
    (40, 1.5),
    (40, 2.0),
    (40, 2.5),
    (50, 1.5),
    (50, 2.0),
    (50, 2.5),
    (54, 2.4),  # 原始參數
    (60, 1.5),
    (60, 2.0),
    (60, 2.5),
]

if __name__ == "__main__":
    print("獲取BTCUSDT數據...")
    df = get_binance_data("BTCUSDT", "1h", 60)  # 60天數據

    print(f"數據長度: {len(df)}")
    print("\n測試不同布林參數組合:")
    print("=" * 60)

    results = test_bb_parameters(df, param_combinations)

    # 排序結果
    results_sorted = sorted(
        results, key=lambda x: x["signal_count"], reverse=True
    )

    print("\n" + "=" * 60)
    print("排序結果 (按信號數量):")
    print("=" * 60)

    for i, result in enumerate(results_sorted[:10]):
        print(
            f"{i+1:2d}. 期間: {result['period']:2d}, 標準差: {result['std_dev']:.1f}, "
            f"信號數: {result['signal_count']:3d}, 比例: {result['percentage']:.2f}%"
        )
