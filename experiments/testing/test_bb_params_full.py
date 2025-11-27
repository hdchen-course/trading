import pandas as pd
import numpy as np
import talib


def detect_reversal_patterns(df):
    patterns = {}

    patterns["HAMMER"] = talib.CDLHAMMER(
        df["open"], df["high"], df["low"], df["close"]
    )
    patterns["SHOOTINGSTAR"] = talib.CDLSHOOTINGSTAR(
        df["open"], df["high"], df["low"], df["close"]
    )
    patterns["DOJI"] = talib.CDLDOJI(
        df["open"], df["high"], df["low"], df["close"]
    )
    patterns["ENGULFING"] = talib.CDLENGULFING(
        df["open"], df["high"], df["low"], df["close"]
    )
    patterns["HARAMI"] = talib.CDLHARAMI(
        df["open"], df["high"], df["low"], df["close"]
    )

    reversal_signal = np.zeros(len(df), dtype=bool)
    for pattern_name, pattern_values in patterns.items():
        pattern_condition = np.abs(pattern_values) >= 85
        reversal_signal = reversal_signal | pattern_condition

    return reversal_signal


def test_bb_parameters(df, param_combinations):
    results = []

    volume_ma = df["volume"].rolling(20).mean()
    volume_condition = df["volume"] >= (volume_ma * 1.2)
    reversal_condition = detect_reversal_patterns(df)

    for period, std_dev in param_combinations:
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            df["close"], timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
        )

        bb_condition = (df["close"] <= bb_lower) | (df["close"] >= bb_upper)
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
            f"期間: {period:2d}, 標準差: {std_dev:.1f}, 信號數: {signal_count:4d}, 比例: {(signal_count/len(df)*100):.2f}%"
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
    # 讀取一年份數據
    print("讀取一年份數據...")
    df = pd.read_csv("training_data/btc_12months_15m.csv")

    print(f"數據長度: {len(df)}")
    print("\n測試不同布林參數組合:")
    print("=" * 70)

    results = test_bb_parameters(df, param_combinations)

    results_sorted = sorted(
        results, key=lambda x: x["signal_count"], reverse=True
    )

    print("\n" + "=" * 70)
    print("排序結果 (按信號數量):")
    print("=" * 70)

    for i, result in enumerate(results_sorted):
        print(
            f"{i+1:2d}. 期間: {result['period']:2d}, 標準差: {result['std_dev']:.1f}, "
            f"信號數: {result['signal_count']:4d}, 比例: {result['percentage']:.2f}%"
        )
