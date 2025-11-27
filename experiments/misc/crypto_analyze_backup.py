#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import talib
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def fetch_data(start_date, end_date):
    """獲取指定時間範圍的數據"""
    try:
        # 嘗試從現有數據文件讀取
        df = pd.read_csv("training_data/btc_15m_2022_2025.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 篩選時間範圍
        mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
        filtered_df = df[mask].copy().reset_index(drop=True)

        if len(filtered_df) == 0:
            print(f"No data found for period {start_date} to {end_date}")
            return None

        print(
            f"Loaded {len(filtered_df)} records from "
            f"{filtered_df['timestamp'].min()} to "
            f"{filtered_df['timestamp'].max()}"
        )
        return filtered_df

    except FileNotFoundError:
        print(
            "Data file not found. Please ensure btc_15m_2022_2025.csv "
            "exists in training_data/"
        )
        return None


def calculate_amplitude(high, low, close):
    """計算振幅 (高低價差/收盤價)"""
    return (high - low) / close


def calculate_indicators(df):
    """計算技術指標"""
    # 基本指標
    df["amplitude"] = calculate_amplitude(df["high"], df["low"], df["close"])

    # MACD (12, 48, 9)
    exp1 = df["close"].ewm(span=12).mean()
    exp2 = df["close"].ewm(span=48).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # 成交量變化率
    df["volume_change"] = df["volume"].pct_change()

    # SAR (使用talib)
    df["sar"] = talib.SAR(df["high"].values, df["low"].values)
    df["price_above_sar"] = df["close"] > df["sar"]

    # RSI
    df["rsi"] = talib.RSI(df["close"].values, timeperiod=14)

    # ATR
    df["atr"] = talib.ATR(
        df["high"].values, df["low"].values, df["close"].values, timeperiod=14
    )

    # 布林帶
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(df["close"].values)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )

    return df


# 配置參數
ROLLING_WINDOW_HOURS = 4  # 滾動窗口小時數
ROLLING_WINDOW_SIZE = ROLLING_WINDOW_HOURS * 4  # 轉換為15分鐘週期數


def analyze_cumulative_metrics(df):
    """分析滾動窗口指標變化 - 完全向量化版本"""
    window_size = ROLLING_WINDOW_SIZE

    if len(df) < window_size:
        return pd.DataFrame()

    # 向量化計算滾動指標
    results = {
        "period": range(window_size, len(df) + 1),
        "timestamp": df["timestamp"].iloc[window_size - 1 :].values,
        "close_price": df["close"].iloc[window_size - 1 :].values,
        "avg_amplitude": df["amplitude"]
        .rolling(window=window_size)
        .mean()
        .iloc[window_size - 1 :]
        .values,
        "avg_volume": df["volume"]
        .rolling(window=window_size)
        .mean()
        .iloc[window_size - 1 :]
        .values,
        "avg_macd": df["macd"]
        .rolling(window=window_size)
        .mean()
        .iloc[window_size - 1 :]
        .values,
        "avg_rsi": df["rsi"]
        .rolling(window=window_size)
        .mean()
        .iloc[window_size - 1 :]
        .values,
        "price_above_sar_ratio": df["price_above_sar"]
        .rolling(window=window_size)
        .mean()
        .iloc[window_size - 1 :]
        .values,
        "avg_bb_position": df["bb_position"]
        .rolling(window=window_size)
        .mean()
        .iloc[window_size - 1 :]
        .values,
        "volatility": df["close"]
        .pct_change()
        .rolling(window=window_size)
        .std()
        .iloc[window_size - 1 :]
        .values,
    }

    # 向量化計算價格變化百分比
    close_start = df["close"].iloc[: -window_size + 1].values  # 窗口開始價格
    close_end = df["close"].iloc[window_size - 1 :].values  # 窗口結束價格
    results["price_change_pct"] = (close_end - close_start) / close_start * 100

    # 計算1小時前價格變化相對ATR比值
    price_1h_ago = df["close"].shift(4)  # 1小時前 = 4個15分鐘週期
    price_diff_1h = df["close"] - price_1h_ago
    atr_ratio_1h = price_diff_1h / df["atr"]
    results["price_atr_ratio_1h"] = atr_ratio_1h.iloc[window_size - 1 :].values

    # 向量化計算成交量趨勢 (窗口內前半段 vs 後半段)
    half_window = window_size // 2
    vol_recent = (
        df["volume"].rolling(window=half_window).mean().iloc[window_size - 1 :].values
    )  # 窗口後半段平均
    vol_prev = (
        df["volume"]
        .rolling(window=half_window)
        .mean()
        .shift(half_window)
        .iloc[window_size - 1 :]
        .values
    )  # 窗口前半段平均
    results["volume_trend"] = np.where(vol_prev > 0, vol_recent / vol_prev, 1.0)

    return pd.DataFrame(results)


def format_output(analysis_df, sample_points=None):
    """格式化輸出結果"""
    # 如果指定樣本點數且數據量大於樣本點數，則取樣
    if sample_points and len(analysis_df) > sample_points:
        step = len(analysis_df) // sample_points
        sample_df = analysis_df.iloc[::step].copy()
    else:
        sample_df = analysis_df.copy()

    # 格式化數值
    sample_df["close_price"] = sample_df["close_price"].apply(lambda x: f"{x:.2f}")
    sample_df["avg_amplitude"] = sample_df["avg_amplitude"].apply(lambda x: f"{x:.4f}")
    sample_df["avg_volume"] = sample_df["avg_volume"].apply(lambda x: f"{x:.0f}")
    sample_df["avg_macd"] = sample_df["avg_macd"].apply(lambda x: f"{x:.2f}")
    sample_df["avg_rsi"] = sample_df["avg_rsi"].apply(lambda x: f"{x:.1f}")
    sample_df["price_above_sar_ratio"] = sample_df["price_above_sar_ratio"].apply(
        lambda x: f"{x:.2f}"
    )
    sample_df["avg_bb_position"] = sample_df["avg_bb_position"].apply(
        lambda x: f"{x:.3f}"
    )
    sample_df["price_change_pct"] = sample_df["price_change_pct"].apply(
        lambda x: f"{x:+.2f}%"
    )
    sample_df["price_atr_ratio_1h"] = sample_df["price_atr_ratio_1h"].apply(
        lambda x: f"{x:+.2f}"
    )
    sample_df["volume_trend"] = sample_df["volume_trend"].apply(lambda x: f"{x:.2f}")
    sample_df["volatility"] = sample_df["volatility"].apply(lambda x: f"{x:.4f}")

    return sample_df


def main():
    parser = argparse.ArgumentParser(description="Crypto Technical Analysis")
    parser.add_argument(
        "--start",
        required=True,
        help="Start datetime (YYYY-MM-DD or YYYY-MM-DD HH:MM)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End datetime (YYYY-MM-DD or YYYY-MM-DD HH:MM)",
    )
    parser.add_argument("--output", help="Output CSV file path")
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of sample points to display",
    )

    args = parser.parse_args()

    # 解析日期時間 - 支援多種格式
    def parse_datetime(date_str):
        formats = ["%Y-%m-%d %H:%M", "%Y-%m-%d"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(
            f"Invalid datetime format: {date_str}. "
            f"Use YYYY-MM-DD or YYYY-MM-DD HH:MM"
        )

    try:
        start_date = parse_datetime(args.start)
        end_date = parse_datetime(args.end)
    except ValueError as e:
        print(e)
        return

    print(f"Analyzing crypto data from {args.start} to {args.end}")
    print("=" * 60)

    # 獲取數據
    df = fetch_data(start_date, end_date)
    if df is None:
        return

    # 計算技術指標
    print("Calculating technical indicators...")
    df = calculate_indicators(df)

    # 分析累計指標
    print("Analyzing cumulative metrics...")
    analysis_df = analyze_cumulative_metrics(df)

    # 格式化輸出
    display_df = format_output(
        analysis_df, args.samples if args.samples != 20 else None
    )

    # 顯示結果
    print(
        f"\nCumulative Technical Analysis " f"(showing {len(display_df)} sample points)"
    )
    print("=" * 140)

    # 設置pandas顯示選項
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 15)

    print(
        display_df[
            [
                "period",
                "timestamp",
                "close_price",
                "avg_amplitude",
                "avg_macd",
                "avg_rsi",
                "price_above_sar_ratio",
                "avg_bb_position",
                "price_change_pct",
                "price_atr_ratio_1h",
                "volume_trend",
                "volatility",
            ]
        ].to_string(index=False)
    )

    # 保存到文件
    if args.output:
        analysis_df.to_csv(args.output, index=False)
        print(f"\nFull analysis saved to {args.output}")

    # 總結統計
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total periods analyzed: {len(analysis_df)}")
    print(f"Final price change: {analysis_df['price_change_pct'].iloc[-1]:+.2f}%")
    print(f"Average amplitude: {analysis_df['avg_amplitude'].mean():.4f}")
    print(f"Average RSI: {analysis_df['avg_rsi'].mean():.1f}")
    print(f"Time above SAR: {analysis_df['price_above_sar_ratio'].mean():.1%}")
    print(f"Average BB position: {analysis_df['avg_bb_position'].mean():.3f}")


if __name__ == "__main__":
    main()
