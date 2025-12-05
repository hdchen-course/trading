#!/usr/bin/env python3
"""
Crypto Trading System Utilities

統一的技術指標計算函數，確保所有模組使用相同的計算邏輯。
"""

import pandas as pd
import numpy as np
import talib
import warnings

warnings.filterwarnings("ignore")


def calculate_amplitude(high, low, close):
    """計算振幅"""
    return (high - low) / close


def calculate_volume_trend(df, window_size):
    """計算成交量趨勢 (窗口內前半段 vs 後半段)"""
    half_window = window_size // 2
    vol_recent = (
        df["volume"].rolling(window=half_window).mean().iloc[window_size - 1 :].values
    )
    vol_prev = (
        df["volume"]
        .rolling(window=half_window)
        .mean()
        .shift(half_window)
        .iloc[window_size - 1 :]
        .values
    )

    volume_trend = np.ones(len(df))
    volume_trend[window_size - 1 :] = np.where(vol_prev > 0, vol_recent / vol_prev, 1.0)
    return volume_trend


def calculate_price_change_pct(df, window_size):
    """計算窗口內價格變化百分比"""
    close_start = df["close"].iloc[: -window_size + 1].values
    close_end = df["close"].iloc[window_size - 1 :].values
    price_change_window = (close_end - close_start) / close_start * 100

    price_change_pct = np.zeros(len(df))
    price_change_pct[window_size - 1 :] = price_change_window
    return price_change_pct


def calculate_price_atr_ratio_1h(df):
    """計算1小時前價格變化相對ATR比值"""
    atr = talib.ATR(
        df["high"].values, df["low"].values, df["close"].values, timeperiod=14
    )
    price_1h_ago = df["close"].shift(4)
    price_diff_1h = df["close"] - price_1h_ago
    return price_diff_1h / atr


def calculate_technical_indicators(df, fixed_window=1000):
    """
    計算所有基本技術指標

    使用固定窗口大小來計算 EMA 相關指標，確保不同數據長度下結果一致。

    優化策略：
    - 如果數據長度 > fixed_window，後面的點已經穩定，直接用全部數據計算
    - 只有前 fixed_window 個點需要用固定窗口逐一計算

    Args:
        df: 包含 OHLCV 數據的 DataFrame
        fixed_window: 固定窗口大小，默認 1000 期
                     對於 EMA(100)，1000 期後初始值影響 < 0.001%

    Returns:
        添加了技術指標的 DataFrame
    """
    df = df.copy()

    # 基本指標
    df["amplitude"] = calculate_amplitude(df["high"], df["low"], df["close"])

    # 使用固定窗口計算 MACD 和 EMA，確保一致性
    # 每個時間點都用最近 fixed_window 個點計算
    n = len(df)

    # 初始化結果數組
    macd_values = np.full(n, np.nan)
    macd_signal_values = np.full(n, np.nan)
    macd_hist_values = np.full(n, np.nan)
    ema_50_values = np.full(n, np.nan)
    ema_100_values = np.full(n, np.nan)

    # 初始化其他指標的結果數組
    rsi_values = np.full(n, np.nan)
    stochrsi_k_values = np.full(n, np.nan)
    stochrsi_d_values = np.full(n, np.nan)
    atr_values = np.full(n, np.nan)
    sar_values = np.full(n, np.nan)
    bb_upper_values = np.full(n, np.nan)
    bb_middle_values = np.full(n, np.nan)
    bb_lower_values = np.full(n, np.nan)

    # 對每個時間點，用固定窗口計算
    for i in range(n):
        # 取最近 fixed_window 個點（或所有可用數據）
        start = max(0, i - fixed_window + 1)
        window_data = df.iloc[start : i + 1]

        # 至少需要 100 個點才能計算 EMA(100)
        if len(window_data) >= 100:
            # MACD
            macd, signal, hist = talib.MACD(
                window_data["close"].values,
                fastperiod=12,
                slowperiod=48,
                signalperiod=9,
            )
            macd_values[i] = macd[-1]
            macd_signal_values[i] = signal[-1]
            macd_hist_values[i] = hist[-1]

            # EMA
            ema_50_values[i] = talib.EMA(window_data["close"].values, timeperiod=50)[-1]
            ema_100_values[i] = talib.EMA(window_data["close"].values, timeperiod=100)[
                -1
            ]

            # RSI
            rsi_values[i] = talib.RSI(window_data["close"].values, timeperiod=14)[-1]

            # StochRSI
            stochrsi_k, stochrsi_d = talib.STOCHRSI(
                window_data["close"].values,
                timeperiod=14,
                fastk_period=3,
                fastd_period=3,
            )
            stochrsi_k_values[i] = stochrsi_k[-1]
            stochrsi_d_values[i] = stochrsi_d[-1]

            # ATR
            atr_values[i] = talib.ATR(
                window_data["high"].values,
                window_data["low"].values,
                window_data["close"].values,
                timeperiod=14,
            )[-1]

            # SAR
            sar_values[i] = talib.SAR(
                window_data["high"].values, window_data["low"].values
            )[-1]

            # 布林帶 (period=32, std=2.4)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                window_data["close"].values, timeperiod=32, nbdevup=2.4, nbdevdn=2.4
            )
            bb_upper_values[i] = bb_upper[-1]
            bb_middle_values[i] = bb_middle[-1]
            bb_lower_values[i] = bb_lower[-1]

    df["macd"] = macd_values
    df["macd_signal"] = macd_signal_values
    df["macd_histogram"] = macd_hist_values
    df["ema_50"] = ema_50_values
    df["ema_100"] = ema_100_values

    # Close price 相對於 EMA 的漲跌幅
    df["close_ema50_pct"] = (df["close"] - df["ema_50"]) / df["ema_50"] * 100
    df["close_ema100_pct"] = (df["close"] - df["ema_100"]) / df["ema_100"] * 100
    df["rsi"] = rsi_values
    df["stochrsi_k"] = stochrsi_k_values
    df["stochrsi_d"] = stochrsi_d_values
    df["atr"] = atr_values
    df["sar"] = sar_values
    df["bb_upper"] = bb_upper_values
    df["bb_middle"] = bb_middle_values
    df["bb_lower"] = bb_lower_values

    # 布林通道衍生指標
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )  # %B: 價格在通道中的相對位置 (0-1)
    df["bb_width_pct"] = (
        (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"] * 100
    )  # 布林寬百分比：波動性指標

    # 計算價格到 SAR 的距離百分比
    # 正值表示價格在 SAR 上方，負值表示價格在 SAR 下方
    df["sar_distance_pct"] = (df["close"] - df["sar"]) / df["close"] * 100

    # 成交量變化率
    df["volume_change"] = df["volume"].pct_change()

    # 成交量指標
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    # 價格動量特徵
    df["price_change_3"] = df["close"].pct_change(3)
    df["price_change_5"] = df["close"].pct_change(5)

    return df


def calculate_second_derivative(series):
    """計算序列的二階導數 (加速度)"""
    first_derivative = series.diff()
    second_derivative = first_derivative.diff()
    return second_derivative


def calculate_z_score(series, window=200):
    """計算z-score標準化"""
    rolling_mean = series.rolling(window=window, min_periods=20).mean()
    rolling_std = series.rolling(window=window, min_periods=20).std()
    z_score = (series - rolling_mean) / rolling_std
    return z_score.fillna(0)


def analyze_cumulative_metrics(df, window_size=16):
    """
    分析累計技術指標

    改進版：保留所有數據，前面無法計算的部分填 NaN
    這樣可以保持與原始 df 相同的長度和索引對齊
    """
    n = len(df)
    if n < window_size:
        raise ValueError(f"Data length {n} is less than window size {window_size}")

    # 使用 rolling 計算，自動處理前面的 NaN
    results = {
        "period": range(1, n + 1),
        "timestamp": df["timestamp"].values,
        "close_price": df["close"].values,
        "ema_100": df["ema_100"].values,
        "ema_50": df["ema_50"].values,
        "avg_amplitude": df["amplitude"]
        .rolling(window=window_size, min_periods=1)
        .mean()
        .values,
        "avg_volume": df["volume"]
        .rolling(window=window_size, min_periods=1)
        .mean()
        .values,
        "avg_macd": df["macd"].rolling(window=window_size, min_periods=1).mean().values,
        "avg_rsi": df["rsi"].rolling(window=window_size, min_periods=1).mean().values,
        "avg_bb_position": df["bb_position"]
        .rolling(window=window_size, min_periods=1)
        .mean()
        .values,
        "volatility": df["close"]
        .pct_change()
        .rolling(window=window_size, min_periods=1)
        .std()
        .values,
    }

    # 計算窗口內價格變化百分比（保留所有數據）
    results["price_change_pct"] = calculate_price_change_pct(df, window_size)

    # 計算1小時前價格變化相對ATR比值（保留所有數據）
    price_atr_ratio_1h = calculate_price_atr_ratio_1h(df)
    results["price_atr_ratio_1h"] = price_atr_ratio_1h.values

    # 計算成交量趨勢（保留所有數據）
    volume_trend = calculate_volume_trend(df, window_size)
    results["volume_trend"] = volume_trend

    # 計算 LIS/LDS（使用相同的 window_size）
    lis_values = [np.nan] * window_size
    lds_values = [np.nan] * window_size

    for idx in range(window_size, n):
        start_idx = idx - window_size
        end_idx = idx
        lis_length = calculate_longest_increasing_subsequence(
            df, "close", start_idx, end_idx
        )
        lds_length = calculate_longest_decreasing_subsequence(
            df, "close", start_idx, end_idx
        )
        lis_values.append(lis_length)
        lds_values.append(lds_length)

    lis_series = pd.Series(lis_values)
    lds_series = pd.Series(lds_values)

    # 計算二階導數 (加速度) 並標準化為z-score
    macd_series = pd.Series(results["avg_macd"])
    price_series = pd.Series(results["close_price"])
    volume_trend_series = pd.Series(results["volume_trend"])
    atr_series = pd.Series(
        df["atr"].rolling(window=window_size, min_periods=1).mean().values
    )
    sar_distance_series = pd.Series(df["sar_distance_pct"].values)

    # 計算原始二階導數
    macd_2nd_raw = calculate_second_derivative(macd_series)
    price_2nd_raw = calculate_second_derivative(price_series)
    vol_2nd_raw = calculate_second_derivative(volume_trend_series)
    atr_2nd_raw = calculate_second_derivative(atr_series)
    lis_2nd_raw = calculate_second_derivative(lis_series)
    lds_2nd_raw = calculate_second_derivative(lds_series)
    sar_distance_2nd_raw = calculate_second_derivative(sar_distance_series)

    # 儲存原始 LIS/LDS 值
    results["lis"] = lis_series.values
    results["lds"] = lds_series.values

    # 標準化為z-score (200期滾動)
    results["macd_2nd"] = calculate_z_score(macd_2nd_raw, window=200).values
    results["price_2nd"] = calculate_z_score(price_2nd_raw, window=200).values
    results["vol_2nd"] = calculate_z_score(vol_2nd_raw, window=200).values
    results["atr_2nd"] = calculate_z_score(atr_2nd_raw, window=200).values
    results["lis_2nd"] = calculate_z_score(lis_2nd_raw, window=200).values
    results["lds_2nd"] = calculate_z_score(lds_2nd_raw, window=200).values
    results["sar_distance_2nd"] = calculate_z_score(
        sar_distance_2nd_raw, window=200
    ).values

    # 把原始df中所有欄位複製到results中（除了timestamp和已存在的欄位）
    # 改進：不再 slice，保持完整長度
    for col in df.columns:
        if col != "timestamp" and col not in results:
            results[col] = df[col].values

    return pd.DataFrame(results)


def calculate_longest_increasing_subsequence(
    df: pd.DataFrame, price_col: str = "close", start_idx: int = 0, end_idx: int = None
) -> int:
    """
    計算指定區間內的最長遞增子序列長度（使用動態規劃 + 二分搜索，O(n log n)）

    LIS 是指可以跳過元素的最長遞增子序列，不需要連續。

    算法：
    - 使用 tails 數組維護不同長度的 LIS 的最小尾部值
    - 使用二分搜索找到當前元素應該放置的位置
    - 時間複雜度：O(n log n)

    Args:
        df: 包含價格數據的 DataFrame
        price_col: 價格列名，默認為 'close'
        start_idx: 起始索引（包含），默認為 0
        end_idx: 結束索引（不包含），默認為 None（表示到末尾）

    Returns:
        int: 該區間內的最長遞增子序列長度

    Example:
        價格序列: [10, 9, 2, 5, 3, 7, 101, 18]
        LIS 長度: 4
        LIS 序列: [2, 5, 7, 101] 或 [2, 3, 7, 101]
    """
    prices = df[price_col].values

    if end_idx is None:
        end_idx = len(prices)

    # 只處理指定範圍
    prices_range = prices[start_idx:end_idx]
    n = len(prices_range)

    if n == 0:
        return 0

    # tails[i] 表示長度為 i+1 的 LIS 的最小尾部值
    tails = []

    for price in prices_range:
        # 二分搜索找到 price 應該放置的位置
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < price:
                left = mid + 1
            else:
                right = mid

        # 更新 tails
        if left == len(tails):
            tails.append(price)
        else:
            tails[left] = price

    return len(tails)


def calculate_longest_decreasing_subsequence(
    df: pd.DataFrame, price_col: str = "close", start_idx: int = 0, end_idx: int = None
) -> int:
    """
    計算指定區間內的最長遞減子序列長度（使用動態規劃 + 二分搜索，O(n log n)）

    LDS 是指可以跳過元素的最長遞減子序列，不需要連續。

    算法：
    - 使用 tails 數組維護不同長度的 LDS 的最大尾部值
    - 使用二分搜索找到當前元素應該放置的位置
    - 時間複雜度：O(n log n)

    Args:
        df: 包含價格數據的 DataFrame
        price_col: 價格列名，默認為 'close'
        start_idx: 起始索引（包含），默認為 0
        end_idx: 結束索引（不包含），默認為 None（表示到末尾）

    Returns:
        int: 該區間內的最長遞減子序列長度

    Example:
        價格序列: [10, 9, 2, 5, 3, 7, 101, 18]
        LDS 長度: 3
        LDS 序列: [10, 9, 2] 或 [10, 5, 3]
    """
    prices = df[price_col].values

    if end_idx is None:
        end_idx = len(prices)

    # 只處理指定範圍
    prices_range = prices[start_idx:end_idx]
    n = len(prices_range)

    if n == 0:
        return 0

    # 將 LDS 問題轉換為 LIS 問題：對所有價格取負數
    # 這樣遞減序列就變成了遞增序列
    neg_prices = [-p for p in prices_range]

    # tails[i] 表示長度為 i+1 的 LIS 的最小尾部值（對負數）
    tails = []

    for price in neg_prices:
        # 二分搜索找到 price 應該放置的位置
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < price:
                left = mid + 1
            else:
                right = mid

        # 更新 tails
        if left == len(tails):
            tails.append(price)
        else:
            tails[left] = price

    return len(tails)


def add_subsequence_indicators(
    df: pd.DataFrame, price_col: str = "close", window: int = 20
) -> pd.DataFrame:
    """
    為 DataFrame 添加滾動窗口的最長遞增/遞減子序列指標

    對每個位置，計算過去 window 個數據點的 LIS/LDS 長度。

    這些指標可以用來：
    1. 識別趨勢強度：LIS/LDS 長度越大，趨勢越強
    2. 檢測趨勢轉折：LIS 從大變小可能是上漲趨勢結束
    3. 確認趨勢延續：持續的高 LIS/LDS 值表示趨勢穩定
    4. 比較相對強度：LIS > LDS 表示上漲趨勢主導

    Args:
        df: 包含價格數據的 DataFrame
        price_col: 價格列名，默認為 'close'
        window: 滾動窗口大小，默認為 20

    Returns:
        添加了 'lis_length'、'lds_length' 和 'trend_indicator' 列的 DataFrame

    Example:
        對於窗口大小 = 5:
        價格: [10, 9, 2, 5, 3, 7, 101, 18]
        位置4的LIS: 計算 [10, 9, 2, 5, 3] 的 LIS = 2 ([2, 3])
        位置5的LIS: 計算 [9, 2, 5, 3, 7] 的 LIS = 3 ([2, 3, 7])
    """
    df = df.copy()
    n = len(df)

    lis_lengths = np.zeros(n)
    lds_lengths = np.zeros(n)

    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1

        lis_lengths[i] = calculate_longest_increasing_subsequence(
            df, price_col, start, end
        )
        lds_lengths[i] = calculate_longest_decreasing_subsequence(
            df, price_col, start, end
        )

    df["lis_length"] = lis_lengths
    df["lds_length"] = lds_lengths

    # 計算趨勢指標：LIS - LDS
    # 正值表示上漲趨勢主導，負值表示下跌趨勢主導
    df["trend_indicator"] = df["lis_length"] - df["lds_length"]

    return df
