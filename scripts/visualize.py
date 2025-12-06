#!/usr/bin/env python3
import argparse
import warnings
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.providers.binance import BinanceDataProvider
from detection import (
    BTCMACDBuyDetection,
    MacdReversalDetection,
    MacdVolumeDetection,
    MLReversalDetector,
    PeakValleyDetection,
    TripleConfirmationReversalDetector,
)
from detection.macd_v1 import MACDV1Detector
from detection.ml_reversal_v3 import MLReversalV3Detector
from detection.ml_reversal_v4 import MLReversalV4Detector
from detection.ml_reversal_v5 import MLReversalV5Detector
from detection.ml_reversal_v6 import MLReversalV6Detector
from detection.ml_reversal_v7 import MLReversalV7Detector
from utils.indicators import analyze_cumulative_metrics, calculate_technical_indicators

warnings.filterwarnings("ignore")


def fetch_data(start_date, end_date):
    """Load data for specified time range"""
    try:
        df = pd.read_csv("training_data/btc_15m_2022_2025.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
        filtered_df = df[mask].copy().reset_index(drop=True)

        # 如果CSV沒有足夠數據，使用API抓取
        if len(filtered_df) == 0 or (
            len(filtered_df) > 0 and filtered_df["timestamp"].max() < end_date
        ):
            print("CSV數據不足，使用API抓取即時數據...")
            provider = BinanceDataProvider()

            # 轉換為毫秒時間戳
            start_time_ms = int(start_date.timestamp() * 1000)
            end_time_ms = int(end_date.timestamp() * 1000)

            api_df = provider.get_data(
                "BTCUSDT", "15m", start_time=start_time_ms, end_time=end_time_ms
            )

            if len(api_df) > 0:
                print(
                    f"從API載入 {len(api_df)} 筆記錄從 "
                    f"{api_df['timestamp'].min()} 到 "
                    f"{api_df['timestamp'].max()}"
                )
                return api_df

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
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def run_detection(df, detection_methods):
    """Run all detection methods and return results"""
    results = {}

    # Calculate all technical indicators once (includes ATR)
    df = calculate_technical_indicators(df)

    # Calculate cumulative metrics and second derivatives (for methods that need them)
    # 這樣所有檢測器都可以使用預先計算好的指標
    df_with_metrics = analyze_cumulative_metrics(df, window_size=16)

    for method in detection_methods:
        # Get signal array from vectorized detection
        # 傳入包含累計指標的完整數據
        signal_array = method.detect(df_with_metrics)

        # 檢查返回的是 boolean 還是 integer 類型
        if signal_array.dtype == bool:
            # 舊式 boolean 返回：轉換為索引
            signal_indices = np.where(signal_array)[0]
            signal_types = None  # 沒有方向信息
        else:
            # 新式 integer 返回：0=無信號, 1=做空, 2=做多
            signal_indices = np.where(signal_array > 0)[0]
            signal_types = signal_array[signal_indices]  # 保存信號類型

        results[method.name] = {
            "method": method,
            "signals": signal_indices,
            "signal_types": signal_types,  # 新增：信號類型 (1=做空, 2=做多)
            "signal_array": signal_array,
            "count": len(signal_indices),
        }

    return results


def _annotate_signals(ax, signal_times, signal_prices, color, n_signals):
    """
    智能標註信號價格

    Args:
        ax: matplotlib axis
        signal_times: 信號時間序列
        signal_prices: 信號價格序列
        color: 標註顏色
        n_signals: 信號數量
    """
    if n_signals <= 20:
        # 少於20個信號：全部標註
        for time, price in zip(signal_times, signal_prices):
            ax.annotate(
                f"${price:.0f}",
                xy=(time, price),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=color,
                    alpha=0.3,
                ),
            )
    elif n_signals <= 50:
        # 20-50個信號：每隔一個標註
        for i, (time, price) in enumerate(zip(signal_times, signal_prices)):
            if i % 2 == 0:
                ax.annotate(
                    f"${price:.0f}",
                    xy=(time, price),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=7,
                    alpha=0.6,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor=color,
                        alpha=0.3,
                    ),
                )
    else:
        # 超過50個信號：只標註極值點（最高和最低的各5個）
        prices_array = np.array(signal_prices)
        top_5_idx = np.argsort(prices_array)[-5:]
        bottom_5_idx = np.argsort(prices_array)[:5]
        important_idx = set(top_5_idx) | set(bottom_5_idx)

        for i, (time, price) in enumerate(zip(signal_times, signal_prices)):
            if i in important_idx:
                ax.annotate(
                    f"${price:.0f}",
                    xy=(time, price),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=7,
                    alpha=0.6,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor=color,
                        alpha=0.3,
                    ),
                )


def create_visualization(df, detection_results, output_file=None):
    """Create price chart with detection signals - 3 subplots style"""

    # 計算時間範圍
    time_range_days = (df["timestamp"].max() - df["timestamp"].min()).days

    # 根據時間範圍決定顯示方式
    if time_range_days <= 10:
        # 短時間範圍：使用原來的2圖顯示
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # Price chart
        ax1.plot(
            df["timestamp"], df["close"], "b-", linewidth=1, alpha=0.7, label="Price"
        )
        ax1.set_ylabel("Price (USD)", fontsize=12)
        ax1.set_title(
            "BTC Price with Detection Signals", fontsize=14, fontweight="bold"
        )
        ax1.grid(True, alpha=0.3)

        # Plot detection signals
        for method_name, result in detection_results.items():
            method = result["method"]
            signals = result["signals"]
            signal_types = result["signal_types"]

            if len(signals) > 0:
                signal_times = df.iloc[signals]["timestamp"]
                signal_prices = df.iloc[signals]["close"]

                # 如果有信號類型信息，分別繪製做多和做空
                if signal_types is not None:
                    # 做空信號 (1): 紅色向下三角形
                    short_mask = signal_types == 1
                    if short_mask.any():
                        ax1.scatter(
                            signal_times[short_mask],
                            signal_prices[short_mask],
                            color="red",
                            marker="v",  # 向下三角形
                            s=15,
                            label=f"{method_name} Short ({short_mask.sum()})",
                            alpha=0.8,
                            zorder=5,
                            edgecolors="darkred",
                            linewidths=1,
                        )

                    # 做多信號 (2): 綠色向上三角形
                    long_mask = signal_types == 2
                    if long_mask.any():
                        ax1.scatter(
                            signal_times[long_mask],
                            signal_prices[long_mask],
                            color="green",
                            marker="^",  # 向上三角形
                            s=15,
                            label=f"{method_name} Long ({long_mask.sum()})",
                            alpha=0.8,
                            zorder=5,
                            edgecolors="darkgreen",
                            linewidths=1,
                        )
                else:
                    # 舊式檢測器：使用原來的顏色和標記
                    ax1.scatter(
                        signal_times,
                        signal_prices,
                        color=method.get_color(),
                        marker=method.get_marker(),
                        s=10,
                        label=f"{method_name} ({len(signals)})",
                        alpha=0.8,
                        zorder=5,
                    )

        ax1.legend(loc="upper left")

        # Volume chart
        ax2.bar(df["timestamp"], df["volume"], alpha=0.6, color="gray", width=0.0003)
        ax2.set_ylabel("Volume", fontsize=12)
        ax2.set_xlabel("Time", fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    else:
        # 長時間範圍：使用3圖顯示（完整、3個月、1個月）
        fig, axes = plt.subplots(3, 1, figsize=(20, 12))

        # === 圖1: 完整時間範圍 ===
        ax1 = axes[0]
        ax1.plot(
            df["timestamp"], df["close"], "b-", linewidth=0.5, alpha=0.7, label="Price"
        )

        # 統計所有方法的信號總數
        total_signals = sum(
            len(result["signals"]) for result in detection_results.values()
        )

        # Plot all detection signals
        for method_name, result in detection_results.items():
            method = result["method"]
            signals = result["signals"]
            signal_types = result["signal_types"]

            if len(signals) > 0:
                signal_times = df.iloc[signals]["timestamp"]
                signal_prices = df.iloc[signals]["close"]

                # 如果有信號類型信息，分別繪製做多和做空
                if signal_types is not None:
                    # 做空信號 (1): 紅色向下三角形
                    short_mask = signal_types == 1
                    if short_mask.any():
                        ax1.scatter(
                            signal_times[short_mask],
                            signal_prices[short_mask],
                            color="red",
                            marker="v",
                            s=15,
                            label=f"{method_name} Short ({short_mask.sum()})",
                            alpha=0.8,
                            zorder=5,
                            edgecolors="darkred",
                            linewidths=0.5,
                        )

                    # 做多信號 (2): 綠色向上三角形
                    long_mask = signal_types == 2
                    if long_mask.any():
                        ax1.scatter(
                            signal_times[long_mask],
                            signal_prices[long_mask],
                            color="green",
                            marker="^",
                            s=15,
                            label=f"{method_name} Long ({long_mask.sum()})",
                            alpha=0.8,
                            zorder=5,
                            edgecolors="darkgreen",
                            linewidths=0.5,
                        )
                else:
                    # 舊式檢測器：使用原來的顏色和標記
                    ax1.scatter(
                        signal_times,
                        signal_prices,
                        color=method.get_color(),
                        marker=method.get_marker(),
                        s=10,
                        label=f"{method_name} ({len(signals)})",
                        alpha=0.8,
                        zorder=5,
                    )

        ax1.set_title(
            f"Bitcoin Price with Detection Signals (Full Range)\nTotal: {total_signals} signals",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xlabel("Date", fontsize=12)
        ax1.set_ylabel("Price (USDT)", fontsize=12)
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # === 圖2: 最近3個月 ===
        ax2 = axes[1]
        recent_3m = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=90)]

        ax2.plot(
            recent_3m["timestamp"],
            recent_3m["close"],
            "b-",
            linewidth=1,
            alpha=0.7,
            label="Price",
        )

        # Plot signals in recent 3 months
        signals_3m_count = 0
        for method_name, result in detection_results.items():
            method = result["method"]
            signals = result["signals"]
            signal_types = result["signal_types"]

            if len(signals) > 0:
                # Filter signals in recent 3 months
                time_mask = df.iloc[signals]["timestamp"] >= df[
                    "timestamp"
                ].max() - pd.Timedelta(days=90)
                recent_signals = signals[time_mask]
                signals_3m_count += len(recent_signals)

                if len(recent_signals) > 0:
                    signal_times = df.iloc[recent_signals]["timestamp"]
                    signal_prices = df.iloc[recent_signals]["close"]

                    # 如果有信號類型信息，分別繪製做多和做空
                    if signal_types is not None:
                        recent_signal_types = signal_types[time_mask]

                        # 做空信號 (1): 紅色向下三角形
                        short_mask = recent_signal_types == 1
                        if short_mask.any():
                            ax2.scatter(
                                signal_times[short_mask],
                                signal_prices[short_mask],
                                color="red",
                                marker="v",
                                s=15,
                                label=f"{method_name} Short ({short_mask.sum()})",
                                alpha=0.8,
                                zorder=5,
                                edgecolors="darkred",
                                linewidths=0.8,
                            )

                        # 做多信號 (2): 綠色向上三角形
                        long_mask = recent_signal_types == 2
                        if long_mask.any():
                            ax2.scatter(
                                signal_times[long_mask],
                                signal_prices[long_mask],
                                color="green",
                                marker="^",
                                s=15,
                                label=f"{method_name} Long ({long_mask.sum()})",
                                alpha=0.8,
                                zorder=5,
                                edgecolors="darkgreen",
                                linewidths=0.8,
                            )
                    else:
                        # 舊式檢測器：使用原來的顏色和標記
                        ax2.scatter(
                            signal_times,
                            signal_prices,
                            color=method.get_color(),
                            marker=method.get_marker(),
                            s=10,
                            label=f"{method_name} ({len(recent_signals)})",
                            alpha=0.8,
                            zorder=5,
                        )

        ax2.set_title(
            f"Last 3 Months Detail ({signals_3m_count} signals)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Price (USDT)", fontsize=12)
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        # === 圖3: 最近1個月 ===
        ax3 = axes[2]
        recent_1m = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=30)]

        ax3.plot(
            recent_1m["timestamp"],
            recent_1m["close"],
            "b-",
            linewidth=1.5,
            alpha=0.7,
            label="Price",
        )

        # Plot signals in recent 1 month with price annotations
        signals_1m_count = 0
        for method_name, result in detection_results.items():
            method = result["method"]
            signals = result["signals"]
            signal_types = result["signal_types"]

            if len(signals) > 0:
                # Filter signals in recent 1 month
                time_mask = df.iloc[signals]["timestamp"] >= df[
                    "timestamp"
                ].max() - pd.Timedelta(days=30)
                recent_signals = signals[time_mask]
                signals_1m_count += len(recent_signals)

                if len(recent_signals) > 0:
                    signal_times = df.iloc[recent_signals]["timestamp"]
                    signal_prices = df.iloc[recent_signals]["close"]

                    # 如果有信號類型信息，分別繪製做多和做空
                    if signal_types is not None:
                        recent_signal_types = signal_types[time_mask]

                        # 做空信號 (1): 紅色向下三角形
                        short_mask = recent_signal_types == 1
                        if short_mask.any():
                            short_times = signal_times[short_mask]
                            short_prices = signal_prices[short_mask]

                            ax3.scatter(
                                short_times,
                                short_prices,
                                color="red",
                                marker="v",
                                s=15,
                                label=f"{method_name} Short ({short_mask.sum()})",
                                alpha=0.8,
                                zorder=5,
                                edgecolors="darkred",
                                linewidths=1,
                            )

                            # 標註做空信號價格
                            _annotate_signals(
                                ax3, short_times, short_prices, "red", len(short_times)
                            )

                        # 做多信號 (2): 綠色向上三角形
                        long_mask = recent_signal_types == 2
                        if long_mask.any():
                            long_times = signal_times[long_mask]
                            long_prices = signal_prices[long_mask]

                            ax3.scatter(
                                long_times,
                                long_prices,
                                color="green",
                                marker="^",
                                s=15,
                                label=f"{method_name} Long ({long_mask.sum()})",
                                alpha=0.8,
                                zorder=5,
                                edgecolors="darkgreen",
                                linewidths=1,
                            )

                            # 標註做多信號價格
                            _annotate_signals(
                                ax3, long_times, long_prices, "green", len(long_times)
                            )
                    else:
                        # 舊式檢測器：使用原來的顏色和標記
                        ax3.scatter(
                            signal_times,
                            signal_prices,
                            color=method.get_color(),
                            marker=method.get_marker(),
                            s=10,
                            label=f"{method_name} ({len(recent_signals)})",
                            alpha=0.8,
                            zorder=5,
                        )

                        # 標註價格
                        _annotate_signals(
                            ax3,
                            signal_times,
                            signal_prices,
                            method.get_color(),
                            len(signal_times),
                        )

        # 根據信號數量添加標註說明
        if signals_1m_count <= 20:
            label_note = "all signals labeled"
        elif signals_1m_count <= 50:
            label_note = "every 2nd signal labeled"
        else:
            label_note = "top/bottom 10 signals labeled"

        ax3.set_title(
            f"Last 1 Month Detail ({signals_1m_count} signals, {label_note})",
            fontsize=14,
            fontweight="bold",
        )
        ax3.set_xlabel("Date", fontsize=12)
        ax3.set_ylabel("Price (USDT)", fontsize=12)
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {output_file}")

    # 不自動顯示圖表，只保存
    # plt.show()


def parse_datetime(date_str):
    """解析日期時間"""
    formats = ["%Y-%m-%d %H:%M", "%Y-%m-%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Invalid datetime format: {date_str}. " f"Use YYYY-MM-DD or YYYY-MM-DD HH:MM"
    )


def run_visualization(start_date_str, end_date_str, methods=None, output_file=None):
    """
    執行視覺化分析

    Args:
        start_date_str: 開始日期字串
        end_date_str: 結束日期字串
        methods: 檢測方法列表 (可選)
        output_file: 輸出圖片路徑 (可選，默認 crypto_chart.png)
    """
    if output_file is None:
        output_file = "crypto_chart.png"
    try:
        start_date = parse_datetime(start_date_str)
        end_date = parse_datetime(end_date_str)
    except ValueError as e:
        print(e)
        return

    print(f"Visualizing crypto data from {start_date_str} to {end_date_str}")
    print("=" * 60)

    # Load data
    df = fetch_data(start_date, end_date)
    if df is None:
        return

    # Initialize detection methods
    available_methods = {
        "macd": MacdReversalDetection(),
        "macdv1": MACDV1Detector(),
        "volume": MacdVolumeDetection(),
        "btc": BTCMACDBuyDetection(),
        "peak": PeakValleyDetection(distance=20, height_threshold=0.001),
        "triple": TripleConfirmationReversalDetector(extreme_window=5, window_size=16),
        "ml": MLReversalDetector(extreme_window=10, window_size=16),
        "mlv3": MLReversalV3Detector(prob_threshold=0.5),
        "mlv4": MLReversalV4Detector(),
        "mlv5": MLReversalV5Detector(),
        "mlv6": MLReversalV6Detector(),
        "mlv7": MLReversalV7Detector(),
    }

    # 使用默認方法如果沒有指定 - 默認使用全部方法
    if not methods:
        methods = list(available_methods.keys())
        print(f"使用全部檢測方法: {', '.join(methods)}")

    detection_methods = []
    for method_name in methods:
        if method_name in available_methods:
            detection_methods.append(available_methods[method_name])
        else:
            print(f"Unknown method: {method_name}")

    if not detection_methods:
        print("No valid detection methods specified")
        return

    # Run detection
    print("Running detection methods...")
    detection_results = run_detection(df, detection_methods)

    # Print results
    print("\nDetection Results:")
    print("-" * 40)
    for method_name, result in detection_results.items():
        print(f"{method_name}: {result['count']} signals")

    # Create visualization (使用原始 df，因為只需要價格數據)
    create_visualization(df, detection_results, output_file)
    print(f"\n✅ Chart saved to {output_file}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Crypto Visualization with Detection")
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
    parser.add_argument(
        "--output",
        default="crypto_chart.png",
        help="Output image file path (default: crypto_chart.png)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Detection methods: macd, volume, lstm, btc, peak, triple, ml, mlv3 (默認使用全部)",
    )
    args = parser.parse_args()

    run_visualization(args.start, args.end, args.methods, args.output)


if __name__ == "__main__":
    main()
