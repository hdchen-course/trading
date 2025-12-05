#!/usr/bin/env python3
"""
生成大規模訓練數據

1. 從 Binance 抓取 2022-2025/09/01 的歷史數據
2. 使用 peak_valley.py 檢測反轉點
3. 使用 indicators.py 計算所有指標
4. 保存為訓練數據集
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402
from datetime import datetime  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
from data.providers.binance import BinanceDataProvider  # noqa: E402
from detection.peak_valley import PeakValleyDetection  # noqa: E402
from utils.indicators import (  # noqa: E402
    calculate_technical_indicators,
    analyze_cumulative_metrics,
)


def fetch_historical_data(start_date, end_date, interval="15m"):
    """
    抓取歷史數據

    Args:
        start_date: 開始日期 (YYYY-MM-DD)
        end_date: 結束日期 (YYYY-MM-DD)
        interval: K線間隔 (15m, 1h, 4h, 1d)
    """
    print("=" * 80)
    print("📥 抓取歷史數據")
    print("=" * 80)

    print(f"\n時間範圍: {start_date} 到 {end_date}")
    print(f"K線間隔: {interval}")

    provider = BinanceDataProvider()

    # 轉換日期格式
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    print(f"\n開始抓取數據...")

    all_data = []
    current_start = start_ts

    # Binance API 限制：每次最多1000根K線
    # 15分鐘 * 1000 = 10.4天
    chunk_size = 1000

    if interval == "15m":
        chunk_duration = chunk_size * 15 * 60 * 1000  # 毫秒
    elif interval == "1h":
        chunk_duration = chunk_size * 60 * 60 * 1000
    elif interval == "4h":
        chunk_duration = chunk_size * 4 * 60 * 60 * 1000
    else:
        chunk_duration = chunk_size * 24 * 60 * 60 * 1000

    chunk_count = 0

    while current_start < end_ts:
        chunk_end = min(current_start + chunk_duration, end_ts)

        try:
            # 抓取數據
            df_chunk = provider.get_data(
                symbol="BTCUSDT",
                interval=interval,
                start_time=current_start,
                end_time=chunk_end,
            )

            if df_chunk is not None and len(df_chunk) > 0:
                all_data.append(df_chunk)
                chunk_count += 1

                # 顯示進度
                progress = (current_start - start_ts) / (end_ts - start_ts) * 100
                print(
                    f"  進度: {progress:.1f}% - 已抓取 {len(df_chunk)} 根K線 (Chunk {chunk_count})"
                )

            current_start = chunk_end

        except Exception as e:
            print(f"  ⚠️  抓取失敗: {e}")
            current_start = chunk_end
            continue

    if not all_data:
        print("\n❌ 沒有抓取到任何數據")
        return None

    # 合併所有數據
    df = pd.concat(all_data, ignore_index=True)

    # 移除重複
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"\n✅ 數據抓取完成！")
    print(f"   總K線數: {len(df)}")
    print(f"   時間範圍: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"   價格範圍: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return df


def detect_reversal_points(df):
    """
    使用 PeakValleyDetection 檢測反轉點，並區分峰值和谷值

    PeakValleyDetection.detect() 返回：
    - 0: 無信號
    - 1: 波峰（做空信號）
    - 2: 波谷（做多信號）

    添加兩個欄位：
    - reverse: 是否為反轉點 (0/1)
    - reverse_type: 反轉類型 ('peak'/'valley'/None)
    """
    print("\n" + "=" * 80)
    print("🔍 檢測反轉點")
    print("=" * 80)

    # 使用與 visualize.py 相同的參數
    detector = PeakValleyDetection(distance=20, height_threshold=0.001)

    # detect() 返回: 0=無信號, 1=波峰, 2=波谷
    signals = detector.detect(df)

    # 初始化標籤
    df["reverse"] = 0
    df["reverse_type"] = None

    # 標記波峰 (signal == 1)
    peak_mask = signals == 1
    df.loc[peak_mask, "reverse"] = 1
    df.loc[peak_mask, "reverse_type"] = "peak"

    # 標記波谷 (signal == 2)
    valley_mask = signals == 2
    df.loc[valley_mask, "reverse"] = 1
    df.loc[valley_mask, "reverse_type"] = "valley"

    # 統計
    reversal_count = df["reverse"].sum()
    peak_count = (df["reverse_type"] == "peak").sum()
    valley_count = (df["reverse_type"] == "valley").sum()
    reversal_pct = reversal_count / len(df) * 100

    print(f"\n✅ 反轉點檢測完成！")
    print(f"   總反轉點: {reversal_count} 個 ({reversal_pct:.2f}%)")
    print(f"   峰值 (Peak): {peak_count} 個")
    print(f"   谷值 (Valley): {valley_count} 個")

    # 顯示反轉點分佈
    if reversal_count > 0:
        reversal_df = df[df["reverse"] == 1]
        print(f"\n反轉點統計:")
        print(f"   平均價格: ${reversal_df['close'].mean():.2f}")
        print(
            f"   價格範圍: ${reversal_df['close'].min():.2f} - ${reversal_df['close'].max():.2f}"
        )

    return df


def calculate_all_indicators(df, window_size=16):
    """
    計算所有技術指標
    """
    print("\n" + "=" * 80)
    print("📊 計算技術指標")
    print("=" * 80)

    print(f"\n1️⃣  計算基礎技術指標...")
    df = calculate_technical_indicators(df)

    print(f"2️⃣  計算累計指標和二階導數...")
    result_df = analyze_cumulative_metrics(df, window_size=window_size)

    print(f"\n✅ 指標計算完成！")
    print(f"   總欄位數: {len(result_df.columns)}")
    print(f"   數據點數: {len(result_df)}")

    return result_df


def visualize_reversal_points(df, output_filename="reversal_points_visualization.png"):
    """
    視覺化價格曲線和反轉點
    """
    print("\n" + "=" * 80)
    print("📈 視覺化反轉點")
    print("=" * 80)

    # 轉換時間戳
    df["datetime"] = pd.to_datetime(df["timestamp"])

    # 找出反轉點
    reversal_points = df[df["reverse"] == 1]

    print(f"\n繪製圖表...")
    print(f"   總數據點: {len(df)}")
    print(f"   反轉點: {len(reversal_points)}")

    # 創建圖表
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))

    # === 圖1: 完整時間範圍 ===
    ax1 = axes[0]
    ax1.plot(df["datetime"], df["close"], "b-", linewidth=0.5, alpha=0.7, label="Price")
    ax1.scatter(
        reversal_points["datetime"],
        reversal_points["close"],
        c="red",
        s=50,
        marker="o",
        zorder=5,
        label="Reversal Points",
    )
    ax1.set_title(
        f"Bitcoin Price with Reversal Points (Full Range)\nTotal: {len(reversal_points)} reversal points",
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
    recent_3m = df[df["datetime"] >= df["datetime"].max() - pd.Timedelta(days=90)]
    recent_3m_reversals = recent_3m[recent_3m["reverse"] == 1]

    ax2.plot(
        recent_3m["datetime"],
        recent_3m["close"],
        "b-",
        linewidth=1,
        alpha=0.7,
        label="Price",
    )
    ax2.scatter(
        recent_3m_reversals["datetime"],
        recent_3m_reversals["close"],
        c="red",
        s=100,
        marker="o",
        zorder=5,
        label="Reversal Points",
    )
    ax2.set_title(
        f"Last 3 Months Detail ({len(recent_3m_reversals)} reversal points)",
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
    recent_1m = df[df["datetime"] >= df["datetime"].max() - pd.Timedelta(days=30)]
    recent_1m_reversals = recent_1m[recent_1m["reverse"] == 1]

    ax3.plot(
        recent_1m["datetime"],
        recent_1m["close"],
        "b-",
        linewidth=1.5,
        alpha=0.7,
        label="Price",
    )
    ax3.scatter(
        recent_1m_reversals["datetime"],
        recent_1m_reversals["close"],
        c="red",
        s=150,
        marker="o",
        zorder=5,
        label="Reversal Points",
    )

    # 標註反轉點價格
    for idx, row in recent_1m_reversals.iterrows():
        ax3.annotate(
            f'${row["close"]:.0f}',
            xy=(row["datetime"], row["close"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
        )

    ax3.set_title(
        f"Last 1 Month Detail ({len(recent_1m_reversals)} reversal points)",
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

    # 保存圖表
    os.makedirs("output", exist_ok=True)
    output_path = f"output/{output_filename}"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ 圖表已保存到: {output_path}")

    # 顯示統計
    print(f"\n反轉點統計:")
    print(f"   完整範圍: {len(reversal_points)} 個")
    print(f"   最近3個月: {len(recent_3m_reversals)} 個")
    print(f"   最近1個月: {len(recent_1m_reversals)} 個")

    return output_path


def save_training_data(df, filename):
    """保存訓練數據"""
    print("\n" + "=" * 80)
    print("💾 保存訓練數據")
    print("=" * 80)

    # 確保目錄存在
    os.makedirs("training_data", exist_ok=True)

    filepath = f"training_data/{filename}"
    df.to_csv(filepath, index=False)

    print(f"\n✅ 數據已保存到: {filepath}")
    print(f"   檔案大小: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")

    # 顯示數據摘要
    print(f"\n數據摘要:")
    print(f"   總數據點: {len(df)}")
    print(
        f"   反轉點: {df['reverse'].sum()} 個 ({df['reverse'].sum()/len(df)*100:.2f}%)"
    )
    print(f"   時間範圍: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"   欄位: {list(df.columns[:10])}... (顯示前10個)")

    return filepath


def verify_data_alignment(original_df, result_df):
    """
    驗證數據對齊

    檢查：
    1. 反轉點標籤是否正確對齊
    2. 時間戳是否匹配
    3. 數據是否被正確 truncate
    """
    print("\n" + "=" * 80)
    print("🔍 驗證數據對齊")
    print("=" * 80)

    print(f"\n原始數據: {len(original_df)} 筆")
    print(f"結果數據: {len(result_df)} 筆")
    print(f"差異: {len(original_df) - len(result_df)} 筆 (因為窗口計算被 truncate)")

    # 檢查反轉點數量
    original_reversals = original_df["reverse"].sum()
    result_reversals = result_df["reverse"].sum()

    print(f"\n反轉點數量:")
    print(f"  原始: {original_reversals} 個")
    print(f"  結果: {result_reversals} 個")

    if result_reversals < original_reversals:
        lost_reversals = original_reversals - result_reversals
        print(f"  ⚠️  丟失: {lost_reversals} 個反轉點（在窗口計算時被 truncate）")

    # 檢查時間戳對齊
    if "timestamp" in result_df.columns:
        # 找出結果數據的時間範圍在原始數據中的位置
        result_start_time = result_df["timestamp"].iloc[0]
        result_end_time = result_df["timestamp"].iloc[-1]

        print(f"\n時間範圍:")
        print(f"  結果數據開始: {result_start_time}")
        print(f"  結果數據結束: {result_end_time}")

        # 檢查前5個反轉點是否對齊
        result_reversal_times = (
            result_df[result_df["reverse"] == 1]["timestamp"].head(5).tolist()
        )
        original_reversal_times = (
            original_df[original_df["reverse"] == 1]["timestamp"].head(10).tolist()
        )

        print(f"\n前5個反轉點時間戳對齊檢查:")
        for i, rt in enumerate(result_reversal_times, 1):
            if rt in original_reversal_times:
                print(f"  ✅ 反轉點 {i}: {rt} - 對齊正確")
            else:
                print(f"  ❌ 反轉點 {i}: {rt} - 對齊錯誤！")

    print(f"\n✅ 數據對齊驗證完成")


def main(start_date=None, end_date=None, interval="15m", window_size=16):
    """
    主函數

    Args:
        start_date: 開始日期 (YYYY-MM-DD)，默認 2022-01-01
        end_date: 結束日期 (YYYY-MM-DD)，默認 2025-09-01
        interval: K線間隔，默認 15m
        window_size: 窗口大小，默認 16
    """
    print("=" * 80)
    print("🚀 生成訓練數據")
    print("=" * 80)

    # 設定默認參數
    if start_date is None:
        start_date = "2022-01-01"
    if end_date is None:
        end_date = "2025-09-01"

    print(f"\n參數設定:")
    print(f"  開始日期: {start_date}")
    print(f"  結束日期: {end_date}")
    print(f"  K線間隔: {interval}")
    print(f"  窗口大小: {window_size}")

    # 計算預期數據量
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end_dt - start_dt).days

    if interval == "15m":
        expected_klines = days * 24 * 4
    elif interval == "1h":
        expected_klines = days * 24
    elif interval == "4h":
        expected_klines = days * 6
    else:
        expected_klines = days

    print(f"\n預期數據量:")
    print(f"  天數: {days} 天")
    print(f"  預期K線數: ~{expected_klines:,} 根")
    print(f"  窗口計算後: ~{expected_klines - window_size:,} 根")

    # 1. 抓取歷史數據
    df = fetch_historical_data(start_date, end_date, interval)

    if df is None:
        print("\n❌ 數據抓取失敗，程序終止")
        return

    # 2. 檢測反轉點
    df = detect_reversal_points(df)

    # 3. 視覺化反轉點（在計算指標前，使用原始數據）
    viz_filename = f"reversal_points_{start_date}_to_{end_date}_{interval}.png"
    viz_path = visualize_reversal_points(df, viz_filename)

    # 4. 計算所有指標
    result_df = calculate_all_indicators(df, window_size)

    # 5. 驗證數據對齊
    verify_data_alignment(df, result_df)

    # 6. 保存訓練數據
    filename = f"reversal_training_data_{start_date}_to_{end_date}_{interval}.csv"
    filepath = save_training_data(result_df, filename)

    print("\n" + "=" * 80)
    print("✅ 訓練數據生成完成！")
    print("=" * 80)

    print(f"\n生成的檔案:")
    print(f"  📊 訓練數據: {filepath}")
    print(f"  📈 視覺化圖表: {viz_path}")

    print(f"\n下一步:")
    print(f"  1. 檢查視覺化圖表，確認反轉點檢測正確")
    print(f"  2. 使用新數據重新訓練模型")
    print(f"  3. 比較新舊數據的性能差異")
    print(f"  4. 分析更多反轉點的特徵")

    return filepath, viz_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成反轉檢測訓練數據")
    parser.add_argument(
        "--start",
        type=str,
        default="2022-01-01",
        help="開始日期 (YYYY-MM-DD)，默認 2022-01-01",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-09-01",
        help="結束日期 (YYYY-MM-DD)，默認 2025-09-01",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="15m",
        choices=["15m", "1h", "4h", "1d"],
        help="K線間隔，默認 15m",
    )
    parser.add_argument("--window", type=int, default=16, help="窗口大小，默認 16")

    args = parser.parse_args()

    main(
        start_date=args.start,
        end_date=args.end,
        interval=args.interval,
        window_size=args.window,
    )
