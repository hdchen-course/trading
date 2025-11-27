#!/usr/bin/env python3
import pandas as pd
import argparse
from datetime import datetime
import warnings
from utils.indicators import calculate_technical_indicators, analyze_cumulative_metrics

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

        # 如果CSV沒有足夠數據，使用API抓取
        if len(filtered_df) == 0 or (
            len(filtered_df) > 0 and filtered_df["timestamp"].max() < end_date
        ):
            print("CSV數據不足，使用API抓取即時數據...")
            from data.providers.binance import BinanceDataProvider

            provider = BinanceDataProvider()
            
            # 轉換為毫秒時間戳
            start_time_ms = int(start_date.timestamp() * 1000)
            end_time_ms = int(end_date.timestamp() * 1000)

            api_df = provider.get_data("BTCUSDT", "15m", 
                                       start_time=start_time_ms, 
                                       end_time=end_time_ms)

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


# 配置參數
ROLLING_WINDOW_HOURS = 4  # 滾動窗口小時數
ROLLING_WINDOW_SIZE = ROLLING_WINDOW_HOURS * 4  # 轉換為15分鐘週期數


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


def parse_datetime(date_str):
    """解析日期時間 - 支援多種格式"""
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


def run_analysis(start_date_str, end_date_str, output_file=None):
    """
    執行技術分析
    
    Args:
        start_date_str: 開始日期字串 (YYYY-MM-DD)
        end_date_str: 結束日期字串 (YYYY-MM-DD)
        output_file: 輸出CSV文件路徑 (可選)
    """
    try:
        start_date = parse_datetime(start_date_str)
        end_date = parse_datetime(end_date_str)
    except ValueError as e:
        print(e)
        return

    print(f"Analyzing crypto data from {start_date_str} to {end_date_str}")
    print("=" * 60)

    # 獲取數據
    df = fetch_data(start_date, end_date)
    if df is None:
        return

    # 計算技術指標
    print("Calculating technical indicators...")
    df = calculate_technical_indicators(df)

    # 分析累計指標
    print("Analyzing cumulative metrics...")
    analysis_df = analyze_cumulative_metrics(df, ROLLING_WINDOW_SIZE)

    # 顯示結果
    print(f"\nCumulative Technical Analysis ({len(analysis_df)} periods)")
    print("=" * 140)

    # 設置pandas顯示選項
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 15)

    print(
        analysis_df[
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
                "macd_2nd",
                "price_2nd",
                "vol_2nd",
                "atr_2nd",
            ]
        ].to_string(index=False)
    )

    # 保存到文件
    if output_file:
        analysis_df.to_csv(output_file, index=False)
        print(f"\nFull analysis saved to {output_file}")

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


def main():
    """命令行入口"""
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
    args = parser.parse_args()
    
    run_analysis(args.start, args.end, args.output)


if __name__ == "__main__":
    main()
