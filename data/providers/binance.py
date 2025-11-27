from .base import DataProvider

from abc import ABC, abstractmethod
import pandas as pd
import requests
import argparse
from utils.indicators import calculate_technical_indicators, analyze_cumulative_metrics


class DataProvider(ABC):
    @abstractmethod
    def get_data(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        pass


class BinanceDataProvider(DataProvider):
    def __init__(self, base_url: str = "https://api.binance.com"):
        self.base_url = base_url

    def get_data(self, symbol: str, interval: str, limit: int = 1000, 
                 start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        獲取 K 線數據
        
        Args:
            symbol: 交易對符號 (例如 BTCUSDT)
            interval: 時間間隔 (例如 15m, 1h, 1d)
            limit: 返回的數據點數量 (默認 1000，最大 1000)
            start_time: 開始時間戳 (毫秒) (可選)
            end_time: 結束時間戳 (毫秒) (可選)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        all_data = []
        max_per_request = 1000

        # 如果提供了時間範圍，使用時間範圍模式
        if start_time is not None and end_time is not None:
            current_start = start_time
            
            while current_start < end_time:
                url = f"{self.base_url}/api/v3/klines"
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": end_time,
                    "limit": max_per_request,
                }

                try:
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()

                    if not data:
                        break

                    all_data.extend(data)
                    
                    # 更新下一次請求的開始時間
                    current_start = data[-1][0] + 1
                    
                    if len(data) < max_per_request:
                        break

                except Exception as e:
                    print(f"獲取數據時發生錯誤: {e}")
                    break
        else:
            # 使用 limit 模式（向後抓取）
            while limit > 0:
                current_limit = min(limit, max_per_request)
                url = f"{self.base_url}/api/v3/klines"
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "limit": current_limit,
                }

                if all_data:
                    params["endTime"] = all_data[0][0] - 1

                try:
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()

                    if not data:
                        break

                    all_data = data + all_data
                    limit -= len(data)

                    if len(data) < current_limit:
                        break

                except Exception as e:
                    print(f"獲取數據時發生錯誤: {e}")
                    break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_data,
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

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])

        return df[["timestamp", "open", "high", "low", "close", "volume"]]


def main():
    parser = argparse.ArgumentParser(description='Download crypto data and calculate indicators')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., BTCUSDT)')
    
    args = parser.parse_args()
    
    # Calculate data points needed (15min intervals)
    start_date = pd.to_datetime(args.start)
    end_date = pd.to_datetime(args.end)
    days = (end_date - start_date).days + 1
    data_points = days * 24 * 4  # 4 data points per hour for 15min intervals
    
    # Download data
    provider = BinanceDataProvider()
    df = provider.get_data(args.symbol, "15m", data_points)
    
    # Filter by date range
    df = df[(df['timestamp'] >= args.start) & (df['timestamp'] <= args.end)]
    
    # Calculate indicators
    df_with_indicators = calculate_technical_indicators(df)
    df_final = analyze_cumulative_metrics(df_with_indicators)
    
    # Save to CSV
    filename = f"{args.symbol}_{args.start}_{args.end}_indicators.csv"
    df_final.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


if __name__ == "__main__":
    main()
