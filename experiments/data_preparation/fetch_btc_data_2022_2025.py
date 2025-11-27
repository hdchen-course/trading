import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import time

def fetch_btc_data_2022_2025():
    """抓取2022年到2025年10月的BTC 15分鐘K線數據"""
    
    # 初始化Binance客戶端 (不需要API key獲取公開數據)
    client = Client()
    
    # 設定時間範圍
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2025, 10, 31)
    
    print(f"開始抓取BTC 15分鐘K線數據: {start_date} 到 {end_date}")
    
    all_data = []
    current_date = start_date
    
    # 分批抓取數據 (每次抓取30天)
    while current_date < end_date:
        batch_end = min(current_date + timedelta(days=30), end_date)
        
        print(f"抓取 {current_date.strftime('%Y-%m-%d')} 到 {batch_end.strftime('%Y-%m-%d')}")
        
        try:
            # 獲取K線數據
            klines = client.get_historical_klines(
                "BTCUSDT",
                Client.KLINE_INTERVAL_15MINUTE,
                current_date.strftime("%Y-%m-%d"),
                batch_end.strftime("%Y-%m-%d")
            )
            
            # 轉換數據格式
            for kline in klines:
                data_point = {
                    'timestamp': pd.to_datetime(kline[0], unit='ms'),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                }
                all_data.append(data_point)
            
            print(f"  獲取了 {len(klines)} 條記錄")
            
            # 避免API限制
            time.sleep(1)
            
        except Exception as e:
            print(f"錯誤: {e}")
            time.sleep(5)
            continue
        
        current_date = batch_end
    
    # 轉換為DataFrame並保存
    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    
    print(f"\n總共獲取了 {len(df)} 條記錄")
    print(f"時間範圍: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    # 保存數據
    filename = 'training_data/btc_15m_2022_2025.csv'
    df.to_csv(filename, index=False)
    print(f"數據已保存到: {filename}")
    
    # 顯示數據統計
    print(f"\n數據統計:")
    print(f"開始時間: {df['timestamp'].min()}")
    print(f"結束時間: {df['timestamp'].max()}")
    print(f"價格範圍: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"平均成交量: {df['volume'].mean():.2f}")
    
    return df

if __name__ == "__main__":
    # 創建training_data目錄
    import os
    os.makedirs('training_data', exist_ok=True)
    
    # 抓取數據
    df = fetch_btc_data_2022_2025()
