#!/usr/bin/env python3
"""
加密貨幣監控模組

實時監控加密貨幣價格並生成交易信號:
- 從 Binance API 獲取實時價格數據
- 使用策略管理器生成交易信號
- 支持多幣種同時監控
- 定時任務調度功能

主要功能:
1. 數據獲取: 從 Binance 獲取 K 線數據
2. 信號生成: 使用技術分析策略生成交易信號
3. 實時監控: 定期檢查價格變化和信號
4. 日誌記錄: 記錄所有監控活動和信號
"""

import logging  # 日誌記錄庫, 用於記錄監控活動
import time  # 時間處理庫, 用於延遲和時間戳
from datetime import timedelta  # 時間差計算
from typing import List  # 類型提示, 提高代碼可讀性

import pandas as pd  # 數據處理庫, 用於時間轉換
import requests  # HTTP 請求庫, 用於發送 Telegram 通知
import schedule  # 任務調度庫, 用於定時執行監控

from data.providers.binance import BinanceDataProvider  # 導入數據提供者
from strategies import CoinConfig, StrategyManager  # 導入策略相關類
from utils.indicators import (
    analyze_cumulative_metrics,
    calculate_technical_indicators,
)


class CryptoMonitor:
    """
    加密貨幣監控器 - 實時監控價格並生成交易信號

    核心功能:
    1. 數據獲取: 從 Binance API 獲取實時 K 線數據
    2. 信號分析: 使用策略管理器分析價格趨勢
    3. 實時監控: 定期檢查多個幣種的交易機會
    4. 風險控制: 監控價格異常和市場風險

    使用場景:
    - 自動化交易系統的信號源
    - 價格預警和通知系統
    - 市場趨勢分析工具
    """

    def __init__(self, config, debug_mode=False, debug_time=None):
        """
        初始化監控器

        參數:
            config: 配置對象, 包含 API 設置和監控參數
            debug_mode: 是否為 debug 模式
            debug_time: debug 模式下的指定時間點 (datetime 對象)
        """
        self.config = config  # 保存配置對象
        self.strategy_manager = StrategyManager()  # 初始化策略管理器
        self.data_provider = BinanceDataProvider()  # 初始化數據提供者
        self.debug_mode = debug_mode  # Debug 模式標記
        self.debug_time = debug_time  # Debug 時間點

    def send_telegram_notification(self, message: str):
        """
        發送 Telegram 通知消息

        用於將交易信號和重要事件通知到 Telegram 群組或個人:
        1. 構建 Telegram Bot API 請求
        2. 發送消息到指定聊天室
        3. 處理發送失敗的情況

        參數:
            message: 要發送的消息內容
        """
        try:
            # 從配置中獲取 Telegram Bot 信息
            token = self.config.telegram_bot_token  # Bot 令牌
            chat_id = self.config.telegram_chat_id  # 聊天室 ID

            # 構建 Telegram Bot API URL
            bot_url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = {"chat_id": chat_id, "text": message}  # 請求數據

            # 發送 POST 請求到 Telegram API
            requests.post(bot_url, data=data)
        except Exception as e:
            logging.error(f"Failed to send notification: {e}")

    def check_signals(self, coin_config: CoinConfig):
        """
        檢查交易信號 - 使用綜合策略分析

        這是監控系統的核心方法, 執行完整的信號分析流程:
        1. 獲取最新的價格數據
        2. 使用策略管理器生成綜合信號
        3. 分析買入和賣出信號強度
        4. 記錄信號結果和價格信息
        5. 發送重要信號的通知

        參數:
            coin_config: 幣種配置對象, 包含交易對和策略設置
        """
        logging.info(f"🔍 檢查 {coin_config.name} 信號...")

        # 固定窗口計算需要至少 1000 根 K 線
        min_periods = 1000

        if self.debug_mode and self.debug_time:
            # Debug 模式: 獲取到指定時間點為止的數據
            # 計算需要的時間範圍 (1000 根 15 分鐘 K 線 = 250 小時)
            hours_needed = min_periods * 15 / 60  # 15分鐘 K 線
            start_time = self.debug_time - timedelta(hours=hours_needed)
            end_time = self.debug_time + timedelta(minutes=15)  # 包含目標時間點

            logging.info(f"🐛 Debug 模式: 獲取到 {self.debug_time} 為止的數據")
            logging.info(f"   時間範圍: {start_time} 到 {end_time}")
            logging.info(f"   需要至少 {min_periods} 根 K 線")

            # 轉換為毫秒時間戳
            start_time_ms = int(start_time.timestamp() * 1000)
            end_time_ms = int(end_time.timestamp() * 1000)

            df = self.data_provider.get_data(
                symbol=coin_config.symbol,
                interval=self.config.timeframe_str,
                limit=min_periods + 100,  # 額外加 100 根緩衝
                start_time=start_time_ms,
                end_time=end_time_ms,
            )
        else:
            # 正常模式: 獲取最新數據（至少 1000 根）
            df = self.data_provider.get_data(
                coin_config.symbol, self.config.timeframe_str, min_periods + 100
            )

        if df is None or df.empty:
            logging.error(f"❌ 無法取得 {coin_config.symbol} 數據")
            return

        df = calculate_technical_indicators(df)
        df = analyze_cumulative_metrics(df, window_size=16)

        # Debug 模式: 顯示詳細信息並保存數據
        if self.debug_mode:
            logging.info(f"\n📊 數據信息:")
            logging.info(f"   總 K 線數: {len(df)}")
            logging.info(
                f"   時間範圍 (UTC): {df['timestamp'].min()} 到 {df['timestamp'].max()}"
            )

            # Debug 模式：找到目標時間點的數據
            target_time_str = self.debug_time.strftime("%Y-%m-%d %H:%M:%S")
            target_row = df[df["timestamp"] == target_time_str]

            if target_row.empty:
                logging.error(f"   ❌ 找不到目標時間點 {target_time_str}")
                logging.info(
                    f"   可用時間範圍: {df['timestamp'].min()} 到 {df['timestamp'].max()}"
                )
                return

            # 使用目標時間點的數據
            target_idx = target_row.index[0]
            current_price = df["close"].iloc[target_idx]
            timestamp = df["timestamp"].iloc[target_idx]

            logging.info(f"   目標時間 (UTC): {timestamp}")
            logging.info(f"   目標價格: {current_price:.2f}")
            logging.info(f"   目標索引: {target_idx}")

            # 裁剪 df 到目標時間點（模擬實時監控只能看到這個時間點之前的數據）
            df = df.iloc[: target_idx + 1].copy()
            logging.info(f"   裁剪後數據量: {len(df)} 根")
        else:
            # 正常模式：使用最新數據
            current_price = df["close"].iloc[-1]
            timestamp = df["timestamp"].iloc[-1]

        # 使用策略管理器生成綜合信號
        # 返回: signal (1=買入, -1=賣出, 0=持有), 買入強度, 賣出強度
        signal, buy_strength, sell_strength = (
            self.strategy_manager.generate_combined_signal(df, coin_config)
        )

        # Debug 模式：顯示所有信號
        if self.debug_mode:
            logging.info(f"\n🔍 Debug: 顯示所有信號")
            signals_all, signal_strength_all, _, _ = (
                self.strategy_manager.generate_combined_signal_vectorized(
                    df, coin_config
                )
            )
            buy_signals = signals_all == 1
            sell_signals = signals_all == -1

            if buy_signals.sum() > 0:
                logging.info(f"   買入信號數量: {buy_signals.sum()}")
                for idx in df[buy_signals].index[:5]:
                    logging.info(f"      {df.loc[idx, 'timestamp']}")

            if sell_signals.sum() > 0:
                logging.info(f"   賣出信號數量: {sell_signals.sum()}")
                for idx in df[sell_signals].index[:5]:
                    logging.info(f"      {df.loc[idx, 'timestamp']}")

            if buy_signals.sum() == 0 and sell_signals.sum() == 0:
                logging.info(f"   無買入或賣出信號")

        # 記錄信號強度信息
        logging.info(
            f"\n📊 綜合信號強度 - 買入: {buy_strength:.1%}, "
            f"賣出: {sell_strength:.1%}"
        )

        if signal == 1:  # 買入信號
            # 轉換為 UTC+8 用於顯示
            timestamp_display = timestamp + pd.Timedelta(hours=8)

            # 構建買入信號通知消息
            message = (
                f"🟢 {coin_config.name} 買入信號！\n"
                f"價格: ${current_price:.4f}\n"
                f"時間: {timestamp_display} (UTC+8)\n"
                f"買入強度: {buy_strength:.1%}\n"
                f"賣出強度: {sell_strength:.1%}"
            )
            logging.info(f"📢 BUY 信號: {coin_config.name} at {timestamp} UTC")
            self.send_telegram_notification(message)  # 發送通知

        elif signal == -1:  # 賣出信號
            # 轉換為 UTC+8 用於顯示
            timestamp_display = timestamp + pd.Timedelta(hours=8)

            # 構建賣出信號通知消息
            message = (
                f"🔴 {coin_config.name} 賣出信號！\n"
                f"價格: ${current_price:.4f}\n"
                f"時間: {timestamp_display} (UTC+8)\n"
                f"買入強度: {buy_strength:.1%}\n"
                f"賣出強度: {sell_strength:.1%}"
            )
            logging.info(f"📢 SELL 信號: {coin_config.name} at {timestamp} UTC")
            self.send_telegram_notification(message)  # 發送通知

    def monitor_coins(self, coin_configs: List[CoinConfig]):
        """
        監控多個幣種的交易信號

        這是監控系統的主循環, 負責:
        1. 設置定時任務調度
        2. 循環檢查所有配置的幣種
        3. 處理監控過程中的異常
        4. 支持優雅的停止機制

        參數:
            coin_configs: 幣種配置列表, 每個配置包含交易對和策略
        """
        logging.info("🚀 開始監控...")

        def check_all_signals():
            """檢查所有幣種的信號 - 內部函數"""
            for coin_config in coin_configs:
                self.check_signals(coin_config)  # 逐個檢查每個幣種

        # 設置定時任務: 每隔指定分鐘數檢查一次信號
        schedule.every(self.config.signal_check_minutes).minutes.do(check_all_signals)

        # 立即執行一次檢查
        check_all_signals()

        # 主監控循環
        while True:
            try:
                schedule.run_pending()  # 執行待處理的定時任務
                time.sleep(self.config.monitoring_interval)  # 等待指定間隔
            except KeyboardInterrupt:
                logging.info("👋 監控已停止")
                break  # 用戶中斷, 優雅退出
            except Exception as e:
                logging.error(f"監控錯誤: {e}")
                time.sleep(60)  # 發生錯誤時等待 1 分鐘後繼續
