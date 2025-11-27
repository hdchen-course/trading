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
import requests  # HTTP 請求庫, 用於發送 Telegram 通知
import schedule  # 任務調度庫, 用於定時執行監控
import pandas as pd  # 數據處理庫, 用於時間轉換
from typing import List  # 類型提示, 提高代碼可讀性
from strategies import CoinConfig, StrategyManager  # 導入策略相關類
from data.providers.binance import BinanceDataProvider  # 導入數據提供者


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

    def __init__(self, config):
        """
        初始化監控器

        參數:
            config: 配置對象, 包含 API 設置和監控參數
        """
        self.config = config  # 保存配置對象
        self.strategy_manager = StrategyManager()  # 初始化策略管理器
        self.data_provider = BinanceDataProvider()  # 初始化數據提供者

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

        # 獲取 320 根 K 線數據, 確保技術指標有足夠的歷史數據
        df = self.data_provider.get_data(
            coin_config.symbol, self.config.timeframe_str, 320
        )
        if df.empty:
            logging.error(f"❌ 無法取得 {coin_config.symbol} 數據")
            return

        # 提取當前價格和時間戳
        current_price = df["close"].iloc[-1]  # 最新收盤價
        # 將 UTC 時間轉換為 UTC+8 (台北時間)
        timestamp_utc = df["timestamp"].iloc[-1]
        timestamp = timestamp_utc + pd.Timedelta(hours=8)

        # 使用策略管理器生成綜合信號
        # 返回: signal (1=買入, -1=賣出, 0=持有), 買入強度, 賣出強度
        signal, buy_strength, sell_strength = (
            self.strategy_manager.generate_combined_signal(df, coin_config)
        )

        # 記錄信號強度信息
        logging.info(
            f"📊 信號強度 - 買入: {buy_strength:.1%}, " f"賣出: {sell_strength:.1%}"
        )

        if signal == 1:  # 買入信號
            # 構建買入信號通知消息
            message = (
                f"🟢 {coin_config.name} 買入信號！\n"
                f"價格: ${current_price:.4f}\n"
                f"時間: {timestamp}\n"
                f"買入強度: {buy_strength:.1%}\n"
                f"賣出強度: {sell_strength:.1%}"
            )
            logging.info(f"📢 BUY 信號: {coin_config.name}")
            self.send_telegram_notification(message)  # 發送通知

        elif signal == -1:  # 賣出信號
            # 構建賣出信號通知消息
            message = (
                f"🔴 {coin_config.name} 賣出信號！\n"
                f"價格: ${current_price:.4f}\n"
                f"時間: {timestamp}\n"
                f"買入強度: {buy_strength:.1%}\n"
                f"賣出強度: {sell_strength:.1%}"
            )
            logging.info(f"📢 SELL 信號: {coin_config.name}")
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
