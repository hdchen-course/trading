#!/usr/bin/env python3
"""
交易配置模組
"""

from dataclasses import dataclass


@dataclass
class TradingConfig:
    """交易系統配置類"""
    
    # 時間相關配置
    timeframe_minutes: int = 15
    monitoring_interval: int = 30
    signal_check_minutes: int = 5
    monitoring_hours: int = 4

    # 資金管理配置
    initial_capital: float = 1000000.0
    position_size_ratio: float = 0.25
    open_fee_rate: float = 0.0002
    close_fee_rate: float = 0.0005

    # 風險控制配置
    stop_loss_pct: float = 0.018
    take_profit_pct: float = 0.10
    enable_stop_loss: bool = True
    enable_take_profit: bool = True

    # 外部服務配置
    binance_base_url: str = "https://api.binance.com"
    telegram_bot_token: str = ""
    telegram_chat_id: int = 0

    @property
    def timeframe_str(self) -> str:
        """將時間間隔轉換為 Binance API 格式"""
        if self.timeframe_minutes == 15:
            return "15m"
        elif self.timeframe_minutes == 30:
            return "30m"
        else:
            return f"{self.timeframe_minutes}m"
