#!/usr/bin/env python3
"""
策略基礎類

定義所有策略的共同接口和管理器
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    """
    策略配置基類

    Attributes:
        name: 策略名稱
        params: 策略參數字典
        weight: 策略權重 (用於綜合策略)
    """

    name: str
    params: Dict[str, Any]
    weight: float = 1.0


class BaseStrategy(ABC):
    """
    策略基類接口

    所有具體策略都必須繼承此類並實現抽象方法
    """

    def __init__(self, config: StrategyConfig):
        """
        初始化策略

        Args:
            config: 策略配置對象
        """
        self.config = config
        self.name = config.name
        self.params = config.params
        self.weight = config.weight

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信號序列

        Args:
            df: 包含 OHLCV 數據的 DataFrame

        Returns:
            pd.Series: 交易信號序列
                      1 = 買入信號
                     -1 = 賣出信號
                      0 = 無信號
        """
        pass

    def generate_signal(self, df: pd.DataFrame) -> str:
        """
        生成單一交易信號 (用於實時監控)

        Args:
            df: 包含 OHLCV 數據的 DataFrame

        Returns:
            str: 'BUY', 'SELL', 或 'HOLD'
        """
        signals = self.generate_signals(df)
        if len(signals) == 0:
            return "HOLD"

        last_signal = signals.iloc[-1]
        if last_signal == 1:
            return "BUY"
        elif last_signal == -1:
            return "SELL"
        else:
            return "HOLD"

    @abstractmethod
    def get_description(self) -> str:
        """
        返回策略描述

        Returns:
            str: 策略的詳細描述
        """
        pass


@dataclass
class CoinConfig:
    """
    幣種配置類

    Attributes:
        name: 幣種名稱
        symbol: 交易對符號
        strategies: 策略列表
        leverage: 槓桿倍數
        enable_monitor: 是否啟用監控
    """

    name: str
    symbol: str
    strategies: List[BaseStrategy]
    leverage: float = 1.0
    enable_monitor: bool = True


class StrategyManager:
    """
    策略管理器

    負責管理多個策略並合併信號
    """

    def __init__(self):
        """初始化策略管理器"""
        self.coin_configs = []

    def add_coin_config(self, coin_config: CoinConfig):
        """
        添加幣種配置

        Args:
            coin_config: 幣種配置對象
        """
        self.coin_configs.append(coin_config)

    def get_all_configs(self) -> List[CoinConfig]:
        """
        獲取所有幣種配置

        Returns:
            List[CoinConfig]: 所有已註冊的幣種配置
        """
        return self.coin_configs

    def generate_combined_signal(self, df: pd.DataFrame, coin_config: CoinConfig):
        """
        生成綜合策略信號 (用於實時監控)

        Args:
            df: 市場數據 DataFrame
            coin_config: 幣種配置對象

        Returns:
            Tuple[float, float, float]: (最新信號, 買入強度, 賣出強度)
        """
        signals, signal_strength, _, _ = self.generate_combined_signal_vectorized(df, coin_config)

        if len(signals) == 0:
            return 0, 0.0, 0.0

        latest_signal = signals.iloc[-1]
        latest_strength = signal_strength.iloc[-1]
        
        # 根據信號方向分配強度
        buy_strength = latest_strength if latest_signal == 1 else 0.0
        sell_strength = latest_strength if latest_signal == -1 else 0.0

        return latest_signal, buy_strength, sell_strength

    def generate_combined_signal_vectorized(
        self, df: pd.DataFrame, coin_config: CoinConfig
    ):
        """
        生成綜合策略信號序列 (向量化版本)

        Args:
            df: 市場數據 DataFrame
            coin_config: 幣種配置對象

        Returns:
            Tuple[pd.Series, pd.Series, float, float]: 
                (信號序列, 信號強度序列, 買入強度, 賣出強度)
                信號強度 = 加權分數的絕對值，範圍 [0, 總權重]
        """
        strategy_signals = []

        for strategy in coin_config.strategies:
            try:
                strategy_signal = strategy.generate_signals(df)
                if strategy_signal is not None and len(strategy_signal) > 0:
                    strategy_signals.append((strategy_signal, strategy.weight))
            except Exception:
                continue

        if not strategy_signals:
            return pd.Series(0, index=df.index), pd.Series(0.0, index=df.index), 0, 0

        # 向量化計算綜合信號
        signals_df = pd.DataFrame(
            {
                f"signal_{i}": signal
                for i, (signal, _) in enumerate(strategy_signals)
            }
        )

        weights = np.array([weight for _, weight in strategy_signals])
        weighted_scores = (signals_df.values * weights).sum(axis=1)

        # 計算信號強度（加權分數的絕對值）
        signal_strength = pd.Series(np.abs(weighted_scores), index=df.index)

        final_signals = pd.Series(0, index=df.index)
        final_signals[weighted_scores > 0] = 1
        final_signals[weighted_scores < 0] = -1

        return final_signals, signal_strength, 0, 0
