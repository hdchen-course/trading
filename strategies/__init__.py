#!/usr/bin/env python3
"""
策略模組

提供所有交易策略的統一接口
"""

from .base import BaseStrategy, StrategyConfig, StrategyManager, CoinConfig

# 技術指標策略
from .technical.macd import (
    MACDStrategy,
    MACDVolumeStrategy,
    MacdReversalStrategy,
    MacdVolumeStrategy,
    BTCMACDBuyStrategy,
)
from .technical.bollinger import (
    BollingerStrategy,
    BollingerExtremeStrategy,
    EnhancedBollingerStrategy,
)
from .technical.momentum import (
    StochRSIStrategy,
    ATRStrategy,
    SARStrategy,
)
from .technical.volume import VolumeStrategy
from .technical.candlestick import CandlestickStrategy
from .technical.trend_confirmation import TrendConfirmationStrategy
from .technical.triple_confirmation import (
    TripleConfirmationStrategy,
    TripleConfirmationBuyStrategy,
    TripleConfirmationSellStrategy,
)

# 機器學習策略
from .ml.trend import TrendMLStrategy, TrendPredictionStrategy, SeparateTrendStrategy
from .ml.reversal import ReversalStrategy, MLReversalStrategy
from .ml.crash import CrashPredictionStrategy
from .ml.lstm import LSTMStrategy

__all__ = [
    # 基礎類
    "BaseStrategy",
    "StrategyConfig",
    "StrategyManager",
    "CoinConfig",
    # 技術指標策略
    "MACDStrategy",
    "MACDVolumeStrategy",
    "MacdReversalStrategy",
    "MacdVolumeStrategy",
    "BTCMACDBuyStrategy",
    "BollingerStrategy",
    "BollingerExtremeStrategy",
    "EnhancedBollingerStrategy",
    "StochRSIStrategy",
    "ATRStrategy",
    "SARStrategy",
    "VolumeStrategy",
    "CandlestickStrategy",
    "TrendConfirmationStrategy",
    "TripleConfirmationStrategy",
    "TripleConfirmationBuyStrategy",
    "TripleConfirmationSellStrategy",
    # 機器學習策略
    "TrendMLStrategy",
    "TrendPredictionStrategy",
    "SeparateTrendStrategy",
    "ReversalStrategy",
    "MLReversalStrategy",
    "CrashPredictionStrategy",
    "LSTMStrategy",
]
