#!/usr/bin/env python3
"""
兼容層 - 保持舊的 import 路徑可用

這個文件提供向後兼容，讓現有代碼可以繼續使用舊的 import 路徑
例如: from crypto_strategies import MACDStrategy

建議: 逐步更新到新的 import 路徑
例如: from strategies.technical.macd import MACDStrategy
"""

import warnings

# 顯示棄用警告
warnings.warn(
    "從 crypto_strategies 導入已棄用，請使用 strategies 模組",
    DeprecationWarning,
    stacklevel=2
)

# 從新模組導入所有內容
from strategies import *

__all__ = [
    "BaseStrategy",
    "StrategyConfig",
    "StrategyManager",
    "CoinConfig",
    "MACDStrategy",
    "MACDVolumeStrategy",
    "BollingerStrategy",
    "BollingerExtremeStrategy",
    "StochRSIStrategy",
    "ATRStrategy",
    "SARStrategy",
    "VolumeStrategy",
    "TrendMLStrategy",
    "TrendPredictionStrategy",
    "ReversalStrategy",
    "MLReversalStrategy",
    "CrashPredictionStrategy",
    "LSTMStrategy",
]
