#!/usr/bin/env python3
"""
機器學習策略模組
"""

from .trend import TrendMLStrategy, TrendPredictionStrategy, SeparateTrendStrategy
from .reversal import ReversalStrategy, MLReversalStrategy
from .ml_reversal_v2 import (
    MLReversalV2Strategy,
    MLReversalV2BuyStrategy,
    MLReversalV2SellStrategy,
)
from .ml_reversal_v3 import MLReversalV3Strategy
from .crash import CrashPredictionStrategy

__all__ = [
    "TrendMLStrategy",
    "TrendPredictionStrategy",
    "SeparateTrendStrategy",
    "ReversalStrategy",
    "MLReversalStrategy",
    "MLReversalV2Strategy",
    "MLReversalV2BuyStrategy",
    "MLReversalV2SellStrategy",
    "MLReversalV3Strategy",
    "CrashPredictionStrategy",
]
