#!/usr/bin/env python3
"""
機器學習策略模組
"""

from .trend import TrendMLStrategy, TrendPredictionStrategy, SeparateTrendStrategy
from .reversal import ReversalStrategy, MLReversalStrategy
from .ml_reversal_v2 import MLReversalV2Strategy
from .ml_reversal_v3 import MLReversalV3Strategy
from .ml_reversal_v4 import MLReversalV4Strategy
from .ml_reversal_v5 import MLReversalV5Strategy
from .macd_v1 import MACDV1Strategy
from .crash import CrashPredictionStrategy

__all__ = [
    "TrendMLStrategy",
    "TrendPredictionStrategy",
    "SeparateTrendStrategy",
    "ReversalStrategy",
    "MLReversalStrategy",
    "MLReversalV2Strategy",
    "MLReversalV3Strategy",
    "MLReversalV4Strategy",
    "MLReversalV5Strategy",
    "MACDV1Strategy",
    "CrashPredictionStrategy",
]
