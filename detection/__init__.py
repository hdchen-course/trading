#!/usr/bin/env python3
"""
detection 模組 - 各種價格反轉和信號檢測方法
"""

from .base import DetectionMethod
from .macd import MacdReversalDetection, BTCMACDBuyDetection, MacdVolumeDetection
from .peak_valley import PeakValleyDetection
from .optimal_reversal import (
        OptimalReversalDetection,
        EnsembleReversalDetection,
        RSIBBReversalDetection,
        PositionBBReversalDetection,
        )
from .triple_confirmation_reversal import TripleConfirmationReversalDetector
from .ml_reversal import MLReversalDetector

__all__ = [
        "DetectionMethod",
        "MacdReversalDetection",
        "BTCMACDBuyDetection",
        "MacdVolumeDetection",
        "PeakValleyDetection",
        "OptimalReversalDetection",
        "EnsembleReversalDetection",
        "RSIBBReversalDetection",
        "PositionBBReversalDetection",
        "TripleConfirmationReversalDetector",
        "MLReversalDetector",
        ]
