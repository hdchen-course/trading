#!/usr/bin/env python3
"""
Optimal Reversal Detection

Based on extensive experimental analysis (Tasks 2-4), this implements
the best-performing reversal detection method:

Primary Rule: Extreme Position + Volatility Acceleration
- F1 Score: 15.3%
- Precision: 10.6%
- Recall: 27.5%

The method detects potential trend reversals by identifying:
1. Extreme price positions (top 20% or bottom 20% of recent range)
2. Accelerating volatility (positive 2nd derivative of ATR)

This combination captures reversals at price extremes where volatility
is increasing, indicating potential momentum shifts.
"""

import numpy as np
import pandas as pd
from .base import DetectionMethod


class OptimalReversalDetection(DetectionMethod):
    """
    Optimal Reversal Detection using Extreme Position + Volatility Acceleration
    
    This method identifies potential trend reversals by detecting:
    - Extreme price positions (relative to 24h range)
    - Accelerating volatility (ATR 2nd derivative)
    
    Parameters:
        extreme_threshold: Position threshold for extremes (default: 0.2)
                          Values < threshold = extreme low
                          Values > (1-threshold) = extreme high
        atr_accel_threshold: ATR 2nd derivative threshold (default: 0.5)
        lookback_hours: Hours to look back for position calculation (default: 24)
        interval_minutes: Data interval in minutes (default: 15)
    """
    
    def __init__(self, 
                 extreme_threshold=0.2,
                 atr_accel_threshold=0.5,
                 lookback_hours=24,
                 interval_minutes=15):
        super().__init__("Optimal Reversal")
        self.extreme_threshold = extreme_threshold
        self.atr_accel_threshold = atr_accel_threshold
        self.lookback_hours = lookback_hours
        self.interval_minutes = interval_minutes
        self.lookback_periods = int(lookback_hours * 60 / interval_minutes)
    
    def detect(self, df):
        """
        Detect potential reversals using extreme position + volatility acceleration
        
        Args:
            df: DataFrame with columns: close, atr (or high/low for ATR calculation)
            
        Returns:
            numpy boolean array indicating reversal signal positions
        """
        n = len(df)
        
        # Need minimum data for calculations
        if n < self.lookback_periods:
            return np.zeros(n, dtype=bool)
        
        # Calculate price position relative to 24h range
        position_24h = self._calculate_price_position(df)
        
        # Calculate ATR 2nd derivative (volatility acceleration)
        atr_2nd = self._calculate_atr_second_derivative(df)
        
        # Detect extreme positions
        is_extreme_low = position_24h < self.extreme_threshold
        is_extreme_high = position_24h > (1 - self.extreme_threshold)
        is_extreme = is_extreme_low | is_extreme_high
        
        # Detect volatility acceleration
        is_accelerating = atr_2nd > self.atr_accel_threshold
        
        # Combine conditions: extreme position AND volatility acceleration
        signals = is_extreme & is_accelerating
        
        return signals.values if isinstance(signals, pd.Series) else signals
    
    def _calculate_price_position(self, df):
        """
        Calculate relative price position within lookback window
        
        Returns:
            Series with values 0-1 indicating position in range
            0 = at low, 1 = at high, 0.5 = middle
        """
        high_24h = df['close'].rolling(window=self.lookback_periods, min_periods=1).max()
        low_24h = df['close'].rolling(window=self.lookback_periods, min_periods=1).min()
        
        # Avoid division by zero
        range_24h = high_24h - low_24h
        range_24h = range_24h.replace(0, 1e-10)
        
        position = (df['close'] - low_24h) / range_24h
        
        return position
    
    def _calculate_atr_second_derivative(self, df):
        """
        Calculate 2nd derivative of ATR (volatility acceleration)
        
        If ATR column exists, use it. Otherwise calculate from high/low/close.
        
        Returns:
            Series with ATR 2nd derivative values
        """
        # Use existing ATR if available
        if 'atr' in df.columns:
            atr = df['atr']
        else:
            # Calculate ATR if not present
            atr = self._calculate_atr(df)
        
        # Calculate 1st derivative (rate of change)
        atr_1st = atr.diff()
        
        # Calculate 2nd derivative (acceleration)
        atr_2nd = atr_1st.diff()
        
        return atr_2nd
    
    def _calculate_atr(self, df, period=14):
        """
        Calculate Average True Range (ATR)
        
        Args:
            df: DataFrame with high, low, close columns
            period: ATR period (default: 14)
            
        Returns:
            Series with ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        # True Range is max of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is exponential moving average of TR
        atr = tr.ewm(span=period, adjust=False).mean()
        
        return atr
    
    def get_color(self):
        """Return color for visualization"""
        return "orange"
    
    def get_marker(self):
        """Return marker style for visualization"""
        return "^"  # Triangle up marker


class EnsembleReversalDetection(DetectionMethod):
    """
    Ensemble Reversal Detection using multiple rules with voting
    
    Combines three top-performing rules:
    1. Extreme Position + Volatility Acceleration (F1: 15.3%)
    2. RSI<30 AND BB<0.2 (F1: 15.1%)
    3. Low Position + BB Lower Band (F1: 14.4%)
    
    Parameters:
        min_votes: Minimum votes required for signal (default: 2)
                   2 = high confidence (2 out of 3 rules agree)
                   1 = exploratory (any single rule)
    """
    
    def __init__(self, min_votes=2, **kwargs):
        super().__init__("Ensemble Reversal")
        self.min_votes = min_votes
        
        # Initialize component detectors
        self.detector1 = OptimalReversalDetection(**kwargs)
        self.detector2 = RSIBBReversalDetection(**kwargs)
        self.detector3 = PositionBBReversalDetection(**kwargs)
    
    def detect(self, df):
        """
        Detect reversals using ensemble voting
        
        Args:
            df: DataFrame with required columns for all detectors
            
        Returns:
            numpy boolean array indicating reversal signals
        """
        # Get signals from each detector
        signals1 = self.detector1.detect(df)
        signals2 = self.detector2.detect(df)
        signals3 = self.detector3.detect(df)
        
        # Count votes
        vote_count = signals1.astype(int) + signals2.astype(int) + signals3.astype(int)
        
        # Signal if minimum votes reached
        ensemble_signals = vote_count >= self.min_votes
        
        return ensemble_signals
    
    def get_color(self):
        return "red"
    
    def get_marker(self):
        return "*"  # Star marker for ensemble


class RSIBBReversalDetection(DetectionMethod):
    """
    RSI + Bollinger Band Reversal Detection
    
    Detects oversold conditions using:
    - RSI < 30 (oversold)
    - BB position < 0.2 (near lower band)
    
    F1 Score: 15.1%
    """
    
    def __init__(self, rsi_threshold=30, bb_threshold=0.2, **kwargs):
        super().__init__("RSI+BB Reversal")
        self.rsi_threshold = rsi_threshold
        self.bb_threshold = bb_threshold
    
    def detect(self, df):
        n = len(df)
        
        # Check for required columns
        if 'rsi' not in df.columns or 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
            return np.zeros(n, dtype=bool)
        
        # Calculate BB position
        bb_range = df['bb_upper'] - df['bb_lower']
        bb_range = bb_range.replace(0, 1e-10)
        bb_position = (df['close'] - df['bb_lower']) / bb_range
        
        # Detect oversold conditions
        rsi_oversold = df['rsi'] < self.rsi_threshold
        bb_lower_touch = bb_position < self.bb_threshold
        
        signals = rsi_oversold & bb_lower_touch
        
        return signals.values if isinstance(signals, pd.Series) else signals
    
    def get_color(self):
        return "green"
    
    def get_marker(self):
        return "v"  # Triangle down


class PositionBBReversalDetection(DetectionMethod):
    """
    Position + Bollinger Band Reversal Detection
    
    Detects reversals at:
    - Low price position (< 0.2 of 24h range)
    - Near BB lower band (< 0.2)
    
    F1 Score: 14.4%
    """
    
    def __init__(self, position_threshold=0.2, bb_threshold=0.2, 
                 lookback_hours=24, interval_minutes=15, **kwargs):
        super().__init__("Position+BB Reversal")
        self.position_threshold = position_threshold
        self.bb_threshold = bb_threshold
        self.lookback_periods = int(lookback_hours * 60 / interval_minutes)
    
    def detect(self, df):
        n = len(df)
        
        # Check for required columns
        if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
            return np.zeros(n, dtype=bool)
        
        if n < self.lookback_periods:
            return np.zeros(n, dtype=bool)
        
        # Calculate price position
        high_24h = df['close'].rolling(window=self.lookback_periods, min_periods=1).max()
        low_24h = df['close'].rolling(window=self.lookback_periods, min_periods=1).min()
        range_24h = high_24h - low_24h
        range_24h = range_24h.replace(0, 1e-10)
        position = (df['close'] - low_24h) / range_24h
        
        # Calculate BB position
        bb_range = df['bb_upper'] - df['bb_lower']
        bb_range = bb_range.replace(0, 1e-10)
        bb_position = (df['close'] - df['bb_lower']) / bb_range
        
        # Detect low position + BB lower band
        is_low_position = position < self.position_threshold
        is_bb_lower = bb_position < self.bb_threshold
        
        signals = is_low_position & is_bb_lower
        
        return signals.values if isinstance(signals, pd.Series) else signals
    
    def get_color(self):
        return "blue"
    
    def get_marker(self):
        return "d"  # Diamond
