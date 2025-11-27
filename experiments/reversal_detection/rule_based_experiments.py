#!/usr/bin/env python3
"""
Rule-Based Reversal Detection Experiments

Tests various technical indicator combinations to identify reversal points.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from experiments.reversal_detection.experiment import ReversalExperiment


# ============================================================================
# MACD-Based Detection Rules
# ============================================================================

def macd_histogram_cross(df: pd.DataFrame) -> np.ndarray:
    """
    Detect MACD histogram zero crossings (upward)
    
    Signal when MACD histogram crosses from negative to positive
    """
    if len(df) < 2:
        return np.zeros(len(df), dtype=bool)
    
    hist = df['macd_histogram'].values
    signals = np.zeros(len(df), dtype=bool)
    
    for i in range(1, len(df)):
        if hist[i-1] < 0 and hist[i] > 0:
            signals[i] = True
    
    return signals


def macd_signal_cross(df: pd.DataFrame) -> np.ndarray:
    """
    Detect MACD line crossing above signal line
    """
    if len(df) < 2:
        return np.zeros(len(df), dtype=bool)
    
    macd = df['macd'].values
    signal = df['macd_signal'].values
    signals = np.zeros(len(df), dtype=bool)
    
    for i in range(1, len(df)):
        if macd[i-1] < signal[i-1] and macd[i] > signal[i]:
            signals[i] = True
    
    return signals


def macd_2nd_derivative(df: pd.DataFrame) -> np.ndarray:
    """
    Detect MACD second derivative sign change (acceleration reversal)
    
    Signal when MACD 2nd derivative changes from negative to positive
    """
    if 'macd_2nd' not in df.columns or len(df) < 2:
        return np.zeros(len(df), dtype=bool)
    
    macd_2nd = df['macd_2nd'].values
    signals = np.zeros(len(df), dtype=bool)
    
    for i in range(1, len(df)):
        if macd_2nd[i-1] < 0 and macd_2nd[i] > 0:
            signals[i] = True
    
    return signals


# ============================================================================
# RSI-Based Detection Rules
# ============================================================================

def rsi_oversold_cross(df: pd.DataFrame, threshold: float = 30) -> np.ndarray:
    """
    Detect RSI crossing above oversold threshold
    
    Args:
        threshold: Oversold threshold (default 30)
    """
    if len(df) < 2:
        return np.zeros(len(df), dtype=bool)
    
    rsi = df['rsi'].values
    signals = np.zeros(len(df), dtype=bool)
    
    for i in range(1, len(df)):
        if rsi[i-1] < threshold and rsi[i] >= threshold:
            signals[i] = True
    
    return signals


def rsi_extreme_oversold(df: pd.DataFrame, threshold: float = 20) -> np.ndarray:
    """
    Detect extreme RSI oversold conditions
    
    Signal when RSI is below extreme threshold
    """
    if len(df) < 1:
        return np.zeros(len(df), dtype=bool)
    
    rsi = df['rsi'].values
    return rsi < threshold


def rsi_divergence_simple(df: pd.DataFrame, window: int = 10) -> np.ndarray:
    """
    Simple RSI divergence detection
    
    Signal when price makes lower low but RSI makes higher low
    """
    if len(df) < window + 1:
        return np.zeros(len(df), dtype=bool)
    
    close = df['close'].values
    rsi = df['rsi'].values
    signals = np.zeros(len(df), dtype=bool)
    
    for i in range(window, len(df)):
        # Check if current is local minimum
        window_close = close[i-window:i+1]
        window_rsi = rsi[i-window:i+1]
        
        if close[i] == np.min(window_close):
            # Find previous local minimum
            for j in range(i-1, i-window, -1):
                if close[j] < close[j-1] and close[j] < close[j+1]:
                    # Bullish divergence: price lower but RSI higher
                    if close[i] < close[j] and rsi[i] > rsi[j]:
                        signals[i] = True
                    break
    
    return signals


# ============================================================================
# Bollinger Band Detection Rules
# ============================================================================

def bb_lower_touch(df: pd.DataFrame, threshold: float = 0.1) -> np.ndarray:
    """
    Detect price touching lower Bollinger Band
    
    Args:
        threshold: How close to lower band (0.0 = at band, 0.1 = within 10%)
    """
    if 'bb_position' not in df.columns:
        return np.zeros(len(df), dtype=bool)
    
    bb_pos = df['bb_position'].values
    return bb_pos < threshold


def bb_squeeze_breakout(df: pd.DataFrame, squeeze_threshold: float = 0.02) -> np.ndarray:
    """
    Detect breakout from Bollinger Band squeeze
    
    Signal when bands are tight and price starts moving
    """
    if len(df) < 2:
        return np.zeros(len(df), dtype=bool)
    
    if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
        return np.zeros(len(df), dtype=bool)
    
    bb_width = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    close = df['close'].values
    signals = np.zeros(len(df), dtype=bool)
    
    for i in range(1, len(df)):
        # Squeeze: narrow bands
        if bb_width.iloc[i-1] < squeeze_threshold:
            # Breakout: price movement
            price_change = abs(close[i] - close[i-1]) / close[i-1]
            if price_change > 0.005:  # 0.5% move
                signals[i] = True
    
    return signals


# ============================================================================
# Volume-Based Detection Rules
# ============================================================================

def volume_spike(df: pd.DataFrame, threshold: float = 2.0) -> np.ndarray:
    """
    Detect volume spikes
    
    Args:
        threshold: Multiple of average volume (default 2.0x)
    """
    if 'volume_trend' not in df.columns:
        return np.zeros(len(df), dtype=bool)
    
    vol_trend = df['volume_trend'].values
    return vol_trend > threshold


def volume_trend_reversal(df: pd.DataFrame) -> np.ndarray:
    """
    Detect volume trend reversals
    
    Signal when volume trend changes from decreasing to increasing
    """
    if len(df) < 2 or 'volume_trend' not in df.columns:
        return np.zeros(len(df), dtype=bool)
    
    vol_trend = df['volume_trend'].values
    signals = np.zeros(len(df), dtype=bool)
    
    for i in range(1, len(df)):
        if vol_trend[i-1] < 1.0 and vol_trend[i] >= 1.0:
            signals[i] = True
    
    return signals


# ============================================================================
# SAR-Based Detection Rules
# ============================================================================

def sar_flip(df: pd.DataFrame) -> np.ndarray:
    """
    Detect SAR position flip (from above to below price)
    
    Signal when SAR flips from bearish to bullish
    """
    if len(df) < 2 or 'price_above_sar' not in df.columns:
        return np.zeros(len(df), dtype=bool)
    
    above_sar = df['price_above_sar'].values
    signals = np.zeros(len(df), dtype=bool)
    
    for i in range(1, len(df)):
        if not above_sar[i-1] and above_sar[i]:
            signals[i] = True
    
    return signals


# ============================================================================
# Price Derivative Detection Rules
# ============================================================================

def price_2nd_derivative_reversal(df: pd.DataFrame) -> np.ndarray:
    """
    Detect price second derivative sign change
    
    Signal when price acceleration changes from negative to positive
    """
    if 'price_2nd' not in df.columns or len(df) < 2:
        return np.zeros(len(df), dtype=bool)
    
    price_2nd = df['price_2nd'].values
    signals = np.zeros(len(df), dtype=bool)
    
    for i in range(1, len(df)):
        if price_2nd[i-1] < 0 and price_2nd[i] > 0:
            signals[i] = True
    
    return signals


# ============================================================================
# Multi-Indicator Combinations
# ============================================================================

def macd_rsi_combo(df: pd.DataFrame) -> np.ndarray:
    """
    Combined MACD and RSI signals
    
    Signal when both MACD crosses up AND RSI is oversold
    """
    macd_sig = macd_histogram_cross(df)
    rsi_sig = rsi_oversold_cross(df, threshold=40)
    
    return macd_sig & rsi_sig


def bb_volume_combo(df: pd.DataFrame) -> np.ndarray:
    """
    Combined Bollinger Band and Volume signals
    
    Signal when price touches lower BB AND volume spikes
    """
    bb_sig = bb_lower_touch(df, threshold=0.15)
    vol_sig = volume_spike(df, threshold=1.5)
    
    return bb_sig & vol_sig


def triple_confirmation(df: pd.DataFrame) -> np.ndarray:
    """
    Triple confirmation: MACD + RSI + BB
    
    Signal when all three indicators suggest reversal
    """
    macd_sig = macd_histogram_cross(df)
    rsi_sig = rsi_oversold_cross(df, threshold=35)
    bb_sig = bb_lower_touch(df, threshold=0.2)
    
    return macd_sig & rsi_sig & bb_sig


def momentum_reversal(df: pd.DataFrame) -> np.ndarray:
    """
    Momentum reversal: Price 2nd derivative + MACD 2nd derivative
    
    Signal when both show acceleration reversal
    """
    price_sig = price_2nd_derivative_reversal(df)
    macd_sig = macd_2nd_derivative(df)
    
    return price_sig & macd_sig


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_all_rule_experiments(data_path='analysis_0911_0913 - data_with_reversal_labels.csv'):
    """Run all rule-based detection experiments"""
    
    print("\n" + "="*80)
    print("🔬 Rule-Based Reversal Detection Experiments")
    print("="*80)
    
    # Initialize experiment
    exp = ReversalExperiment(data_path)
    exp.setup()
    
    # Define all methods to test
    methods = {
        # MACD-based
        'MACD Histogram Cross': macd_histogram_cross,
        'MACD Signal Cross': macd_signal_cross,
        'MACD 2nd Derivative': macd_2nd_derivative,
        
        # RSI-based
        'RSI Oversold (30)': rsi_oversold_cross,
        'RSI Extreme Oversold (20)': lambda df: rsi_extreme_oversold(df, 20),
        'RSI Divergence': rsi_divergence_simple,
        
        # Bollinger Band
        'BB Lower Touch': bb_lower_touch,
        'BB Squeeze Breakout': bb_squeeze_breakout,
        
        # Volume
        'Volume Spike': volume_spike,
        'Volume Trend Reversal': volume_trend_reversal,
        
        # SAR
        'SAR Flip': sar_flip,
        
        # Price derivatives
        'Price 2nd Derivative': price_2nd_derivative_reversal,
        
        # Combinations
        'MACD + RSI': macd_rsi_combo,
        'BB + Volume': bb_volume_combo,
        'Triple Confirmation': triple_confirmation,
        'Momentum Reversal': momentum_reversal,
    }
    
    # Run experiments (without look-ahead for speed)
    print(f"\n📊 Testing {len(methods)} rule-based methods...")
    print("(Running without look-ahead enforcement for faster testing)\n")
    
    results = {}
    for name, method in methods.items():
        try:
            metrics = exp.test_detection_method(
                method,
                name,
                ensure_no_lookahead=False  # Faster for initial exploration
            )
            results[name] = metrics
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
    
    # Print comparison
    exp.evaluator.print_comparison(results)
    
    # Find best method
    if results:
        best_method = max(results.items(), key=lambda x: x[1].f1_score)
        print(f"\n🏆 Best Method: {best_method[0]}")
        print(f"   F1-Score: {best_method[1].f1_score:.4f}")
        print(f"   Precision: {best_method[1].precision:.2%}")
        print(f"   Recall: {best_method[1].recall:.2%}")
        print(f"   Detection Rate: {best_method[1].detection_rate_on_reversals:.2%}")
        
        # Analyze missed reversals for best method
        best_fn = methods[best_method[0]]
        predictions = best_fn(exp.df)
        exp.analyze_missed_reversals(predictions, best_method[0])
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Rule-based reversal detection experiments')
    parser.add_argument('--data', type=str, 
                        default='analysis_0911_0913 - data_with_reversal_labels.csv',
                        help='Path to training data CSV')
    
    args = parser.parse_args()
    
    results = run_all_rule_experiments(args.data)
