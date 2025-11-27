#!/usr/bin/env python3
"""
Example Usage of Optimal Reversal Detection

This script demonstrates how to use the OptimalReversalDetection class
and other reversal detection methods in the production system.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from detection import (
    OptimalReversalDetection,
    EnsembleReversalDetection,
    RSIBBReversalDetection,
    PositionBBReversalDetection
)


def load_sample_data():
    """Load sample data for demonstration"""
    # Try to load training data
    data_path = 'training_data/reversal_training_data_2022-01-01_to_2025-09-01_15m.csv'
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} data points from {data_path}")
        return df
    else:
        print(f"❌ Data file not found: {data_path}")
        print("   Please run generate_training_data.py first")
        return None


def example_basic_usage():
    """Example 1: Basic usage of OptimalReversalDetection"""
    print("\n" + "=" * 80)
    print("Example 1: Basic Usage of OptimalReversalDetection")
    print("=" * 80)
    
    # Load data
    df = load_sample_data()
    if df is None:
        return
    
    # Create detector with default parameters
    detector = OptimalReversalDetection()
    
    # Detect reversals
    signals = detector.detect(df)
    
    # Print results
    signal_count = signals.sum()
    signal_rate = signal_count / len(df) * 100
    
    print(f"\n📊 Detection Results:")
    print(f"   Total data points: {len(df)}")
    print(f"   Signals detected: {signal_count}")
    print(f"   Signal rate: {signal_rate:.2f}%")
    
    # Show some signal examples
    if signal_count > 0:
        signal_indices = np.where(signals)[0]
        print(f"\n🎯 First 5 signals:")
        for i, idx in enumerate(signal_indices[:5]):
            row = df.iloc[idx]
            print(f"   {i+1}. Index {idx}: {row['timestamp']} - Close: ${row['close']:.2f}")


def example_custom_parameters():
    """Example 2: Using custom parameters"""
    print("\n" + "=" * 80)
    print("Example 2: Custom Parameters")
    print("=" * 80)
    
    df = load_sample_data()
    if df is None:
        return
    
    # Create detector with custom parameters
    # More conservative: stricter extreme threshold and higher acceleration
    detector = OptimalReversalDetection(
        extreme_threshold=0.15,      # More extreme positions (15% instead of 20%)
        atr_accel_threshold=0.7,     # Higher volatility acceleration required
        lookback_hours=48,           # Longer lookback period
        interval_minutes=15
    )
    
    signals = detector.detect(df)
    
    print(f"\n📊 Conservative Detection Results:")
    print(f"   Signals detected: {signals.sum()}")
    print(f"   Signal rate: {signals.sum() / len(df) * 100:.2f}%")
    print(f"\n   Parameters:")
    print(f"   - Extreme threshold: {detector.extreme_threshold}")
    print(f"   - ATR acceleration threshold: {detector.atr_accel_threshold}")
    print(f"   - Lookback hours: {detector.lookback_hours}")


def example_ensemble_detection():
    """Example 3: Using ensemble detection with voting"""
    print("\n" + "=" * 80)
    print("Example 3: Ensemble Detection with Voting")
    print("=" * 80)
    
    df = load_sample_data()
    if df is None:
        return
    
    # High confidence: require 2 out of 3 rules
    ensemble_high = EnsembleReversalDetection(min_votes=2)
    signals_high = ensemble_high.detect(df)
    
    # Exploratory: any single rule
    ensemble_low = EnsembleReversalDetection(min_votes=1)
    signals_low = ensemble_low.detect(df)
    
    print(f"\n📊 Ensemble Detection Results:")
    print(f"\n   High Confidence (2/3 votes):")
    print(f"   - Signals: {signals_high.sum()}")
    print(f"   - Rate: {signals_high.sum() / len(df) * 100:.2f}%")
    
    print(f"\n   Exploratory (1/3 votes):")
    print(f"   - Signals: {signals_low.sum()}")
    print(f"   - Rate: {signals_low.sum() / len(df) * 100:.2f}%")
    
    print(f"\n   💡 Use high confidence for main positions")
    print(f"      Use exploratory for smaller test positions")


def example_individual_rules():
    """Example 4: Using individual detection rules"""
    print("\n" + "=" * 80)
    print("Example 4: Individual Detection Rules")
    print("=" * 80)
    
    df = load_sample_data()
    if df is None:
        return
    
    # Create individual detectors
    detector1 = OptimalReversalDetection()
    detector2 = RSIBBReversalDetection()
    detector3 = PositionBBReversalDetection()
    
    # Detect with each
    signals1 = detector1.detect(df)
    signals2 = detector2.detect(df)
    signals3 = detector3.detect(df)
    
    print(f"\n📊 Individual Rule Results:")
    print(f"\n   1. Extreme Position + Volatility Acceleration:")
    print(f"      Signals: {signals1.sum()} ({signals1.sum() / len(df) * 100:.2f}%)")
    
    print(f"\n   2. RSI<30 AND BB<0.2:")
    print(f"      Signals: {signals2.sum()} ({signals2.sum() / len(df) * 100:.2f}%)")
    
    print(f"\n   3. Low Position + BB Lower Band:")
    print(f"      Signals: {signals3.sum()} ({signals3.sum() / len(df) * 100:.2f}%)")
    
    # Find agreement
    all_agree = signals1 & signals2 & signals3
    any_agree = signals1 | signals2 | signals3
    
    print(f"\n   Agreement Analysis:")
    print(f"   - All 3 agree: {all_agree.sum()} signals")
    print(f"   - Any 1 agrees: {any_agree.sum()} signals")


def example_visualization():
    """Example 5: Visualizing detection results"""
    print("\n" + "=" * 80)
    print("Example 5: Visualization")
    print("=" * 80)
    
    df = load_sample_data()
    if df is None:
        return
    
    # Use recent data for better visualization
    df_recent = df.tail(1000).copy()
    df_recent.reset_index(drop=True, inplace=True)
    
    # Detect with optimal method
    detector = OptimalReversalDetection()
    signals = detector.detect(df_recent)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot 1: Price with signals
    ax1.plot(df_recent.index, df_recent['close'], label='Close Price', linewidth=1)
    
    # Mark signals
    signal_indices = np.where(signals)[0]
    if len(signal_indices) > 0:
        ax1.scatter(signal_indices, df_recent.loc[signal_indices, 'close'],
                   color=detector.get_color(), marker=detector.get_marker(),
                   s=100, label='Reversal Signals', zorder=5)
    
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Optimal Reversal Detection - Recent Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Position and ATR 2nd derivative
    # Calculate position
    lookback = 96  # 24 hours
    high_24h = df_recent['close'].rolling(window=lookback, min_periods=1).max()
    low_24h = df_recent['close'].rolling(window=lookback, min_periods=1).min()
    position = (df_recent['close'] - low_24h) / (high_24h - low_24h + 1e-10)
    
    # Calculate ATR 2nd derivative
    if 'atr' in df_recent.columns:
        atr_2nd = df_recent['atr'].diff().diff()
    else:
        atr_2nd = pd.Series(np.zeros(len(df_recent)))
    
    ax2_twin = ax2.twinx()
    
    ax2.plot(df_recent.index, position, label='Price Position (24h)', 
            color='blue', alpha=0.7)
    ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Extreme Thresholds')
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    
    ax2_twin.plot(df_recent.index, atr_2nd, label='ATR 2nd Derivative',
                 color='green', alpha=0.7)
    ax2_twin.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5,
                    label='Acceleration Threshold')
    
    ax2.set_xlabel('Data Point Index')
    ax2.set_ylabel('Price Position', color='blue')
    ax2_twin.set_ylabel('ATR 2nd Derivative', color='green')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'experiments/reversal_detection/example_usage_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")


def main():
    """Run all examples"""
    print("=" * 80)
    print("🚀 Optimal Reversal Detection - Usage Examples")
    print("=" * 80)
    
    # Run examples
    example_basic_usage()
    example_custom_parameters()
    example_ensemble_detection()
    example_individual_rules()
    example_visualization()
    
    print("\n" + "=" * 80)
    print("✅ All Examples Complete!")
    print("=" * 80)
    print("\n💡 Key Takeaways:")
    print("   1. OptimalReversalDetection is the best single method (F1=15.3%)")
    print("   2. Use EnsembleReversalDetection for higher confidence")
    print("   3. Adjust parameters based on your risk tolerance")
    print("   4. Always use with proper risk management (stop-losses, position sizing)")
    print("\n📚 See METHOD_SELECTION.md for detailed analysis and rationale")


if __name__ == '__main__':
    main()
