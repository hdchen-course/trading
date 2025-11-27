#!/usr/bin/env python3
"""
Reversal Detection Experiment Framework

Main experiment runner for testing different detection approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional
from .data_loader import ReversalDataLoader
from .evaluator import ReversalEvaluator, EvaluationMetrics


class ReversalExperiment:
    """
    Framework for testing reversal detection approaches
    
    Provides utilities for:
    - Loading and managing labeled data
    - Testing rule-based and ML-based approaches
    - Evaluating and comparing methods
    - Ensuring no look-ahead bias
    """
    
    def __init__(self, data_path: str):
        """
        Initialize experiment framework
        
        Args:
            data_path: Path to CSV file with labeled reversal data
        """
        self.data_loader = ReversalDataLoader(data_path)
        self.evaluator = ReversalEvaluator()
        
        self.df = None
        self.ground_truth = None
        
    def setup(self):
        """Load data and prepare for experiments"""
        print("🔬 Setting up reversal detection experiment...")
        self.df = self.data_loader.load_data()
        self.ground_truth = self.data_loader.get_ground_truth()
        print("✅ Setup complete!\n")
    
    def test_detection_method(
        self,
        detect_fn: Callable[[pd.DataFrame], np.ndarray],
        method_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        ensure_no_lookahead: bool = True
    ) -> EvaluationMetrics:
        """
        Test a detection method and evaluate its performance
        
        Args:
            detect_fn: Function that takes DataFrame and returns boolean array
            method_name: Name of the detection method
            parameters: Method parameters (for logging)
            ensure_no_lookahead: If True, simulate real-time conditions
            
        Returns:
            EvaluationMetrics with results
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call setup() first.")
        
        print(f"\n🧪 Testing: {method_name}")
        if parameters:
            print(f"   Parameters: {parameters}")
        
        if ensure_no_lookahead:
            # Simulate real-time detection (no look-ahead)
            predictions = self._detect_with_no_lookahead(detect_fn)
        else:
            # Allow full dataset access (for analysis-only methods)
            predictions = detect_fn(self.df)
        
        # Evaluate
        metrics = self.evaluator.evaluate(
            predictions,
            self.ground_truth,
            method_name,
            parameters
        )
        
        print(metrics)
        
        return metrics
    
    def _detect_with_no_lookahead(
        self,
        detect_fn: Callable[[pd.DataFrame], np.ndarray]
    ) -> np.ndarray:
        """
        Run detection ensuring no look-ahead bias
        
        For each time point, only use data available up to that point.
        
        Args:
            detect_fn: Detection function
            
        Returns:
            Boolean array of predictions
        """
        n = len(self.df)
        predictions = np.zeros(n, dtype=bool)
        
        # For efficiency, we can use a sliding window approach
        # But for now, let's use the simple approach
        for i in range(n):
            # Get data available up to current time
            available_data = self.data_loader.get_data_at_time(i)
            
            # Run detection on available data
            result = detect_fn(available_data)
            
            # Take only the last prediction (for current time point)
            if len(result) > 0:
                predictions[i] = result[-1]
        
        return predictions
    
    def compare_methods(
        self,
        methods: Dict[str, Callable[[pd.DataFrame], np.ndarray]],
        ensure_no_lookahead: bool = True
    ) -> Dict[str, EvaluationMetrics]:
        """
        Compare multiple detection methods
        
        Args:
            methods: Dictionary mapping method names to detection functions
            ensure_no_lookahead: If True, simulate real-time conditions
            
        Returns:
            Dictionary mapping method names to their metrics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call setup() first.")
        
        print(f"\n🔬 Comparing {len(methods)} detection methods...")
        
        results = {}
        for method_name, detect_fn in methods.items():
            metrics = self.test_detection_method(
                detect_fn,
                method_name,
                ensure_no_lookahead=ensure_no_lookahead
            )
            results[method_name] = metrics
        
        # Print comparison table
        self.evaluator.print_comparison(results)
        
        return results
    
    def analyze_missed_reversals(
        self,
        predictions: np.ndarray,
        method_name: str = "Method"
    ):
        """
        Analyze which reversal points were missed
        
        Args:
            predictions: Boolean array of predictions
            method_name: Name of the method (for display)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call setup() first.")
        
        timestamps = self.df['timestamp'].values if 'timestamp' in self.df.columns else None
        
        analysis = self.evaluator.analyze_missed_reversals(
            predictions,
            self.ground_truth,
            timestamps
        )
        
        print(f"\n🔍 Missed Reversal Analysis: {method_name}")
        print(f"{'='*60}")
        print(f"Total reversals:     {analysis['total_reversals']}")
        print(f"Missed reversals:    {analysis['missed_count']} ({analysis['missed_percentage']:.1f}%)")
        print(f"False positives:     {analysis['false_positive_count']}")
        
        if timestamps is not None and analysis['missed_count'] > 0:
            print(f"\nMissed reversal timestamps:")
            for ts in analysis['missed_timestamps'][:10]:  # Show first 10
                print(f"  - {ts}")
            if analysis['missed_count'] > 10:
                print(f"  ... and {analysis['missed_count'] - 10} more")
        
        print(f"{'='*60}\n")
        
        return analysis
    
    def get_train_val_test_split(self):
        """
        Get train/val/test split of the data
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        return self.data_loader.split_time_series()
    
    def get_available_indicators(self):
        """Get list of available technical indicators"""
        return self.data_loader.get_available_indicators()
