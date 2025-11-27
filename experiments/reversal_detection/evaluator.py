#!/usr/bin/env python3
"""
Evaluation Metrics for Reversal Detection

Calculates precision, recall, F1-score, and other metrics for detection methods.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    precision: float
    recall: float
    f1_score: float
    
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    
    detection_rate_on_reversals: float
    
    method_name: str
    parameters: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """Pretty print metrics"""
        return (
            f"\n{'='*60}\n"
            f"📊 Evaluation Metrics: {self.method_name}\n"
            f"{'='*60}\n"
            f"Precision:  {self.precision:.2%} ({self.true_positives}/{self.true_positives + self.false_positives})\n"
            f"Recall:     {self.recall:.2%} ({self.true_positives}/{self.true_positives + self.false_negatives})\n"
            f"F1-Score:   {self.f1_score:.4f}\n"
            f"\n"
            f"Detection Rate on Reversals: {self.detection_rate_on_reversals:.2%}\n"
            f"\n"
            f"Confusion Matrix:\n"
            f"  TP: {self.true_positives:4d}  |  FP: {self.false_positives:4d}\n"
            f"  FN: {self.false_negatives:4d}  |  TN: {self.true_negatives:4d}\n"
            f"{'='*60}\n"
        )


class ReversalEvaluator:
    """
    Evaluates reversal detection methods against ground truth
    """
    
    def __init__(self):
        """Initialize evaluator"""
        pass
    
    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        method_name: str = "Unknown",
        parameters: Optional[Dict[str, Any]] = None
    ) -> EvaluationMetrics:
        """
        Calculate evaluation metrics
        
        Args:
            predictions: Boolean array of predicted reversals
            ground_truth: Boolean array of actual reversals
            method_name: Name of the detection method
            parameters: Method parameters (for logging)
            
        Returns:
            EvaluationMetrics object with all calculated metrics
        """
        # Ensure boolean arrays
        predictions = np.asarray(predictions, dtype=bool)
        ground_truth = np.asarray(ground_truth, dtype=bool)
        
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Predictions and ground truth must have same length. "
                f"Got {len(predictions)} vs {len(ground_truth)}"
            )
        
        # Calculate confusion matrix
        tp = np.sum(predictions & ground_truth)
        fp = np.sum(predictions & ~ground_truth)
        fn = np.sum(~predictions & ground_truth)
        tn = np.sum(~predictions & ~ground_truth)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        
        # Detection rate on actual reversals (same as recall)
        detection_rate = recall
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=int(tp),
            false_positives=int(fp),
            false_negatives=int(fn),
            true_negatives=int(tn),
            detection_rate_on_reversals=detection_rate,
            method_name=method_name,
            parameters=parameters
        )
    
    def compare_methods(
        self,
        results: Dict[str, np.ndarray],
        ground_truth: np.ndarray
    ) -> Dict[str, EvaluationMetrics]:
        """
        Compare multiple detection methods
        
        Args:
            results: Dictionary mapping method names to prediction arrays
            ground_truth: Boolean array of actual reversals
            
        Returns:
            Dictionary mapping method names to their metrics
        """
        comparison = {}
        
        for method_name, predictions in results.items():
            metrics = self.evaluate(predictions, ground_truth, method_name)
            comparison[method_name] = metrics
        
        return comparison
    
    def print_comparison(
        self,
        comparison: Dict[str, EvaluationMetrics]
    ):
        """
        Print comparison table of multiple methods
        
        Args:
            comparison: Dictionary of method names to metrics
        """
        print(f"\n{'='*80}")
        print(f"📊 Method Comparison")
        print(f"{'='*80}")
        print(f"{'Method':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Det.Rate':>10}")
        print(f"{'-'*80}")
        
        # Sort by F1-score descending
        sorted_methods = sorted(
            comparison.items(),
            key=lambda x: x[1].f1_score,
            reverse=True
        )
        
        for method_name, metrics in sorted_methods:
            print(
                f"{method_name:<30} "
                f"{metrics.precision:>9.2%} "
                f"{metrics.recall:>9.2%} "
                f"{metrics.f1_score:>10.4f} "
                f"{metrics.detection_rate_on_reversals:>9.2%}"
            )
        
        print(f"{'='*80}\n")
    
    def analyze_missed_reversals(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze which reversal points were missed
        
        Args:
            predictions: Boolean array of predicted reversals
            ground_truth: Boolean array of actual reversals
            timestamps: Optional array of timestamps
            
        Returns:
            Dictionary with analysis of missed reversals
        """
        predictions = np.asarray(predictions, dtype=bool)
        ground_truth = np.asarray(ground_truth, dtype=bool)
        
        # Find missed reversals (false negatives)
        missed_indices = np.where(ground_truth & ~predictions)[0]
        
        # Find false positives
        false_positive_indices = np.where(predictions & ~ground_truth)[0]
        
        analysis = {
            'missed_count': len(missed_indices),
            'missed_indices': missed_indices.tolist(),
            'false_positive_count': len(false_positive_indices),
            'false_positive_indices': false_positive_indices.tolist(),
            'total_reversals': np.sum(ground_truth),
            'missed_percentage': len(missed_indices) / np.sum(ground_truth) * 100 if np.sum(ground_truth) > 0 else 0
        }
        
        if timestamps is not None:
            analysis['missed_timestamps'] = timestamps[missed_indices].tolist()
            analysis['false_positive_timestamps'] = timestamps[false_positive_indices].tolist()
        
        return analysis
