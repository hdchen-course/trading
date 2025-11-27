#!/usr/bin/env python3
"""
Simple test to verify the experimental framework works correctly
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from experiments.reversal_detection.data_loader import ReversalDataLoader
from experiments.reversal_detection.evaluator import ReversalEvaluator
from experiments.reversal_detection.experiment import ReversalExperiment


def test_data_loader():
    """Test data loading functionality"""
    print("\n" + "="*60)
    print("Test 1: Data Loader")
    print("="*60)
    
    try:
        loader = ReversalDataLoader('analysis_0911_0913 - data_with_reversal_labels.csv')
        df = loader.load_data()
        
        print(f"✅ Data loaded: {len(df)} rows")
        print(f"✅ Columns: {len(df.columns)}")
        print(f"✅ Ground truth extracted: {loader.ground_truth.sum()} reversals")
        
        # Test no-lookahead
        data_at_10 = loader.get_data_at_time(10)
        assert len(data_at_10) == 11, "Should have 11 rows (0-10)"
        print(f"✅ No look-ahead works: get_data_at_time(10) returns {len(data_at_10)} rows")
        
        # Test split
        train, val, test = loader.split_time_series()
        print(f"✅ Time-series split: train={len(train)}, val={len(val)}, test={len(test)}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluator():
    """Test evaluation metrics calculation"""
    print("\n" + "="*60)
    print("Test 2: Evaluator")
    print("="*60)
    
    try:
        evaluator = ReversalEvaluator()
        
        # Create test data
        ground_truth = np.array([False, True, False, True, False, True, False, False])
        predictions = np.array([False, True, False, False, True, True, False, False])
        
        metrics = evaluator.evaluate(predictions, ground_truth, "Test Method")
        
        print(f"✅ Metrics calculated:")
        print(f"   Precision: {metrics.precision:.2%}")
        print(f"   Recall: {metrics.recall:.2%}")
        print(f"   F1-Score: {metrics.f1_score:.4f}")
        print(f"   TP={metrics.true_positives}, FP={metrics.false_positives}")
        print(f"   FN={metrics.false_negatives}, TN={metrics.true_negatives}")
        
        # Verify calculations
        assert metrics.true_positives == 2, "Should have 2 TPs"
        assert metrics.false_positives == 1, "Should have 1 FP"
        assert metrics.false_negatives == 1, "Should have 1 FN"
        print(f"✅ Metric calculations verified")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_experiment_framework():
    """Test the main experiment framework"""
    print("\n" + "="*60)
    print("Test 3: Experiment Framework")
    print("="*60)
    
    try:
        exp = ReversalExperiment('analysis_0911_0913 - data_with_reversal_labels.csv')
        exp.setup()
        
        print(f"✅ Experiment initialized")
        
        # Test a simple detector
        def always_false_detector(df):
            """Detector that never signals"""
            return np.zeros(len(df), dtype=bool)
        
        def always_true_detector(df):
            """Detector that always signals"""
            return np.ones(len(df), dtype=bool)
        
        # Test without look-ahead enforcement (faster for testing)
        print("\n📊 Testing simple detectors...")
        metrics1 = exp.test_detection_method(
            always_false_detector, 
            "Always False",
            ensure_no_lookahead=False
        )
        
        assert metrics1.recall == 0.0, "Always false should have 0% recall"
        print(f"✅ Always False detector: recall={metrics1.recall:.2%} (correct)")
        
        metrics2 = exp.test_detection_method(
            always_true_detector,
            "Always True", 
            ensure_no_lookahead=False
        )
        
        assert metrics2.recall == 1.0, "Always true should have 100% recall"
        print(f"✅ Always True detector: recall={metrics2.recall:.2%} (correct)")
        
        # Test comparison
        print("\n📊 Testing method comparison...")
        methods = {
            'Always False': always_false_detector,
            'Always True': always_true_detector,
        }
        results = exp.compare_methods(methods, ensure_no_lookahead=False)
        
        assert len(results) == 2, "Should have 2 results"
        print(f"✅ Method comparison works")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 Testing Reversal Detection Framework")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Data Loader", test_data_loader()))
    results.append(("Evaluator", test_evaluator()))
    results.append(("Experiment Framework", test_experiment_framework()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Framework is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
