# Experiments 目錄

這個目錄包含所有實驗性代碼、分析腳本和測試文件。

## 目錄結構

### analysis/ - 分析腳本
數據分析、檢查、比較和指標分析相關腳本

### data_preparation/ - 數據準備
創建數據集、收集數據、獲取數據和生成訓練數據

### training/ - 模型訓練
訓練和重訓練機器學習模型的腳本

### testing/ - 測試腳本
測試模型、評估性能的腳本

### optimization/ - 優化腳本
參數優化、調參、過濾和查找最佳配置

### visualization/ - 可視化
繪圖和可視化分析結果

### detection/ - 檢測方法
各種檢測方法和監督學習實驗

### prediction/ - 預測模型
ML預測、規則預測和價格預測實驗

### utilities/ - 工具腳本
修復、調試和簡單測試工具

### specific/ - 特定分析
針對特定時期或特定指標的分析

## 統計

- 總共移動: 104 個文件
- 保留核心文件: 18 個

## 移動的文件


### analysis/ (22 個文件)
- analyze_atr_ratio.py
- analyze_crash_signals.py
- analyze_drawdown.py
- analyze_fixed_amplitude.py
- analyze_missed_high.py
- analyze_missed_low.py
- analyze_missed_profits.py
- analyze_missed_reversals.py
- analyze_pre_crash.py
- analyze_profitable_patterns.py
- analyze_reversals.py
- analyze_reversals_simple.py
- analyze_support_resistance.py
- analyze_sustained_crashes.py
- check_1h_changes.py
- check_columns.py
- check_crash_timing.py
- check_high_area.py
- check_missing_points.py
- compare_rsi_stochrsi.py
- compare_rsi_stochrsi_crash.py
- indicator_analysis.py

### data_preparation/ (11 個文件)
- collect_bb_patterns.py
- collect_specific_patterns.py
- create_4h_1h_crash_dataset.py
- create_8h_1h_crash_dataset.py
- create_balanced_4h_1h_model.py
- create_balanced_8h_1h_model.py
- create_balanced_crash_model.py
- create_crash_prediction_dataset.py
- fetch_12months_data.py
- fetch_btc_data_2022_2025.py
- generate_training_data.py

### detection/ (2 個文件)
- detect_crash_start.py
- supervised_crash_detection.py

### optimization/ (10 個文件)
- filter_best_results.py
- filter_low_drawdown.py
- filter_strict_criteria.py
- find_more_trend_patterns.py
- optimize_enhanced_bollinger.py
- optimize_sideways_strategy.py
- optimize_sideways_v2.py
- optimize_sideways_with_drawdown.py
- tune_lstm_logged.py
- tune_with_progress.py

### prediction/ (7 個文件)
- ml_pattern_prediction.py
- ml_spike_prediction.py
- ml_spike_prediction_v2.py
- ml_spike_prediction_v3.py
- price_spike_detection.py
- rule_based_prediction.py
- ultra_precise_prediction.py

### specific/ (4 個文件)
- october_crash_detection.py
- reversal_pattern_analysis.py
- stochrsi_analysis.py
- trend_bollinger_features.py

### testing/ (14 個文件)
- evaluate_accuracy.py
- test_4h_1h_prediction.py
- test_8h_1h_prediction.py
- test_bb_params.py
- test_bb_params_full.py
- test_best_model.py
- test_crash_prediction.py
- test_debug_trailing.py
- test_ml_reversal.py
- test_precise_model.py
- test_selected_features.py
- test_short_sequences.py
- test_simple_conditions.py
- test_wider_range.py

### training/ (15 個文件)
- retrain_bb_model.py
- retrain_bb_model_30_15.py
- retrain_bb_model_40_15.py
- retrain_bb_simple.py
- retrain_optimized_strategy.py
- retrain_original_strategy.py
- train_70percent_model.py
- train_bb_stochrsi_model.py
- train_combined_crash_model.py
- train_comprehensive_talib.py
- train_crash_detection_filtered.py
- train_reversal_model.py
- train_reversal_model_v2.py
- train_separate_models.py
- train_trend_model.py

### utilities/ (4 個文件)
- fix_trailing_tests.py
- simple_test.py
- simple_tune.py
- use_bb_model.py

### visualization/ (15 個文件)
- plot_specific_period.py
- visualize_colored_segments.py
- visualize_crash_detection.py
- visualize_filtered_crash_samples.py
- visualize_price_only.py
- visualize_reversals.py
- visualize_reversals_adaptive.py
- visualize_reversals_balanced.py
- visualize_reversals_enhanced.py
- visualize_reversals_final.py
- visualize_reversals_improved.py
- visualize_trend_quarterly.py
- visualize_trend_zones.py
- visualize_trend_zones_v2.py
- visualize_trend_zones_v3.py
