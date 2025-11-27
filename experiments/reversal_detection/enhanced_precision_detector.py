#!/usr/bin/env python3
"""
Enhanced Precision Reversal Detector

整合多種策略來提高反轉檢測的精確度：
1. 基礎規則：極端位置 + 波動加速（F1=15.3%）
2. 成交量確認：避免低量假信號
3. 時間窗口過濾：避免短時間重複信號
4. 背離強度量化：只在強背離時發出信號
5. 多時間框架確認：結合不同週期的信號
6. 動態閾值調整：根據市場狀態調整參數

目標：將精確度從 10.6% 提升到 15-20%
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from detection.base import DetectionMethod
from detection.optimal_reversal import OptimalReversalDetection


class EnhancedPrecisionDetector(DetectionMethod):
    """
    增強精確度檢測器
    
    在最佳規則基礎上增加多層過濾，提高精確度
    
    Parameters:
        base_extreme_threshold: 基礎極端位置閾值 (default: 0.2)
        base_atr_accel_threshold: 基礎 ATR 加速閾值 (default: 0.5)
        volume_confirmation_threshold: 成交量確認閾值 (default: 1.2)
        cooldown_periods: 信號冷卻期（避免重複信號）(default: 8)
        divergence_strength_threshold: 背離強度閾值 (default: 0.5)
        use_dynamic_thresholds: 是否使用動態閾值 (default: True)
        lookback_hours: 回看時間（小時）(default: 24)
        interval_minutes: 數據間隔（分鐘）(default: 15)
    """
    
    def __init__(
        self,
        base_extreme_threshold=0.2,
        base_atr_accel_threshold=0.5,
        volume_confirmation_threshold=1.2,
        cooldown_periods=8,
        divergence_strength_threshold=0.5,
        use_dynamic_thresholds=True,
        lookback_hours=24,
        interval_minutes=15,
        name="EnhancedPrecision"
    ):
        super().__init__(name)
        
        # 基礎參數
        self.base_extreme_threshold = base_extreme_threshold
        self.base_atr_accel_threshold = base_atr_accel_threshold
        self.lookback_hours = lookback_hours
        self.interval_minutes = interval_minutes
        self.lookback_periods = int(lookback_hours * 60 / interval_minutes)
        
        # 增強參數
        self.volume_confirmation_threshold = volume_confirmation_threshold
        self.cooldown_periods = cooldown_periods
        self.divergence_strength_threshold = divergence_strength_threshold
        self.use_dynamic_thresholds = use_dynamic_thresholds
        
        # 基礎檢測器
        self.base_detector = OptimalReversalDetection(
            extreme_threshold=base_extreme_threshold,
            atr_accel_threshold=base_atr_accel_threshold,
            lookback_hours=lookback_hours,
            interval_minutes=interval_minutes
        )
    
    def detect(self, df):
        """
        檢測反轉點（增強版）
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            
        Returns:
            Boolean array indicating reversal signals
        """
        n = len(df)
        
        if n < self.lookback_periods:
            return np.zeros(n, dtype=bool)
        
        # 步驟 1: 獲取基礎信號
        base_signals = self.base_detector.detect(df)
        
        # 步驟 2: 成交量確認
        volume_confirmed = self._volume_confirmation(df, base_signals)
        
        # 步驟 3: 背離強度過濾
        divergence_filtered = self._divergence_strength_filter(df, volume_confirmed)
        
        # 步驟 4: 時間窗口過濾（避免重複信號）
        time_filtered = self._apply_cooldown(divergence_filtered)
        
        # 步驟 5: 動態閾值調整（可選）
        if self.use_dynamic_thresholds:
            final_signals = self._dynamic_threshold_adjustment(df, time_filtered)
        else:
            final_signals = time_filtered
        
        return final_signals
    
    def _volume_confirmation(self, df, signals):
        """
        成交量確認：只在成交量放大時確認信號
        
        Args:
            df: DataFrame with volume data
            signals: Base signals to confirm
            
        Returns:
            Boolean array with volume-confirmed signals
        """
        if 'volume_trend' not in df.columns:
            # 如果沒有 volume_trend，計算它
            volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()
            volume_trend = df['volume'] / volume_ma
        else:
            volume_trend = df['volume_trend']
        
        # 只在成交量高於閾值時確認信號
        volume_confirmed = signals & (volume_trend > self.volume_confirmation_threshold)
        
        return volume_confirmed
    
    def _divergence_strength_filter(self, df, signals):
        """
        背離強度過濾：量化背離強度，只在強背離時發出信號
        
        Args:
            df: DataFrame with price and volume data
            signals: Signals to filter
            
        Returns:
            Boolean array with strong divergence signals only
        """
        # 計算價格和成交量的變化
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change()
        
        # 背離強度 = |價格變化| * |成交量變化|
        # 當價格下跌但成交量上升時，這是強烈的反轉信號
        divergence_strength = abs(price_change) * abs(volume_change)
        
        # 檢測背離方向
        # 價跌量增（看漲反轉）或 價漲量縮（看跌反轉）
        is_bullish_divergence = (price_change < 0) & (volume_change > 0)
        is_bearish_divergence = (price_change > 0) & (volume_change < 0)
        has_divergence = is_bullish_divergence | is_bearish_divergence
        
        # 只在有背離且強度足夠時確認信號
        strong_divergence = (
            has_divergence & 
            (divergence_strength > self.divergence_strength_threshold)
        )
        
        filtered_signals = signals & strong_divergence
        
        return filtered_signals
    
    def _apply_cooldown(self, signals):
        """
        應用冷卻期：避免在短時間內重複發出信號
        
        Args:
            signals: Boolean array of signals
            
        Returns:
            Boolean array with cooldown applied
        """
        n = len(signals)
        filtered_signals = np.zeros(n, dtype=bool)
        
        last_signal_idx = -self.cooldown_periods - 1
        
        for i in range(n):
            if signals[i]:
                # 檢查是否在冷卻期內
                if i - last_signal_idx > self.cooldown_periods:
                    filtered_signals[i] = True
                    last_signal_idx = i
        
        return filtered_signals
    
    def _dynamic_threshold_adjustment(self, df, signals):
        """
        動態閾值調整：根據市場波動狀態調整檢測靈敏度
        
        在高波動市場中提高閾值（更保守）
        在低波動市場中降低閾值（更積極）
        
        Args:
            df: DataFrame with volatility data
            signals: Signals to adjust
            
        Returns:
            Boolean array with adjusted signals
        """
        # 計算市場波動狀態
        if 'volatility' in df.columns:
            volatility = df['volatility']
        elif 'atr' in df.columns:
            # 使用 ATR 作為波動度量
            volatility = df['atr']
        else:
            # 計算簡單波動度
            volatility = df['close'].rolling(window=20).std()
        
        # 計算波動的分位數
        high_vol_threshold = volatility.quantile(0.75)
        low_vol_threshold = volatility.quantile(0.25)
        
        # 在高波動期間，只保留最強的信號
        # 這裡我們重新運行基礎檢測器，但使用更嚴格的參數
        adjusted_signals = signals.copy()
        
        for i in range(len(df)):
            if signals[i]:
                current_vol = volatility.iloc[i]
                
                # 高波動：需要更強的確認
                if current_vol > high_vol_threshold:
                    # 檢查是否有額外的確認信號
                    # 例如：RSI 是否在極端區域
                    if 'rsi' in df.columns:
                        rsi = df['rsi'].iloc[i]
                        # 只在 RSI 極端時保留信號
                        if not (rsi < 25 or rsi > 75):
                            adjusted_signals[i] = False
                
                # 低波動：可以稍微放寬
                # （但我們已經通過了其他過濾器，所以保持信號）
        
        return adjusted_signals
    
    def get_color(self):
        return "purple"
    
    def get_marker(self):
        return "D"  # Diamond marker


class MultiTimeframeDetector(DetectionMethod):
    """
    多時間框架檢測器
    
    結合不同時間週期的信號來提高精確度
    只在多個時間框架都確認時才發出信號
    
    Parameters:
        primary_lookback: 主要時間框架回看期（小時）(default: 24)
        secondary_lookback: 次要時間框架回看期（小時）(default: 48)
        tertiary_lookback: 第三時間框架回看期（小時）(default: 12)
        min_confirmations: 最少需要幾個時間框架確認 (default: 2)
    """
    
    def __init__(
        self,
        primary_lookback=24,
        secondary_lookback=48,
        tertiary_lookback=12,
        min_confirmations=2,
        interval_minutes=15,
        name="MultiTimeframe"
    ):
        super().__init__(name)
        
        self.primary_lookback = primary_lookback
        self.secondary_lookback = secondary_lookback
        self.tertiary_lookback = tertiary_lookback
        self.min_confirmations = min_confirmations
        self.interval_minutes = interval_minutes
        
        # 創建不同時間框架的檢測器
        self.detector_primary = OptimalReversalDetection(
            lookback_hours=primary_lookback,
            interval_minutes=interval_minutes
        )
        
        self.detector_secondary = OptimalReversalDetection(
            lookback_hours=secondary_lookback,
            interval_minutes=interval_minutes
        )
        
        self.detector_tertiary = OptimalReversalDetection(
            lookback_hours=tertiary_lookback,
            interval_minutes=interval_minutes
        )
    
    def detect(self, df):
        """
        多時間框架檢測
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Boolean array indicating confirmed signals
        """
        # 獲取各時間框架的信號
        signals_primary = self.detector_primary.detect(df)
        signals_secondary = self.detector_secondary.detect(df)
        signals_tertiary = self.detector_tertiary.detect(df)
        
        # 計算確認數
        confirmations = (
            signals_primary.astype(int) +
            signals_secondary.astype(int) +
            signals_tertiary.astype(int)
        )
        
        # 只在達到最少確認數時發出信號
        confirmed_signals = confirmations >= self.min_confirmations
        
        return confirmed_signals
    
    def get_color(self):
        return "darkblue"
    
    def get_marker(self):
        return "s"  # Square marker


class ConservativeHighPrecisionDetector(DetectionMethod):
    """
    保守高精確度檢測器
    
    結合所有增強策略，追求最高精確度
    可能會犧牲一些召回率
    
    Parameters:
        extreme_threshold: 極端位置閾值（更嚴格）(default: 0.15)
        atr_accel_threshold: ATR 加速閾值（更嚴格）(default: 0.7)
        volume_threshold: 成交量閾值（更嚴格）(default: 1.5)
        rsi_extreme_threshold: RSI 極端閾值 (default: 25/75)
    """
    
    def __init__(
        self,
        extreme_threshold=0.15,
        atr_accel_threshold=0.7,
        volume_threshold=1.5,
        rsi_extreme_threshold=25,
        lookback_hours=24,
        interval_minutes=15,
        name="ConservativeHighPrecision"
    ):
        super().__init__(name)
        
        self.extreme_threshold = extreme_threshold
        self.atr_accel_threshold = atr_accel_threshold
        self.volume_threshold = volume_threshold
        self.rsi_extreme_threshold = rsi_extreme_threshold
        self.lookback_hours = lookback_hours
        self.interval_minutes = interval_minutes
        
        # 使用更嚴格參數的基礎檢測器
        self.base_detector = OptimalReversalDetection(
            extreme_threshold=extreme_threshold,
            atr_accel_threshold=atr_accel_threshold,
            lookback_hours=lookback_hours,
            interval_minutes=interval_minutes
        )
    
    def detect(self, df):
        """
        保守檢測：只在所有條件都滿足時發出信號
        
        Args:
            df: DataFrame with all required indicators
            
        Returns:
            Boolean array indicating high-confidence signals
        """
        # 基礎信號
        base_signals = self.base_detector.detect(df)
        
        # 條件 1: 高成交量
        if 'volume_trend' in df.columns:
            high_volume = df['volume_trend'] > self.volume_threshold
        else:
            volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()
            high_volume = (df['volume'] / volume_ma) > self.volume_threshold
        
        # 條件 2: RSI 在極端區域
        if 'rsi' in df.columns:
            rsi_extreme = (
                (df['rsi'] < self.rsi_extreme_threshold) |
                (df['rsi'] > (100 - self.rsi_extreme_threshold))
            )
        else:
            rsi_extreme = np.ones(len(df), dtype=bool)
        
        # 條件 3: 布林帶極端位置
        if 'bb_position' in df.columns:
            bb_extreme = (df['bb_position'] < 0.1) | (df['bb_position'] > 0.9)
        else:
            bb_extreme = np.ones(len(df), dtype=bool)
        
        # 所有條件都滿足
        conservative_signals = base_signals & high_volume & rsi_extreme & bb_extreme
        
        return conservative_signals
    
    def get_color(self):
        return "darkgreen"
    
    def get_marker(self):
        return "^"  # Triangle up


def test_enhanced_detectors():
    """測試所有增強檢測器"""
    from experiments.reversal_detection.experiment import ReversalExperiment
    
    print("=" * 80)
    print("🧪 測試增強精確度檢測器")
    print("=" * 80)
    
    # 使用正確的訓練數據文件
    data_path = "training_data/reversal_training_data_2022-01-01_to_2025-09-01_15m.csv"
    
    if not os.path.exists(data_path):
        print(f"\n❌ 找不到訓練數據: {data_path}")
        print("請先運行: python experiments/reversal_detection/generate_training_data.py")
        return
    
    print(f"\n✅ 使用訓練數據: {data_path}")
    
    experiment = ReversalExperiment(data_path)
    experiment.setup()
    
    # 測試所有檢測器
    detectors = {
        "Baseline (Optimal)": OptimalReversalDetection(),
        "Enhanced Precision": EnhancedPrecisionDetector(),
        "Multi-Timeframe": MultiTimeframeDetector(),
        "Conservative High-Precision": ConservativeHighPrecisionDetector(),
    }
    
    print("\n" + "=" * 80)
    print("📊 檢測器性能比較")
    print("=" * 80)
    print(f"\n{'檢測器':<35} {'精確度':>10} {'召回率':>10} {'F1分數':>10}")
    print("-" * 80)
    
    results = {}
    for name, detector in detectors.items():
        # 使用檢測器的 detect 方法
        predictions = detector.detect(experiment.df)
        
        # 評估
        metrics = experiment.evaluator.evaluate(
            predictions,
            experiment.ground_truth,
            name
        )
        
        results[name] = metrics
        
        print(f"{name:<35} {metrics.precision:>9.2%} {metrics.recall:>9.2%} {metrics.f1_score:>10.4f}")
    
    # 找出最佳檢測器
    print("\n" + "=" * 80)
    print("🏆 最佳檢測器")
    print("=" * 80)
    
    best_f1 = max(results.values(), key=lambda x: x.f1_score)
    best_precision = max(results.values(), key=lambda x: x.precision)
    
    print(f"\n最高 F1 分數: {best_f1.method_name}")
    print(f"  - F1: {best_f1.f1_score:.4f}")
    print(f"  - 精確度: {best_f1.precision:.2%}")
    print(f"  - 召回率: {best_f1.recall:.2%}")
    
    print(f"\n最高精確度: {best_precision.method_name}")
    print(f"  - 精確度: {best_precision.precision:.2%}")
    print(f"  - F1: {best_precision.f1_score:.4f}")
    print(f"  - 召回率: {best_precision.recall:.2%}")
    
    print("\n✅ 測試完成！")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    test_enhanced_detectors()
