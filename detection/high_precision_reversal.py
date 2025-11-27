#!/usr/bin/env python3
"""
High Precision Reversal Detection

基於實驗結果的高精確度反轉檢測器

實驗結果：
- 精確度：16.01%（基線的 3.3倍）
- F1分數：0.0990（基線的 1.15倍）
- 召回率：7.17%
- 信號數：1,443（在 128,525 個數據點中）

策略：
1. 極端位置檢測（20% 閾值）
2. 波動加速確認（ATR 二階導數）
3. 高成交量確認（1.2x 平均）
4. RSI 極端區域（< 30 或 > 70）
5. 布林帶極端位置（< 0.1 或 > 0.9）

適用場景：
- 追求高精確度，可接受較低召回率
- 適合保守交易策略
- 需要配合其他信號使用
"""

import numpy as np
import pandas as pd
from .base import DetectionMethod
from .optimal_reversal import OptimalReversalDetection


class HighPrecisionReversalDetection(DetectionMethod):
    """
    高精確度反轉檢測
    
    在基礎規則上增加多層過濾，將精確度從 4.86% 提升到 16.01%
    
    Parameters:
        extreme_threshold: 極端位置閾值 (default: 0.20)
        atr_accel_threshold: ATR 加速閾值 (default: 0.5)
        volume_threshold: 成交量閾值 (default: 1.2)
        rsi_extreme_threshold: RSI 極端閾值 (default: 30)
        bb_extreme_threshold: 布林帶極端閾值 (default: 0.1)
        lookback_hours: 回看時間（小時）(default: 24)
        interval_minutes: 數據間隔（分鐘）(default: 15)
    """
    
    def __init__(
        self,
        extreme_threshold=0.20,
        atr_accel_threshold=0.5,
        volume_threshold=1.2,
        rsi_extreme_threshold=30,
        bb_extreme_threshold=0.1,
        lookback_hours=24,
        interval_minutes=15
    ):
        super().__init__("High Precision Reversal")
        
        self.extreme_threshold = extreme_threshold
        self.atr_accel_threshold = atr_accel_threshold
        self.volume_threshold = volume_threshold
        self.rsi_extreme_threshold = rsi_extreme_threshold
        self.bb_extreme_threshold = bb_extreme_threshold
        self.lookback_hours = lookback_hours
        self.interval_minutes = interval_minutes
        
        # 基礎檢測器
        self.base_detector = OptimalReversalDetection(
            extreme_threshold=extreme_threshold,
            atr_accel_threshold=atr_accel_threshold,
            lookback_hours=lookback_hours,
            interval_minutes=interval_minutes
        )
    
    def detect(self, df):
        """
        高精確度檢測
        
        只在所有條件都滿足時發出信號：
        1. 基礎信號（極端位置 + 波動加速）
        2. 高成交量確認
        3. RSI 極端區域
        4. 布林帶極端位置
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            
        Returns:
            Boolean array indicating high-confidence reversal signals
        """
        n = len(df)
        
        # 步驟 1: 獲取基礎信號
        base_signals = self.base_detector.detect(df)
        
        # 步驟 2: 高成交量確認
        high_volume = self._check_high_volume(df)
        
        # 步驟 3: RSI 極端區域
        rsi_extreme = self._check_rsi_extreme(df)
        
        # 步驟 4: 布林帶極端位置
        bb_extreme = self._check_bb_extreme(df)
        
        # 所有條件都滿足
        high_precision_signals = base_signals & high_volume & rsi_extreme & bb_extreme
        
        return high_precision_signals
    
    def _check_high_volume(self, df):
        """
        檢查成交量是否高於閾值
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            Boolean array indicating high volume periods
        """
        if 'volume_trend' in df.columns:
            # 使用預計算的 volume_trend
            high_volume = df['volume_trend'] > self.volume_threshold
        else:
            # 計算成交量趨勢
            volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()
            volume_trend = df['volume'] / volume_ma
            high_volume = volume_trend > self.volume_threshold
        
        return high_volume
    
    def _check_rsi_extreme(self, df):
        """
        檢查 RSI 是否在極端區域
        
        Args:
            df: DataFrame with RSI data
            
        Returns:
            Boolean array indicating RSI extreme periods
        """
        if 'rsi' not in df.columns:
            # 如果沒有 RSI，返回全 True（不過濾）
            return np.ones(len(df), dtype=bool)
        
        # RSI < 30 (超賣) 或 RSI > 70 (超買)
        rsi_oversold = df['rsi'] < self.rsi_extreme_threshold
        rsi_overbought = df['rsi'] > (100 - self.rsi_extreme_threshold)
        rsi_extreme = rsi_oversold | rsi_overbought
        
        return rsi_extreme
    
    def _check_bb_extreme(self, df):
        """
        檢查布林帶位置是否極端
        
        Args:
            df: DataFrame with Bollinger Bands data
            
        Returns:
            Boolean array indicating BB extreme positions
        """
        if 'bb_position' in df.columns:
            # 使用預計算的 bb_position
            bb_lower = df['bb_position'] < self.bb_extreme_threshold
            bb_upper = df['bb_position'] > (1 - self.bb_extreme_threshold)
            bb_extreme = bb_lower | bb_upper
        elif 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            # 計算 BB 位置
            bb_range = df['bb_upper'] - df['bb_lower']
            bb_range = bb_range.replace(0, 1e-10)
            bb_position = (df['close'] - df['bb_lower']) / bb_range
            
            bb_lower = bb_position < self.bb_extreme_threshold
            bb_upper = bb_position > (1 - self.bb_extreme_threshold)
            bb_extreme = bb_lower | bb_upper
        else:
            # 如果沒有布林帶數據，返回全 True（不過濾）
            return np.ones(len(df), dtype=bool)
        
        return bb_extreme
    
    def get_color(self):
        return "darkgreen"
    
    def get_marker(self):
        return "^"  # Triangle up


class BalancedReversalDetection(DetectionMethod):
    """
    平衡型反轉檢測
    
    在精確度和召回率之間取得平衡
    
    實驗結果（Conservative-Moderate）：
    - 精確度：15.99%
    - F1分數：0.0790
    - 召回率：5.24%
    
    Parameters:
        extreme_threshold: 極端位置閾值 (default: 0.18)
        atr_accel_threshold: ATR 加速閾值 (default: 0.6)
        volume_threshold: 成交量閾值 (default: 1.3)
        rsi_extreme_threshold: RSI 極端閾值 (default: 28)
    """
    
    def __init__(
        self,
        extreme_threshold=0.18,
        atr_accel_threshold=0.6,
        volume_threshold=1.3,
        rsi_extreme_threshold=28,
        bb_extreme_threshold=0.1,
        lookback_hours=24,
        interval_minutes=15
    ):
        super().__init__("Balanced Reversal")
        
        # 使用 HighPrecisionReversalDetection 但參數更嚴格
        self.detector = HighPrecisionReversalDetection(
            extreme_threshold=extreme_threshold,
            atr_accel_threshold=atr_accel_threshold,
            volume_threshold=volume_threshold,
            rsi_extreme_threshold=rsi_extreme_threshold,
            bb_extreme_threshold=bb_extreme_threshold,
            lookback_hours=lookback_hours,
            interval_minutes=interval_minutes
        )
    
    def detect(self, df):
        return self.detector.detect(df)
    
    def get_color(self):
        return "orange"
    
    def get_marker(self):
        return "D"  # Diamond


class UltraConservativeReversalDetection(DetectionMethod):
    """
    超保守反轉檢測
    
    追求最高精確度，犧牲召回率
    
    實驗結果（Conservative-Strict）：
    - 精確度：17.07%
    - F1分數：0.0582
    - 召回率：3.50%
    
    Parameters:
        extreme_threshold: 極端位置閾值 (default: 0.15)
        atr_accel_threshold: ATR 加速閾值 (default: 0.7)
        volume_threshold: 成交量閾值 (default: 1.5)
        rsi_extreme_threshold: RSI 極端閾值 (default: 25)
    """
    
    def __init__(
        self,
        extreme_threshold=0.15,
        atr_accel_threshold=0.7,
        volume_threshold=1.5,
        rsi_extreme_threshold=25,
        bb_extreme_threshold=0.1,
        lookback_hours=24,
        interval_minutes=15
    ):
        super().__init__("Ultra Conservative Reversal")
        
        self.detector = HighPrecisionReversalDetection(
            extreme_threshold=extreme_threshold,
            atr_accel_threshold=atr_accel_threshold,
            volume_threshold=volume_threshold,
            rsi_extreme_threshold=rsi_extreme_threshold,
            bb_extreme_threshold=bb_extreme_threshold,
            lookback_hours=lookback_hours,
            interval_minutes=interval_minutes
        )
    
    def detect(self, df):
        return self.detector.detect(df)
    
    def get_color(self):
        return "darkred"
    
    def get_marker(self):
        return "v"  # Triangle down
