#!/usr/bin/env python3
"""
Early Reversal Confirmation - 早期反轉確認

策略：檢測反轉正在形成的早期跡象

不是預測反轉（左側），也不是等反轉完成（太晚）
而是在反轉剛開始形成時就識別出來（右側但很早）

關鍵指標：
1. 價格創新高/新低後開始回調
2. 回調幅度達到 0.3-0.5%
3. 成交量確認
4. 動量指標轉向
"""

import numpy as np
import pandas as pd
from .base import DetectionMethod


class EarlyReversalConfirmation(DetectionMethod):
    """
    早期反轉確認檢測器
    
    檢測邏輯：
    1. 識別局部極值（最近 N 個週期的最高/最低點）
    2. 檢測價格開始反向移動
    3. 確認反向移動達到閾值（0.3-0.5%）
    4. 立即發出信號
    
    Parameters:
        lookback_extreme: 檢測極值的回看週期 (default: 20)
        confirmation_pct: 確認反轉的百分比 (default: 0.005 = 0.5%)
        min_extreme_pct: 極值必須是顯著的 (default: 0.01 = 1%)
    """
    
    def __init__(
        self,
        lookback_extreme=20,
        confirmation_pct=0.005,
        min_extreme_pct=0.01,
        name="Early Reversal Confirmation"
    ):
        super().__init__(name)
        self.lookback_extreme = lookback_extreme
        self.confirmation_pct = confirmation_pct
        self.min_extreme_pct = min_extreme_pct
    
    def detect(self, df):
        """
        檢測早期反轉確認信號
        
        Args:
            df: DataFrame with close prices
            
        Returns:
            Boolean array indicating early reversal signals
        """
        n = len(df)
        
        if n < self.lookback_extreme + 5:
            return np.zeros(n, dtype=bool)
        
        prices = df['close'].values
        signals = np.zeros(n, dtype=bool)
        
        # 對每個時間點，檢查是否出現早期反轉跡象
        for i in range(self.lookback_extreme, n):
            # 獲取回看窗口
            window_start = i - self.lookback_extreme
            window_prices = prices[window_start:i+1]
            
            current_price = prices[i]
            
            # 檢查是否是局部高點後的回調（賣出信號）
            if self._is_peak_reversal(window_prices, current_price):
                signals[i] = True
            
            # 檢查是否是局部低點後的反彈（買入信號）
            elif self._is_valley_reversal(window_prices, current_price):
                signals[i] = True
        
        return signals
    
    def _is_peak_reversal(self, window_prices, current_price):
        """
        檢測波峰反轉（賣出信號）
        
        條件：
        1. 最近創了新高
        2. 現在價格開始回落
        3. 回落幅度達到確認閾值
        """
        # 找到窗口內的最高點
        max_price = np.max(window_prices[:-1])  # 不包括當前點
        max_idx = np.argmax(window_prices[:-1])
        
        # 最高點必須在最近幾個週期內
        if max_idx < len(window_prices) - 5:
            return False
        
        # 檢查最高點是否顯著
        # 最高點必須比窗口前半部分高出至少 min_extreme_pct
        first_half_max = np.max(window_prices[:len(window_prices)//2])
        if (max_price - first_half_max) / first_half_max < self.min_extreme_pct:
            return False
        
        # 檢查當前價格是否從最高點回落
        if current_price >= max_price:
            return False
        
        # 計算回落幅度
        pullback = (max_price - current_price) / max_price
        
        # 回落幅度達到確認閾值
        if pullback >= self.confirmation_pct:
            return True
        
        return False
    
    def _is_valley_reversal(self, window_prices, current_price):
        """
        檢測波谷反轉（買入信號）
        
        條件：
        1. 最近創了新低
        2. 現在價格開始反彈
        3. 反彈幅度達到確認閾值
        """
        # 找到窗口內的最低點
        min_price = np.min(window_prices[:-1])  # 不包括當前點
        min_idx = np.argmin(window_prices[:-1])
        
        # 最低點必須在最近幾個週期內
        if min_idx < len(window_prices) - 5:
            return False
        
        # 檢查最低點是否顯著
        # 最低點必須比窗口前半部分低至少 min_extreme_pct
        first_half_min = np.min(window_prices[:len(window_prices)//2])
        if (first_half_min - min_price) / first_half_min < self.min_extreme_pct:
            return False
        
        # 檢查當前價格是否從最低點反彈
        if current_price <= min_price:
            return False
        
        # 計算反彈幅度
        bounce = (current_price - min_price) / min_price
        
        # 反彈幅度達到確認閾值
        if bounce >= self.confirmation_pct:
            return True
        
        return False
    
    def get_color(self):
        return "green"
    
    def get_marker(self):
        return "^"


class VolumeConfirmedEarlyReversal(DetectionMethod):
    """
    成交量確認的早期反轉
    
    在 EarlyReversalConfirmation 基礎上增加成交量確認
    """
    
    def __init__(
        self,
        lookback_extreme=20,
        confirmation_pct=0.005,
        min_extreme_pct=0.01,
        volume_threshold=1.2,
        name="Volume Confirmed Early Reversal"
    ):
        super().__init__(name)
        self.base_detector = EarlyReversalConfirmation(
            lookback_extreme=lookback_extreme,
            confirmation_pct=confirmation_pct,
            min_extreme_pct=min_extreme_pct
        )
        self.volume_threshold = volume_threshold
    
    def detect(self, df):
        """檢測成交量確認的早期反轉"""
        # 獲取基礎信號
        base_signals = self.base_detector.detect(df)
        
        # 成交量確認
        if 'volume' not in df.columns:
            return base_signals
        
        # 計算成交量趨勢
        volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()
        volume_ratio = df['volume'] / volume_ma
        
        # 只在成交量放大時確認
        confirmed = base_signals & (volume_ratio > self.volume_threshold)
        
        return confirmed
    
    def get_color(self):
        return "darkgreen"
    
    def get_marker(self):
        return "D"


class MultiIndicatorEarlyReversal(DetectionMethod):
    """
    多指標確認的早期反轉
    
    結合：
    1. 價格反轉確認
    2. 成交量確認
    3. RSI 確認（如果有）
    """
    
    def __init__(
        self,
        lookback_extreme=20,
        confirmation_pct=0.005,
        min_extreme_pct=0.01,
        volume_threshold=1.2,
        rsi_oversold=30,
        rsi_overbought=70,
        name="Multi-Indicator Early Reversal"
    ):
        super().__init__(name)
        self.base_detector = EarlyReversalConfirmation(
            lookback_extreme=lookback_extreme,
            confirmation_pct=confirmation_pct,
            min_extreme_pct=min_extreme_pct
        )
        self.volume_threshold = volume_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def detect(self, df):
        """檢測多指標確認的早期反轉"""
        # 獲取基礎信號
        base_signals = self.base_detector.detect(df)
        
        # 成交量確認
        volume_confirmed = np.ones(len(df), dtype=bool)
        if 'volume' in df.columns:
            volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()
            volume_ratio = df['volume'] / volume_ma
            volume_confirmed = volume_ratio > self.volume_threshold
        
        # RSI 確認
        rsi_confirmed = np.ones(len(df), dtype=bool)
        if 'rsi' in df.columns:
            # 在超賣或超買區域
            rsi_confirmed = (df['rsi'] < self.rsi_oversold) | (df['rsi'] > self.rsi_overbought)
        
        # 所有條件都滿足
        confirmed = base_signals & volume_confirmed & rsi_confirmed
        
        return confirmed
    
    def get_color(self):
        return "purple"
    
    def get_marker(self):
        return "s"
