#!/usr/bin/env python3
"""
Confirmed Reversal Detection - 右側交易策略

策略邏輯：
1. 使用 peak_valley 找到潛在的局部極值
2. 等待價格確認反轉（回調 0.3-0.5%）
3. 在確認後立即發出信號

優勢：
- 不會在假突破時入場
- 入場點非常接近真實反轉點
- 風險最小化，利潤最大化
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from .base import DetectionMethod


class ConfirmedReversalDetection(DetectionMethod):
    """
    確認反轉檢測器 - 右側交易
    
    在檢測到局部極值後，等待價格確認反轉才發出信號
    
    Parameters:
        distance: 波峰/波谷之間的最小距離 (default: 20)
        prominence: 最小突出度，None 則自動計算 (default: None)
        height_threshold: 高度閾值 (default: 0.02 = 2%)
        confirmation_threshold: 確認閾值 (default: 0.005 = 0.5%)
        confirmation_periods: 確認需要的週期數 (default: 2)
    """
    
    def __init__(
        self,
        distance=20,
        prominence=None,
        height_threshold=0.02,
        confirmation_threshold=0.005,
        confirmation_periods=2,
        name="Confirmed Reversal"
    ):
        super().__init__(name)
        self.distance = distance
        self.prominence = prominence
        self.height_threshold = height_threshold
        self.confirmation_threshold = confirmation_threshold
        self.confirmation_periods = confirmation_periods
    
    def detect(self, df):
        """
        檢測確認的反轉點
        
        Args:
            df: DataFrame with close prices
            
        Returns:
            Boolean array indicating confirmed reversal signals
        """
        n = len(df)
        
        if n < self.distance * 2:
            return np.zeros(n, dtype=bool)
        
        prices = df['close'].values
        
        # 步驟 1: 找到潛在的局部極值
        potential_peaks, potential_valleys = self._find_potential_extremes(prices)
        
        # 步驟 2: 確認反轉
        confirmed_signals = self._confirm_reversals(
            prices, potential_peaks, potential_valleys
        )
        
        return confirmed_signals
    
    def _find_potential_extremes(self, prices):
        """
        找到潛在的波峰和波谷
        
        使用與 peak_valley.py 相同的邏輯
        """
        # 動態計算 prominence
        if self.prominence is None:
            window_size = min(500, len(prices) // 4)
            if window_size < 100:
                window_size = len(prices)
            
            rolling_std = pd.Series(prices).rolling(
                window=window_size, min_periods=1
            ).std()
            prominence = rolling_std.median() * 0.5
        else:
            prominence = self.prominence
        
        # 找波谷（買入機會）
        valleys, _ = find_peaks(
            -prices,
            distance=self.distance,
            prominence=prominence
        )
        
        # 找波峰（賣出機會）
        peaks, _ = find_peaks(
            prices,
            distance=self.distance,
            prominence=prominence
        )
        
        # 過濾顯著的極值
        significant_valleys = []
        for valley_idx in valleys:
            if self._is_significant_valley(prices, valley_idx):
                significant_valleys.append(valley_idx)
        
        significant_peaks = []
        for peak_idx in peaks:
            if self._is_significant_peak(prices, peak_idx):
                significant_peaks.append(peak_idx)
        
        return significant_peaks, significant_valleys
    
    def _is_significant_valley(self, prices, valley_idx):
        """檢查波谷是否顯著"""
        if valley_idx <= 0 or valley_idx >= len(prices) - 1:
            return False
        
        valley_price = prices[valley_idx]
        
        # 計算左右兩側的價格
        left_idx = max(0, valley_idx - self.distance // 2)
        right_idx = min(len(prices) - 1, valley_idx + self.distance // 2)
        
        left_price = prices[left_idx]
        right_price = prices[right_idx]
        
        # 計算相對高度變化
        height_change = min(
            (left_price - valley_price) / valley_price,
            (right_price - valley_price) / valley_price
        )
        
        return height_change >= self.height_threshold
    
    def _is_significant_peak(self, prices, peak_idx):
        """檢查波峰是否顯著"""
        if peak_idx <= 0 or peak_idx >= len(prices) - 1:
            return False
        
        peak_price = prices[peak_idx]
        
        # 計算左右兩側的價格
        left_idx = max(0, peak_idx - self.distance // 2)
        right_idx = min(len(prices) - 1, peak_idx + self.distance // 2)
        
        left_price = prices[left_idx]
        right_price = prices[right_idx]
        
        # 計算相對高度變化
        height_change = min(
            (peak_price - left_price) / left_price,
            (peak_price - right_price) / right_price
        )
        
        return height_change >= self.height_threshold
    
    def _confirm_reversals(self, prices, peaks, valleys):
        """
        確認反轉：等待價格回調後才發出信號
        
        對於波谷（買入）：
        - 檢測到波谷後，等待價格上漲 0.5%
        - 在上漲確認後的 1-2 個週期內發出信號
        
        對於波峰（賣出）：
        - 檢測到波峰後，等待價格下跌 0.5%
        - 在下跌確認後的 1-2 個週期內發出信號
        """
        n = len(prices)
        confirmed = np.zeros(n, dtype=bool)
        
        # 確認波谷反轉（買入信號）
        for valley_idx in valleys:
            # 檢查波谷後的價格走勢
            for i in range(1, self.confirmation_periods + 1):
                check_idx = valley_idx + i
                
                if check_idx >= n:
                    break
                
                valley_price = prices[valley_idx]
                current_price = prices[check_idx]
                
                # 計算價格變化
                price_change = (current_price - valley_price) / valley_price
                
                # 如果價格上漲超過確認閾值，發出買入信號
                if price_change >= self.confirmation_threshold:
                    confirmed[check_idx] = True
                    break  # 只在第一次確認時發出信號
        
        # 確認波峰反轉（賣出信號）
        for peak_idx in peaks:
            # 檢查波峰後的價格走勢
            for i in range(1, self.confirmation_periods + 1):
                check_idx = peak_idx + i
                
                if check_idx >= n:
                    break
                
                peak_price = prices[peak_idx]
                current_price = prices[check_idx]
                
                # 計算價格變化
                price_change = (peak_price - current_price) / peak_price
                
                # 如果價格下跌超過確認閾值，發出賣出信號
                if price_change >= self.confirmation_threshold:
                    confirmed[check_idx] = True
                    break  # 只在第一次確認時發出信號
        
        return confirmed
    
    def get_color(self):
        return "green"
    
    def get_marker(self):
        return "^"  # Triangle up


class AdaptiveConfirmedReversal(DetectionMethod):
    """
    自適應確認反轉檢測器
    
    根據市場波動自動調整確認閾值：
    - 高波動市場：需要更大的確認（0.7-1.0%）
    - 低波動市場：較小的確認即可（0.3-0.5%）
    """
    
    def __init__(
        self,
        distance=20,
        prominence=None,
        height_threshold=0.02,
        base_confirmation=0.005,
        confirmation_periods=2,
        name="Adaptive Confirmed Reversal"
    ):
        super().__init__(name)
        self.distance = distance
        self.prominence = prominence
        self.height_threshold = height_threshold
        self.base_confirmation = base_confirmation
        self.confirmation_periods = confirmation_periods
    
    def detect(self, df):
        """檢測確認的反轉點（自適應版本）"""
        n = len(df)
        
        if n < self.distance * 2:
            return np.zeros(n, dtype=bool)
        
        prices = df['close'].values
        
        # 計算市場波動度
        volatility = self._calculate_volatility(prices)
        
        # 根據波動度調整確認閾值
        adaptive_threshold = self._adapt_threshold(volatility)
        
        # 使用自適應閾值創建檢測器
        detector = ConfirmedReversalDetection(
            distance=self.distance,
            prominence=self.prominence,
            height_threshold=self.height_threshold,
            confirmation_threshold=adaptive_threshold,
            confirmation_periods=self.confirmation_periods
        )
        
        return detector.detect(df)
    
    def _calculate_volatility(self, prices):
        """計算市場波動度"""
        # 使用 20 週期的標準差
        window = min(20, len(prices) // 4)
        volatility = pd.Series(prices).rolling(window=window).std()
        return volatility.median()
    
    def _adapt_threshold(self, volatility):
        """根據波動度調整確認閾值"""
        # 波動度越高，需要越大的確認
        # 基礎閾值 * (1 + 波動度因子)
        
        # 假設正常波動度在 100-500 之間
        if volatility < 100:
            factor = 0.6  # 低波動：0.3%
        elif volatility < 300:
            factor = 1.0  # 正常波動：0.5%
        elif volatility < 500:
            factor = 1.4  # 高波動：0.7%
        else:
            factor = 2.0  # 極高波動：1.0%
        
        return self.base_confirmation * factor
    
    def get_color(self):
        return "blue"
    
    def get_marker(self):
        return "D"  # Diamond


class StrictConfirmedReversal(DetectionMethod):
    """
    嚴格確認反轉檢測器
    
    要求：
    1. 價格確認反轉（0.5%）
    2. 成交量確認（放大）
    3. 連續 2 個週期確認
    
    精確度最高，但信號最少
    """
    
    def __init__(
        self,
        distance=20,
        prominence=None,
        height_threshold=0.02,
        confirmation_threshold=0.005,
        volume_threshold=1.2,
        name="Strict Confirmed Reversal"
    ):
        super().__init__(name)
        self.distance = distance
        self.prominence = prominence
        self.height_threshold = height_threshold
        self.confirmation_threshold = confirmation_threshold
        self.volume_threshold = volume_threshold
    
    def detect(self, df):
        """檢測嚴格確認的反轉點"""
        # 先用基礎檢測器找到確認的反轉
        base_detector = ConfirmedReversalDetection(
            distance=self.distance,
            prominence=self.prominence,
            height_threshold=self.height_threshold,
            confirmation_threshold=self.confirmation_threshold,
            confirmation_periods=2
        )
        
        base_signals = base_detector.detect(df)
        
        # 增加成交量確認
        if 'volume' in df.columns:
            volume_confirmed = self._volume_confirmation(df, base_signals)
            return volume_confirmed
        else:
            return base_signals
    
    def _volume_confirmation(self, df, signals):
        """成交量確認"""
        # 計算成交量趨勢
        volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()
        volume_ratio = df['volume'] / volume_ma
        
        # 只在成交量放大時確認信號
        confirmed = signals & (volume_ratio > self.volume_threshold)
        
        return confirmed
    
    def get_color(self):
        return "darkgreen"
    
    def get_marker(self):
        return "s"  # Square
