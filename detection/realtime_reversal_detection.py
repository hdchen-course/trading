#!/usr/bin/env python3
"""
Real-time Reversal Detection - 實盤可用的反轉檢測

不依賴事後標籤，純粹基於當前和歷史數據判斷反轉

策略：
1. 檢測價格創局部新高/新低
2. 觀察價格開始反向移動
3. 多指標確認
4. 快速入場（1-2 根 K 線內）
"""

import numpy as np
import pandas as pd
from .base import DetectionMethod


class RealtimeReversalDetection(DetectionMethod):
    """
    實時反轉檢測 - 不需要事後標籤
    
    檢測邏輯：
    1. 當前價格是最近 N 根 K 線的極值
    2. 價格開始反向移動（0.1-0.3%）
    3. 成交量確認（可選）
    4. 立即發出信號
    
    Parameters:
        lookback_extreme: 檢測極值的回看期 (default: 20)
        min_reversal_pct: 最小反轉確認 (default: 0.001 = 0.1%)
        max_reversal_pct: 最大反轉確認 (default: 0.003 = 0.3%)
        require_volume: 需要成交量確認 (default: True)
        volume_threshold: 成交量閾值 (default: 1.0)
        require_rsi: 需要 RSI 確認 (default: False)
    """
    
    def __init__(
        self,
        lookback_extreme=20,
        min_reversal_pct=0.001,
        max_reversal_pct=0.003,
        require_volume=True,
        volume_threshold=1.0,
        require_rsi=False,
        name="Realtime Reversal"
    ):
        super().__init__(name)
        self.lookback_extreme = lookback_extreme
        self.min_reversal_pct = min_reversal_pct
        self.max_reversal_pct = max_reversal_pct
        self.require_volume = require_volume
        self.volume_threshold = volume_threshold
        self.require_rsi = require_rsi
    
    def detect(self, df):
        """
        實時檢測反轉信號
        
        這個方法模擬實盤情況：
        - 對每個時間點，只使用該點之前的數據
        - 不使用未來數據
        """
        n = len(df)
        signals = np.zeros(n, dtype=bool)
        
        # 從 lookback_extreme 開始檢測
        for i in range(self.lookback_extreme, n):
            if self._is_reversal_at(df, i):
                signals[i] = True
        
        print(f"\n🔴 實時反轉檢測:")
        print(f"  檢測到 {np.sum(signals)} 個反轉信號")
        
        return signals
    
    def _is_reversal_at(self, df, idx):
        """
        判斷在 idx 位置是否有反轉信號
        
        只使用 idx 及之前的數據（模擬實盤）
        """
        # 1. 檢查是否是局部極值
        extreme_type = self._check_local_extreme(df, idx)
        
        if extreme_type is None:
            return False
        
        # 2. 檢查價格是否開始反向移動
        if not self._check_reversal_move(df, idx, extreme_type):
            return False
        
        # 3. 成交量確認
        if self.require_volume:
            if not self._check_volume(df, idx):
                return False
        
        # 4. RSI 確認
        if self.require_rsi:
            if not self._check_rsi(df, idx, extreme_type):
                return False
        
        return True
    
    def _check_local_extreme(self, df, idx):
        """
        檢查是否是局部極值
        
        Returns:
            'peak': 局部高點
            'valley': 局部低點
            None: 不是極值
        """
        # 獲取回看窗口
        start_idx = max(0, idx - self.lookback_extreme)
        window_prices = df['close'].iloc[start_idx:idx+1].values
        
        if len(window_prices) < self.lookback_extreme:
            return None
        
        current_price = window_prices[-1]
        
        # 檢查是否是最高點
        if current_price == np.max(window_prices):
            # 確保不是平頂（有多個相同的最高點）
            max_count = np.sum(window_prices == current_price)
            if max_count == 1:
                return 'peak'
        
        # 檢查是否是最低點
        if current_price == np.min(window_prices):
            # 確保不是平底
            min_count = np.sum(window_prices == current_price)
            if min_count == 1:
                return 'valley'
        
        return None
    
    def _check_reversal_move(self, df, idx, extreme_type):
        """
        檢查價格是否開始反向移動
        
        對於波峰：檢查後續 1-2 根 K 線是否下跌
        對於波谷：檢查後續 1-2 根 K 線是否上漲
        """
        # 注意：這裡我們需要看「當前」K 線相對於「極值」K 線的變化
        # 在實盤中，極值可能是前 1-2 根 K 線
        
        # 找到極值位置
        start_idx = max(0, idx - 2)  # 最多往前看 2 根
        window_prices = df['close'].iloc[start_idx:idx+1].values
        
        if extreme_type == 'peak':
            extreme_price = np.max(window_prices)
            extreme_idx_in_window = np.argmax(window_prices)
        else:  # valley
            extreme_price = np.min(window_prices)
            extreme_idx_in_window = np.argmin(window_prices)
        
        extreme_idx = start_idx + extreme_idx_in_window
        
        # 如果極值就是當前點，還沒有反轉
        if extreme_idx == idx:
            return False
        
        # 計算從極值到當前的價格變化
        current_price = df['close'].iloc[idx]
        price_change = abs(current_price - extreme_price) / extreme_price
        
        # 檢查變化幅度
        if price_change < self.min_reversal_pct or price_change > self.max_reversal_pct:
            return False
        
        # 檢查方向
        if extreme_type == 'peak':
            # 波峰後應該下跌
            if current_price >= extreme_price:
                return False
        else:  # valley
            # 波谷後應該上漲
            if current_price <= extreme_price:
                return False
        
        return True
    
    def _check_volume(self, df, idx):
        """成交量確認"""
        if 'volume' not in df.columns:
            return True
        
        # 計算成交量均線
        start_idx = max(0, idx - 20)
        volume_ma = df['volume'].iloc[start_idx:idx+1].mean()
        
        current_volume = df['volume'].iloc[idx]
        
        return current_volume >= volume_ma * self.volume_threshold
    
    def _check_rsi(self, df, idx, extreme_type):
        """RSI 確認"""
        if 'rsi' not in df.columns:
            return True
        
        rsi = df['rsi'].iloc[idx]
        
        if extreme_type == 'peak':
            # 波峰應該在超買區
            return rsi > 70
        else:  # valley
            # 波谷應該在超賣區
            return rsi < 30
    
    def get_color(self):
        return "red"
    
    def get_marker(self):
        return "o"


class ConservativeRealtimeReversal(DetectionMethod):
    """
    保守實時反轉檢測
    
    更嚴格的條件，更高的精確度
    """
    
    def __init__(
        self,
        lookback_extreme=20,
        min_reversal_pct=0.002,  # 0.2%
        max_reversal_pct=0.005,  # 0.5%
        require_volume=True,
        volume_threshold=1.2,
        require_rsi=True,
        name="Conservative Realtime"
    ):
        super().__init__(name)
        self.base_detector = RealtimeReversalDetection(
            lookback_extreme=lookback_extreme,
            min_reversal_pct=min_reversal_pct,
            max_reversal_pct=max_reversal_pct,
            require_volume=require_volume,
            volume_threshold=volume_threshold,
            require_rsi=require_rsi
        )
    
    def detect(self, df):
        return self.base_detector.detect(df)
    
    def get_color(self):
        return "darkred"
    
    def get_marker(self):
        return "D"


class AggressiveRealtimeReversal(DetectionMethod):
    """
    激進實時反轉檢測
    
    更寬鬆的條件，更多信號
    """
    
    def __init__(
        self,
        lookback_extreme=15,
        min_reversal_pct=0.0005,  # 0.05%
        max_reversal_pct=0.002,   # 0.2%
        require_volume=False,
        require_rsi=False,
        name="Aggressive Realtime"
    ):
        super().__init__(name)
        self.base_detector = RealtimeReversalDetection(
            lookback_extreme=lookback_extreme,
            min_reversal_pct=min_reversal_pct,
            max_reversal_pct=max_reversal_pct,
            require_volume=require_volume,
            require_rsi=require_rsi
        )
    
    def detect(self, df):
        return self.base_detector.detect(df)
    
    def get_color(self):
        return "orange"
    
    def get_marker(self):
        return "^"
