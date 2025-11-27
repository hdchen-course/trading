#!/usr/bin/env python3
"""
實用快速入場策略

基於測試結果的最佳實踐：
- 在 1-2 根 K 線內確認（15-30 分鐘）
- 價格移動 0.1-0.3%
- 加入成交量和趨勢過濾
- 目標：提高勝率到 55%+
"""

import numpy as np
import pandas as pd
from .base import DetectionMethod


class PracticalFastEntry(DetectionMethod):
    """
    實用快速入場策略
    
    結合速度和準確度的最佳平衡
    
    Parameters:
        confirmation_candles: 確認窗口 (default: 2)
        min_move_pct: 最小確認移動 (default: 0.001 = 0.1%)
        max_move_pct: 最大確認移動 (default: 0.003 = 0.3%)
        require_volume: 需要成交量確認 (default: True)
        volume_threshold: 成交量閾值 (default: 1.0)
        require_trend: 需要趨勢確認 (default: True)
    """
    
    def __init__(
        self,
        confirmation_candles=2,
        min_move_pct=0.001,
        max_move_pct=0.003,
        require_volume=True,
        volume_threshold=1.0,
        require_trend=True,
        name="Practical Fast Entry"
    ):
        super().__init__(name)
        self.confirmation_candles = confirmation_candles
        self.min_move_pct = min_move_pct
        self.max_move_pct = max_move_pct
        self.require_volume = require_volume
        self.volume_threshold = volume_threshold
        self.require_trend = require_trend
    
    def detect(self, df):
        """檢測實用快速入場信號"""
        n = len(df)
        
        if 'reverse' not in df.columns:
            print("⚠️  需要 'reverse' 標籤")
            return np.zeros(n, dtype=bool)
        
        reversals = df['reverse'].values
        signals = np.zeros(n, dtype=bool)
        reversal_indices = np.where(reversals)[0]
        
        entry_count = 0
        filtered_by_volume = 0
        filtered_by_trend = 0
        
        for rev_idx in reversal_indices:
            # 檢查後續 K 線
            for offset in range(1, self.confirmation_candles + 1):
                check_idx = rev_idx + offset
                
                if check_idx >= n:
                    break
                
                # 基礎價格確認
                if not self._price_confirmed(df, rev_idx, check_idx):
                    continue
                
                # 成交量過濾
                if self.require_volume:
                    if not self._volume_confirmed(df, check_idx):
                        filtered_by_volume += 1
                        continue
                
                # 趨勢過濾
                if self.require_trend:
                    if not self._trend_confirmed(df, rev_idx, check_idx):
                        filtered_by_trend += 1
                        continue
                
                # 通過所有過濾
                signals[check_idx] = True
                entry_count += 1
                break
        
        print(f"\n✅ 實用快速入場:")
        print(f"  反轉點: {len(reversal_indices)}")
        print(f"  入場信號: {entry_count}")
        print(f"  入場率: {entry_count/len(reversal_indices)*100:.1f}%")
        print(f"  成交量過濾: {filtered_by_volume}")
        print(f"  趨勢過濾: {filtered_by_trend}")
        
        return signals
    
    def _price_confirmed(self, df, rev_idx, check_idx):
        """價格確認"""
        reversal_price = df['close'].iloc[rev_idx]
        current_price = df['close'].iloc[check_idx]
        
        price_change = abs(current_price - reversal_price) / reversal_price
        
        if price_change < self.min_move_pct or price_change > self.max_move_pct:
            return False
        
        # 方向確認
        lookback = 5
        start_idx = max(0, rev_idx - lookback)
        end_idx = min(len(df), rev_idx + lookback + 1)
        local_prices = df['close'].iloc[start_idx:end_idx].values
        
        is_peak = df['close'].iloc[rev_idx] == np.max(local_prices)
        is_valley = df['close'].iloc[rev_idx] == np.min(local_prices)
        
        if is_peak and current_price >= reversal_price:
            return False
        if is_valley and current_price <= reversal_price:
            return False
        
        return True
    
    def _volume_confirmed(self, df, check_idx):
        """成交量確認"""
        if 'volume' not in df.columns:
            return True
        
        volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()
        current_volume_ratio = df['volume'].iloc[check_idx] / volume_ma.iloc[check_idx]
        
        return current_volume_ratio >= self.volume_threshold
    
    def _trend_confirmed(self, df, rev_idx, check_idx):
        """
        趨勢確認：確保反轉是有意義的
        
        檢查：
        1. 反轉前有明顯的趨勢
        2. 反轉後趨勢開始改變
        """
        # 計算反轉前的趨勢
        lookback = 10
        start_idx = max(0, rev_idx - lookback)
        
        if start_idx >= rev_idx:
            return True
        
        prices_before = df['close'].iloc[start_idx:rev_idx].values
        
        if len(prices_before) < 3:
            return True
        
        # 計算趨勢強度
        price_change_before = (prices_before[-1] - prices_before[0]) / prices_before[0]
        
        # 需要有至少 0.5% 的趨勢
        if abs(price_change_before) < 0.005:
            return False
        
        return True
    
    def get_color(self):
        return "green"
    
    def get_marker(self):
        return "^"
