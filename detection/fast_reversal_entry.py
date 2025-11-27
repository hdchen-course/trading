#!/usr/bin/env python3
"""
Fast Reversal Entry - 快速反轉入場

目標：在反轉點出現後 1-2 根 K 線內確認並入場

策略：
1. 使用已標記的反轉點（peak_valley）
2. 在反轉點後的 1-2 根 K 線內檢測確認信號
3. 確認條件：
   - 價格開始反向移動（0.1-0.3%）
   - 成交量確認（可選）
   - K 線形態確認（可選）

優勢：
- 入場非常快（15-30 分鐘內）
- 距離反轉點很近（風險小）
- 不會錯過快速反轉
"""

import numpy as np
import pandas as pd
from .base import DetectionMethod


class FastReversalEntry(DetectionMethod):
    """
    快速反轉入場檢測器
    
    在已知反轉點的基礎上，快速確認並發出入場信號
    
    Parameters:
        confirmation_candles: 確認需要的 K 線數 (default: 2)
        min_move_pct: 最小價格移動百分比 (default: 0.001 = 0.1%)
        max_move_pct: 最大價格移動百分比 (default: 0.003 = 0.3%)
        require_volume: 是否需要成交量確認 (default: False)
        volume_threshold: 成交量閾值 (default: 1.0)
    """
    
    def __init__(
        self,
        confirmation_candles=2,
        min_move_pct=0.001,
        max_move_pct=0.003,
        require_volume=False,
        volume_threshold=1.0,
        name="Fast Reversal Entry"
    ):
        super().__init__(name)
        self.confirmation_candles = confirmation_candles
        self.min_move_pct = min_move_pct
        self.max_move_pct = max_move_pct
        self.require_volume = require_volume
        self.volume_threshold = volume_threshold
    
    def detect(self, df):
        """
        檢測快速入場信號
        
        Args:
            df: DataFrame with close prices and 'reverse' labels
            
        Returns:
            Boolean array indicating entry signals
        """
        n = len(df)
        
        if 'reverse' not in df.columns:
            print("⚠️  數據中沒有 'reverse' 標籤，無法使用快速入場策略")
            return np.zeros(n, dtype=bool)
        
        prices = df['close'].values
        reversals = df['reverse'].values
        signals = np.zeros(n, dtype=bool)
        
        # 找到所有反轉點
        reversal_indices = np.where(reversals)[0]
        
        print(f"\n🔍 快速入場分析:")
        print(f"  找到 {len(reversal_indices)} 個反轉點")
        print(f"  確認窗口: {self.confirmation_candles} 根 K 線")
        print(f"  價格移動範圍: {self.min_move_pct*100:.2f}% - {self.max_move_pct*100:.2f}%")
        
        entry_count = 0
        
        # 對每個反轉點，檢查後續 K 線是否確認
        for rev_idx in reversal_indices:
            # 檢查後續的 K 線
            for offset in range(1, self.confirmation_candles + 1):
                check_idx = rev_idx + offset
                
                if check_idx >= n:
                    break
                
                # 檢查是否確認反轉
                if self._is_confirmed(df, rev_idx, check_idx):
                    signals[check_idx] = True
                    entry_count += 1
                    break  # 只在第一次確認時入場
        
        print(f"  生成入場信號: {entry_count} 個")
        print(f"  入場率: {entry_count / len(reversal_indices) * 100:.1f}%")
        
        return signals
    
    def _is_confirmed(self, df, reversal_idx, check_idx):
        """
        檢查反轉是否被確認
        
        Args:
            df: DataFrame
            reversal_idx: 反轉點索引
            check_idx: 檢查點索引
            
        Returns:
            Boolean indicating if reversal is confirmed
        """
        reversal_price = df['close'].iloc[reversal_idx]
        current_price = df['close'].iloc[check_idx]
        
        # 計算價格變化
        price_change = abs(current_price - reversal_price) / reversal_price
        
        # 檢查價格移動是否在範圍內
        if price_change < self.min_move_pct or price_change > self.max_move_pct:
            return False
        
        # 檢查方向：需要確認是反轉方向
        # 如果是波谷（低點），價格應該上漲
        # 如果是波峰（高點），價格應該下跌
        
        # 判斷是波峰還是波谷
        lookback = 5
        start_idx = max(0, reversal_idx - lookback)
        end_idx = min(len(df), reversal_idx + lookback + 1)
        
        local_prices = df['close'].iloc[start_idx:end_idx].values
        reversal_price_in_window = df['close'].iloc[reversal_idx]
        
        is_peak = reversal_price_in_window == np.max(local_prices)
        is_valley = reversal_price_in_window == np.min(local_prices)
        
        # 波峰：價格應該下跌
        if is_peak and current_price < reversal_price:
            pass  # 確認
        # 波谷：價格應該上漲
        elif is_valley and current_price > reversal_price:
            pass  # 確認
        else:
            return False
        
        # 成交量確認（可選）
        if self.require_volume and 'volume' in df.columns:
            volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()
            current_volume_ratio = df['volume'].iloc[check_idx] / volume_ma.iloc[check_idx]
            
            if current_volume_ratio < self.volume_threshold:
                return False
        
        return True
    
    def get_color(self):
        return "green"
    
    def get_marker(self):
        return "^"


class UltraFastEntry(DetectionMethod):
    """
    超快速入場 - 第一根 K 線就入場
    
    最激進的策略：
    - 只要反轉點出現，下一根 K 線立即入場
    - 不等待任何確認
    - 風險：可能是假突破
    - 優勢：入場最快，距離反轉點最近
    """
    
    def __init__(self, name="Ultra Fast Entry"):
        super().__init__(name)
    
    def detect(self, df):
        """在反轉點後立即入場"""
        n = len(df)
        
        if 'reverse' not in df.columns:
            return np.zeros(n, dtype=bool)
        
        reversals = df['reverse'].values
        signals = np.zeros(n, dtype=bool)
        
        # 找到所有反轉點
        reversal_indices = np.where(reversals)[0]
        
        # 在每個反轉點的下一根 K 線入場
        for rev_idx in reversal_indices:
            if rev_idx + 1 < n:
                signals[rev_idx + 1] = True
        
        print(f"\n⚡ 超快速入場: {np.sum(signals)} 個信號")
        
        return signals
    
    def get_color(self):
        return "red"
    
    def get_marker(self):
        return "^"


class SmartFastEntry(DetectionMethod):
    """
    智能快速入場
    
    結合多個條件來平衡速度和準確度：
    1. 在 1-2 根 K 線內確認
    2. 價格移動 0.1-0.2%（很小的確認）
    3. K 線形態確認（收盤價確認方向）
    4. 可選：成交量確認
    """
    
    def __init__(
        self,
        confirmation_candles=2,
        min_move_pct=0.001,  # 0.1%
        require_candle_close=True,
        require_volume=False,
        volume_threshold=1.0,
        name="Smart Fast Entry"
    ):
        super().__init__(name)
        self.confirmation_candles = confirmation_candles
        self.min_move_pct = min_move_pct
        self.require_candle_close = require_candle_close
        self.require_volume = require_volume
        self.volume_threshold = volume_threshold
    
    def detect(self, df):
        """智能快速入場檢測"""
        n = len(df)
        
        if 'reverse' not in df.columns:
            return np.zeros(n, dtype=bool)
        
        reversals = df['reverse'].values
        signals = np.zeros(n, dtype=bool)
        
        # 找到所有反轉點
        reversal_indices = np.where(reversals)[0]
        
        entry_count = 0
        
        for rev_idx in reversal_indices:
            # 檢查後續 K 線
            for offset in range(1, self.confirmation_candles + 1):
                check_idx = rev_idx + offset
                
                if check_idx >= n:
                    break
                
                if self._smart_confirm(df, rev_idx, check_idx):
                    signals[check_idx] = True
                    entry_count += 1
                    break
        
        print(f"\n🧠 智能快速入場: {entry_count} 個信號")
        
        return signals
    
    def _smart_confirm(self, df, reversal_idx, check_idx):
        """智能確認"""
        reversal_price = df['close'].iloc[reversal_idx]
        current_price = df['close'].iloc[check_idx]
        
        # 1. 價格移動確認
        price_change = abs(current_price - reversal_price) / reversal_price
        if price_change < self.min_move_pct:
            return False
        
        # 2. 判斷反轉方向
        lookback = 5
        start_idx = max(0, reversal_idx - lookback)
        end_idx = min(len(df), reversal_idx + lookback + 1)
        local_prices = df['close'].iloc[start_idx:end_idx].values
        
        is_peak = df['close'].iloc[reversal_idx] == np.max(local_prices)
        is_valley = df['close'].iloc[reversal_idx] == np.min(local_prices)
        
        # 3. K 線收盤價確認（可選）
        if self.require_candle_close:
            if 'open' in df.columns:
                open_price = df['open'].iloc[check_idx]
                close_price = df['close'].iloc[check_idx]
                
                # 波峰：需要陰線（收盤 < 開盤）
                if is_peak and close_price >= open_price:
                    return False
                
                # 波谷：需要陽線（收盤 > 開盤）
                if is_valley and close_price <= open_price:
                    return False
        
        # 4. 方向確認
        if is_peak and current_price >= reversal_price:
            return False
        if is_valley and current_price <= reversal_price:
            return False
        
        # 5. 成交量確認（可選）
        if self.require_volume and 'volume' in df.columns:
            volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()
            current_volume_ratio = df['volume'].iloc[check_idx] / volume_ma.iloc[check_idx]
            
            if current_volume_ratio < self.volume_threshold:
                return False
        
        return True
    
    def get_color(self):
        return "blue"
    
    def get_marker(self):
        return "D"


class AdaptiveFastEntry(DetectionMethod):
    """
    自適應快速入場
    
    根據市場波動自動調整確認閾值：
    - 高波動：需要更大的確認（0.2-0.3%）
    - 低波動：更小的確認即可（0.05-0.1%）
    """
    
    def __init__(
        self,
        confirmation_candles=2,
        base_move_pct=0.001,
        name="Adaptive Fast Entry"
    ):
        super().__init__(name)
        self.confirmation_candles = confirmation_candles
        self.base_move_pct = base_move_pct
    
    def detect(self, df):
        """自適應快速入場"""
        n = len(df)
        
        if 'reverse' not in df.columns:
            return np.zeros(n, dtype=bool)
        
        # 計算市場波動度
        volatility = df['close'].pct_change().rolling(window=20).std()
        median_vol = volatility.median()
        
        reversals = df['reverse'].values
        signals = np.zeros(n, dtype=bool)
        reversal_indices = np.where(reversals)[0]
        
        entry_count = 0
        
        for rev_idx in reversal_indices:
            # 根據當前波動度調整閾值
            current_vol = volatility.iloc[rev_idx]
            
            if current_vol < median_vol * 0.5:
                # 低波動：0.05%
                threshold = self.base_move_pct * 0.5
            elif current_vol < median_vol:
                # 正常波動：0.1%
                threshold = self.base_move_pct
            elif current_vol < median_vol * 2:
                # 高波動：0.2%
                threshold = self.base_move_pct * 2
            else:
                # 極高波動：0.3%
                threshold = self.base_move_pct * 3
            
            # 檢查後續 K 線
            for offset in range(1, self.confirmation_candles + 1):
                check_idx = rev_idx + offset
                
                if check_idx >= n:
                    break
                
                reversal_price = df['close'].iloc[rev_idx]
                current_price = df['close'].iloc[check_idx]
                price_change = abs(current_price - reversal_price) / reversal_price
                
                if price_change >= threshold:
                    signals[check_idx] = True
                    entry_count += 1
                    break
        
        print(f"\n🎯 自適應快速入場: {entry_count} 個信號")
        
        return signals
    
    def get_color(self):
        return "purple"
    
    def get_marker(self):
        return "s"
