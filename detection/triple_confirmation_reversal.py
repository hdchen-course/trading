"""
Triple Confirmation Reversal Detection
三重確認反轉檢測器

使用兩階段檢測：
Stage 1: 局部極值檢測（高召回率）
Stage 2: Price_2nd AND Vol_2nd AND MACD_2nd 過濾（高精確度 81.2%）

精確度：81.2%
覆蓋率：40.4%
每天約1.2個訊號

優化說明：
- 向量化處理：直接遍歷極值點而非所有點，減少循環次數
- 索引映射：正確處理 df 和 analysis_df 長度不同的問題
  * analysis_df[0] 對應 df[window_size - 1]
  * 映射公式：analysis_idx = df_idx - (window_size - 1)
"""

import numpy as np
import pandas as pd
from utils.indicators import calculate_technical_indicators, analyze_cumulative_metrics
from utils.extremes import find_local_extremes
from .base import DetectionMethod


class TripleConfirmationReversalDetector(DetectionMethod):
    """
    三重確認反轉檢測器
    
    使用 Price_2nd, Vol_2nd, MACD_2nd 三個二階導數特徵的符號改變
    來確認反轉點，達到 81.2% 的精確度
    """
    
    def __init__(self, distance: int = 20, window_size: int = 16):
        """
        初始化檢測器
        
        Args:
            distance: 局部極值的最小距離
            window_size: 滾動窗口大小（用於計算指標）
        """
        super().__init__("Triple Confirmation Reversal")
        self.distance = distance
        self.window_size = window_size
    

    
    def detect(self, df: pd.DataFrame) -> np.ndarray:
        """
        檢測反轉點（符合 DetectionMethod 接口）- 向量化優化版本
        
        Args:
            df: 包含 OHLCV 數據和累計指標的 DataFrame
                必須包含: price_2nd, vol_2nd, macd_2nd 等二階導數
            
        Returns:
            布爾數組，True 表示該點為反轉點
        """
        n = len(df)
        if n < self.window_size + self.distance:
            return np.zeros(n, dtype=bool)
        
        # 檢查必要的欄位是否存在
        required_cols = ['price_2nd', 'vol_2nd', 'macd_2nd']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Please call analyze_cumulative_metrics() before detection."
            )
        
        # 直接使用傳入的 df（已包含二階導數）
        analysis_df = df
        
        # Stage 1: 找局部極值
        prices = df['close'].values
        extremes, peaks, valleys = find_local_extremes(prices, distance=self.distance)
        
        # 創建極值標記數組（用於快速查找）
        is_extreme = np.zeros(n, dtype=bool)
        is_extreme[list(extremes)] = True
        
        # 獲取二階導數符號數組
        price_2nd_sign = np.sign(analysis_df['price_2nd'].values)
        vol_2nd_sign = np.sign(analysis_df['vol_2nd'].values)
        macd_2nd_sign = np.sign(analysis_df['macd_2nd'].values)
        
        # Stage 2: 三重確認過濾 - 向量化版本
        signals = np.zeros(n, dtype=bool)
        
        # 向量化處理：為每個極值點檢查後續1-2個點的符號變化
        for extreme_idx in extremes:
            # 極值點必須在有效範圍內
            if extreme_idx < self.window_size - 1:
                continue
            
            # 現在 analysis_df 與 df 長度相同，直接使用索引
            analysis_extreme_idx = extreme_idx
            
            # 檢查後續1-2個點
            for offset in [1, 2]:
                current_idx = extreme_idx + offset
                
                # 確保當前點在有效範圍內
                if current_idx >= n:
                    continue
                
                analysis_current_idx = current_idx
                
                # 檢查 analysis 索引有效性
                if analysis_current_idx >= len(analysis_df):
                    continue
                
                # 向量化檢查三重確認（符號改變）
                price_changed = (price_2nd_sign[analysis_extreme_idx] != 
                               price_2nd_sign[analysis_current_idx])
                vol_changed = (vol_2nd_sign[analysis_extreme_idx] != 
                             vol_2nd_sign[analysis_current_idx])
                macd_changed = (macd_2nd_sign[analysis_extreme_idx] != 
                              macd_2nd_sign[analysis_current_idx])
                
                # 三個條件都滿足則標記為反轉點
                if price_changed and vol_changed and macd_changed:
                    signals[extreme_idx] = True
                    break  # 找到確認就停止檢查後續點
        
        return signals
    
    def get_color(self):
        """返回檢測方法的顏色（用於可視化）"""
        return "red"
