"""
ML Reversal Detection
機器學習反轉檢測器

使用兩階段檢測：
Stage 1: 局部極值檢測（高召回率）
Stage 2: RandomForest 模型預測（機器學習過濾）

模型特徵：
- rsi, macd, macd_histogram, bb_position, volume_ratio
- price_momentum_5, price_momentum_10
- volatility_10, volume_trend

訓練數據：2023-2025 (2年)
驗證數據：2025 (未見過的數據)
"""

import numpy as np
import pandas as pd
import joblib
import os
from utils.indicators import calculate_technical_indicators, analyze_cumulative_metrics
from utils.extremes import find_local_extremes
from .base import DetectionMethod


class MLReversalDetector(DetectionMethod):
    """
    機器學習反轉檢測器
    
    使用 RandomForest 模型預測反轉點
    """
    
    def __init__(
        self, 
        distance: int = 20, 
        window_size: int = 16,
        model_path: str = "models/reversal/reversal_model_v2.pkl"
    ):
        """
        初始化檢測器
        
        Args:
            distance: 局部極值的最小距離
            window_size: 滾動窗口大小（用於計算指標）
            model_path: 模型檔案路徑
        """
        super().__init__("ML Reversal Detection")
        self.distance = distance
        self.window_size = window_size
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """載入訓練好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"找不到模型檔案: {self.model_path}\n"
                f"請先執行訓練: python model_training/reversal/train_ml_reversal_v2.py"
            )
        
        self.model = joblib.load(self.model_path)
        print(f"✅ ML Reversal 模型載入成功: {self.model_path}")
    

    
    def _extract_features_with_context(
        self, 
        df: pd.DataFrame, 
        entry_idx: int, 
        reversal_idx: int,
        analysis_df: pd.DataFrame
    ) -> dict:
        """
        提取入場點的特徵，包含反轉點的上下文信息
        
        與訓練時的特徵提取保持一致（Top 10 特徵）
        
        Args:
            df: 包含所有指標的 DataFrame（已調用 analyze_cumulative_metrics）
            entry_idx: 入場點索引 [t+1] 或 [t+2]
            reversal_idx: 反轉點索引 [t]
            analysis_df: 與 df 相同（為了向後兼容保留參數）
            
        Returns:
            特徵字典（Top 10 特徵）
        """
        if entry_idx < 10 or reversal_idx < 10:
            return None
        
        # 現在 df 已經包含所有指標，直接使用索引
        reversal_data = df.iloc[reversal_idx]
        entry_data = df.iloc[entry_idx]
        
        # 檢查是否有 NaN（前面的數據可能無法計算）
        if pd.isna(reversal_data['price_2nd']) or pd.isna(entry_data['price_2nd']):
            return None
        
        # Top 10 特徵（與訓練時完全一致）
        features = {
            # 1. price_change (18.1%)
            'price_change': (entry_data['close'] - reversal_data['close']) / reversal_data['close'],
            
            # 2. bb_position_at_t (13.8%)
            'bb_position_at_t': reversal_data['bb_position'],
            
            # 3. rsi_change (11.2%)
            'rsi_change': entry_data['rsi'] - reversal_data['rsi'],
            
            # 4. macd_histogram_at_t (12.8%)
            'macd_histogram_at_t': reversal_data['macd_histogram'],
            
            # 5. macd_histogram_at_entry (9.8%)
            'macd_histogram_at_entry': entry_data['macd_histogram'],
            
            # 6. rsi_at_t (9.9%)
            'rsi_at_t': reversal_data['rsi'],
            
            # 7. macd_at_entry (6.4%)
            'macd_at_entry': entry_data['macd'],
            
            # 8. price_2nd_at_entry (6.9%)
            'price_2nd_at_entry': entry_data['price_2nd'],
            
            # 9. macd_at_t (5.7%)
            'macd_at_t': reversal_data['macd'],
            
            # 10. volatility_10 (5.5%)
            'volatility_10': df.iloc[entry_idx - 10:entry_idx]['close'].std() / df.iloc[entry_idx - 10:entry_idx]['close'].mean(),
        }
        
        return features
    
    def detect(self, df: pd.DataFrame) -> np.ndarray:
        """
        檢測反轉點（符合 DetectionMethod 接口）
        
        兩階段檢測：
        Stage 1: 找局部極值（高召回率，覆蓋 99.9% 真實反轉點）
        Stage 2: ML 模型檢查極值點後 1-2 根K線是否符合入場條件
        
        Args:
            df: 包含 OHLCV 數據和累計指標的 DataFrame
                必須包含: rsi, macd, bb_position, price_2nd 等指標
            
        Returns:
            布爾數組，True 表示該點為入場點
        """
        n = len(df)
        if n < self.window_size + self.distance:
            return np.zeros(n, dtype=bool)
        
        # 檢查必要的欄位是否存在
        required_cols = ['rsi', 'macd', 'macd_histogram', 'bb_position', 'price_2nd']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Please call calculate_technical_indicators() and analyze_cumulative_metrics() before detection."
            )
        
        # 直接使用傳入的 df（已包含所有指標）
        analysis_df = df
        
        # Stage 1: 找局部極值
        prices = df['close'].values
        extremes, peaks, valleys = find_local_extremes(prices, distance=self.distance)
        
        # Stage 2: ML 模型預測 - 檢查極值點後 1-2 根K線
        signals = np.zeros(n, dtype=bool)
        
        # Top 10 特徵順序（必須與訓練時一致）
        feature_columns = [
            "price_change",
            "bb_position_at_t",
            "rsi_change",
            "macd_histogram_at_t",
            "macd_histogram_at_entry",
            "rsi_at_t",
            "macd_at_entry",
            "price_2nd_at_entry",
            "macd_at_t",
            "volatility_10",
        ]
        
        # 為每個極值點檢查後續 1-2 根K線
        for extreme_idx in extremes:
            # 極值點必須在有效範圍內
            if extreme_idx < self.window_size:
                continue
            
            # 檢查極值點後的 1-2 根K線
            for offset in [1, 2]:
                check_idx = extreme_idx + offset
                
                # 確保索引有效
                if check_idx >= n:
                    continue
                
                # 提取該點的特徵（包含反轉點上下文）
                # 現在 analysis_df 就是 df，不需要索引映射
                features = self._extract_features_with_context(
                    df, check_idx, extreme_idx, df
                )
                
                if features is None:
                    continue
                
                # 準備特徵向量
                try:
                    feature_vector = np.array([features[col] for col in feature_columns]).reshape(1, -1)
                    
                    # 檢查是否有 NaN
                    if np.isnan(feature_vector).any():
                        continue
                    
                    # 模型預測
                    prediction = self.model.predict(feature_vector)[0]
                    
                    # 預測為入場點（1=A_turn, 2=V_turn）
                    if prediction in [1, 2]:
                        # 標記入場點
                        signals[check_idx] = True
                        break  # 找到確認就停止檢查後續點
                        
                except Exception as e:
                    # 跳過有問題的點
                    continue
        
        return signals
    
    def get_color(self):
        """返回檢測方法的顏色（用於可視化）"""
        return "purple"
