#!/usr/bin/env python3
"""
數據提供者基類
"""

from abc import ABC, abstractmethod
import pandas as pd


class DataProvider(ABC):
    """數據提供者抽象基類"""
    
    @abstractmethod
    def get_data(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """
        獲取市場數據
        
        Args:
            symbol: 交易對符號
            interval: 時間間隔
            limit: 數據筆數
            
        Returns:
            包含 OHLCV 數據的 DataFrame
        """
        pass
