#!/usr/bin/env python3
"""
Detection Base Classes
檢測方法基類
"""


class DetectionMethod:
    """檢測方法基類"""

    def __init__(self, name):
        self.name = name

    def detect(self, df):
        """
        Vectorized detection method
        Returns boolean array indicating signal positions
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            numpy boolean array indicating signal positions
        """
        raise NotImplementedError("Subclasses must implement detect method")

    def get_color(self):
        """返回圖表中使用的顏色"""
        return "red"

    def get_marker(self):
        """返回圖表中使用的標記"""
        return "o"
