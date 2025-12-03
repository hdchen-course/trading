#!/usr/bin/env python3
"""
Detection Base Classes
檢測方法基類
"""

import numpy as np
import pandas as pd


class DetectionMethod:
    """
    檢測方法基類

    所有檢測方法都應該繼承此類並實現 detect() 方法
    """

    def __init__(self, name: str):
        """
        初始化檢測方法

        Args:
            name: 檢測方法的名稱
        """
        self.name = name

    def detect(self, df: pd.DataFrame) -> np.ndarray:
        """
        向量化檢測方法

        Args:
            df: 包含 OHLCV 數據和技術指標的 DataFrame

        Returns:
            信號數組，可以是以下兩種格式之一：

            1. 布爾數組（傳統格式，用於不區分信號類型的檢測器）：
               - True: 有信號
               - False: 無信號

            2. 整數數組（新格式，用於區分信號類型的檢測器，如 ML 模型）：
               - 0: 無信號
               - 1: 做空信號（A_turn）
               - 2: 做多信號（V_turn）
               - 或其他自定義的信號類型編碼

        注意：
            - 子類必須實現此方法
            - 返回數組的長度必須與 df 的長度相同
            - 對於新的檢測器，建議使用整數數組格式以支持多種信號類型
        """
        raise NotImplementedError("Subclasses must implement detect method")

    def get_color(self) -> str:
        """
        返回圖表中使用的顏色

        Returns:
            顏色字符串（如 'red', 'blue', '#FF0000'）
        """
        return "red"

    def get_marker(self) -> str:
        """
        返回圖表中使用的標記樣式

        Returns:
            標記字符串（如 'o', '^', 'v', 's'）
        """
        return "o"
