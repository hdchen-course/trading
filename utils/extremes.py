"""
局部極值檢測工具

提供統一的局部極值（峰值和谷值）檢測功能
用於反轉點檢測的第一階段
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Tuple


def find_local_extremes(
    prices: np.ndarray, 
    distance: int = 20
) -> Tuple[set, set, set]:
    """
    尋找局部極值點（峰和谷）
    
    使用 scipy.signal.find_peaks 檢測價格序列中的局部極值點。
    這些極值點包含真實反轉點和假性反轉點，需要後續階段過濾。
    
    Args:
        prices: 價格序列（numpy array）
        distance: 局部極值的最小距離（默認 20）
                 表示兩個極值點之間至少相隔 20 個數據點
    
    Returns:
        extremes: 所有極值點的集合（峰值 + 谷值）
        peaks: 峰值點的集合
        valleys: 谷值點的集合
    
    Example:
        >>> prices = np.array([100, 105, 110, 108, 102, 98, 103, 107])
        >>> extremes, peaks, valleys = find_local_extremes(prices, distance=2)
        >>> print(f"峰值: {peaks}, 谷值: {valleys}")
    """
    # 找峰值（局部最大值）
    peaks_idx, _ = find_peaks(prices, distance=distance)
    
    # 找谷值（局部最小值）
    # 通過反轉價格序列來找谷值
    valleys_idx, _ = find_peaks(-prices, distance=distance)
    
    # 轉換為集合以便快速查找
    peaks = set(peaks_idx)
    valleys = set(valleys_idx)
    extremes = peaks | valleys
    
    return extremes, peaks, valleys


def is_local_peak(prices: np.ndarray, idx: int, window: int = 5) -> bool:
    """
    檢查指定索引是否為局部峰值
    
    Args:
        prices: 價格序列
        idx: 要檢查的索引
        window: 檢查窗口大小（前後各 window 個點）
    
    Returns:
        True 如果是局部峰值，否則 False
    """
    if idx < window or idx >= len(prices) - window:
        return False
    
    local_prices = prices[idx - window : idx + window + 1]
    return prices[idx] == np.max(local_prices)


def is_local_valley(prices: np.ndarray, idx: int, window: int = 5) -> bool:
    """
    檢查指定索引是否為局部谷值
    
    Args:
        prices: 價格序列
        idx: 要檢查的索引
        window: 檢查窗口大小（前後各 window 個點）
    
    Returns:
        True 如果是局部谷值，否則 False
    """
    if idx < window or idx >= len(prices) - window:
        return False
    
    local_prices = prices[idx - window : idx + window + 1]
    return prices[idx] == np.min(local_prices)
