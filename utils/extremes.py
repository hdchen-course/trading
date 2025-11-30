"""
局部極值檢測工具

提供統一的局部極值（峰值和谷值）檢測功能
用於反轉點檢測的第一階段

⚠️ 重要：
- 這個模組只看「過去數據」，適用於實時交易和回測
- 使用 prominence（突出度）來過濾不顯著的極值點
- 找出的是「候選極值點」，不是真實反轉點
- 真實反轉點用 detection/peak_valley.py（可以看未來數據）來標記
"""

import numpy as np
import pandas as pd
from typing import Tuple


def find_local_extremes(
    prices: np.ndarray, 
    window: int = 10,
    prominence_window: int = None,
    prominence_factor: float = 0.3
) -> Tuple[set, set, set]:
    """
    尋找局部極值點（峰和谷）- 只看過去數據，使用 prominence 過濾
    
    使用滾動窗口方法，只看過去的數據來判斷當前是否為顯著的局部極值。
    類似 peak_valley.py 的方法，但只看過去數據。
    
    Args:
        prices: 價格序列（numpy array）
        window: 回看窗口大小（默認 10，對應 peak_valley distance=20 的一半）
                用於判斷是否為局部最高/最低點
        prominence_window: 用於計算 prominence 的窗口大小（默認 500，約 5 天）
                          如果為 None，自動設為 min(500, len(prices)//4)
        prominence_factor: 突出度因子（默認 0.3，平衡召回率和精準度）
                          極值點必須比局部平均價格高/低至少 prominence_factor * global_std
    
    Returns:
        extremes: 所有極值點的集合（峰值 + 谷值）
        peaks: 峰值點的集合
        valleys: 谷值點的集合
    
    Example:
        >>> prices = np.array([100, 105, 110, 108, 102, 98, 103, 107])
        >>> extremes, peaks, valleys = find_local_extremes(prices, window=10)
        >>> print(f"峰值: {peaks}, 谷值: {valleys}")
    
    注意：
        - 只看 [idx-window, idx] 的數據，不看未來
        - 使用滾動標準差的中位數來計算全局 prominence（類似 peak_valley.py）
        - 不是每個點都會是極值，只有顯著的才會被標記
    """
    peaks_list = []
    valleys_list = []
    
    # 計算 prominence（類似 peak_valley.py 的方法）
    if prominence_window is None:
        prominence_window = min(200, len(prices) // 4)
        if prominence_window < 200:
            prominence_window = len(prices)
    
    # 計算滾動標準差的中位數作為 prominence
    rolling_std = pd.Series(prices).rolling(window=prominence_window, min_periods=1).std()
    prominence = rolling_std.median() * prominence_factor
    
    # 從 window 開始遍歷（前面的點沒有足夠歷史數據）
    for idx in range(window, len(prices)):
        # 只看過去 window+1 個點（包含當前點）
        local_prices = prices[idx - window : idx + 1]
        current_price = prices[idx]
        
        # 計算局部統計量（使用過去 window 個點，不包含當前點）
        past_prices = prices[idx - window : idx]
        local_mean = np.mean(past_prices)
        
        # 檢查是否為顯著的局部峰值
        # 條件：1) 是局部最高點  2) 比局部平均高出至少 prominence
        if current_price == np.max(local_prices):
            if current_price > local_mean + prominence:
                peaks_list.append(idx)
        
        # 檢查是否為顯著的局部谷值
        # 條件：1) 是局部最低點  2) 比局部平均低至少 prominence
        if current_price == np.min(local_prices):
            if current_price < local_mean - prominence:
                valleys_list.append(idx)
    
    # 轉換為集合以便快速查找
    peaks = set(peaks_list)
    valleys = set(valleys_list)
    extremes = peaks | valleys
    
    return extremes, peaks, valleys


def is_local_peak(prices: np.ndarray, idx: int, window: int = 5) -> bool:
    """
    檢查指定索引是否為局部峰值 - 只看過去數據
    
    Args:
        prices: 價格序列
        idx: 要檢查的索引
        window: 回看窗口大小（默認 5）
    
    Returns:
        True 如果是局部峰值，否則 False
    
    注意：
        - 只看 [idx-window, idx] 的數據，不看未來
        - 適用於實時交易和回測
    """
    if idx < window:
        return False
    
    local_prices = prices[idx - window : idx + 1]
    return prices[idx] == np.max(local_prices)


def is_local_valley(prices: np.ndarray, idx: int, window: int = 5) -> bool:
    """
    檢查指定索引是否為局部谷值 - 只看過去數據
    
    Args:
        prices: 價格序列
        idx: 要檢查的索引
        window: 回看窗口大小（默認 5）
    
    Returns:
        True 如果是局部谷值，否則 False
    
    注意：
        - 只看 [idx-window, idx] 的數據，不看未來
        - 適用於實時交易和回測
    """
    if idx < window:
        return False
    
    local_prices = prices[idx - window : idx + 1]
    return prices[idx] == np.min(local_prices)
