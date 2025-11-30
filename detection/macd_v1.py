#!/usr/bin/env python3
"""
MACD V1 反轉檢測器

基於 MACD 交叉點的機器學習檢測器：
- 檢測 MACD 黃金交叉/死亡交叉（MACD 線穿越信號線）
- 使用 Random Forest 模型預測交叉後的反轉概率
- Top 5 特徵，測試集準確率 56.01%

模型性能：
- 測試集準確率: 56.01%
- ROC AUC: 0.5808
- 特徵數: 5
"""

import numpy as np
import joblib
from pathlib import Path


class MACDV1Detector:
    """
    MACD V1 檢測器 - 基於 MACD 交叉點的機器學習預測

    檢測邏輯：
    1. 找出 MACD 黃金交叉/死亡交叉點（MACD 線穿越信號線）
    2. 使用 Random Forest 模型預測交叉後的反轉概率
    3. 在 [t+1] 生成信號
    """

    def __init__(
        self,
        model_path: str = "models/macd/macd_model_v1_rf_top5.pkl",
    ):
        """
        初始化檢測器

        Args:
            model_path: 模型路徑（默認使用 Top 5 特徵模型）
        """
        self.name = "MACD V1"

        # 載入模型
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]

        print(f"✅ 載入 MACD V1 模型:")
        print(f"   特徵數: {len(self.feature_columns)}")
        print(f"   特徵: {', '.join(self.feature_columns)}")

    def detect(self, df, prob_threshold: float = 0.5, debug: bool = False):
        """
        檢測 MACD 交叉點並預測反轉信號（向量化版本）

        Args:
            df: 包含價格和技術指標的 DataFrame
            prob_threshold: 概率閾值
            debug: 是否輸出調試信息

        Returns:
            買入信號數組（布爾值）
        """
        if len(df) < 2:
            return np.zeros(len(df), dtype=bool)

        # 檢測 MACD 交叉點（向量化）
        cross_indices = self._find_macd_crosses(df)

        # 初始化信號數組
        signals = np.zeros(len(df), dtype=bool)

        if len(cross_indices) == 0:
            return signals

        # 批量提取特徵（向量化）
        entry_indices = cross_indices + 1

        # 過濾掉超出範圍的索引
        valid_mask = entry_indices < len(df)
        cross_indices = cross_indices[valid_mask]
        entry_indices = entry_indices[valid_mask]

        if len(cross_indices) == 0:
            return signals

        # 批量提取特徵
        features_list = []
        valid_indices = []

        for i, (t_idx, entry_idx) in enumerate(zip(cross_indices, entry_indices)):
            features = self._extract_features(df, t_idx, entry_idx)
            if features is not None:
                features_list.append(features)
                valid_indices.append(entry_idx)

        if len(features_list) == 0:
            return signals

        # 批量預測（向量化）
        features_array = np.array(features_list)
        probas = self.model.predict_proba(features_array)[:, 1]

        # 生成信號
        signal_mask = probas >= prob_threshold
        signal_indices = np.array(valid_indices)[signal_mask]
        signals[signal_indices] = True

        predictions_count = len(signal_indices)

        # Debug 信息
        if len(cross_indices) > 0:
            print(f"   [MACD V1] 交叉點: {len(cross_indices)} 個")
            print(f"   [MACD V1] 預測信號: {predictions_count} 個")
            print(
                f"   [MACD V1] 信號率: {predictions_count/len(cross_indices)*100:.1f}%"
            )

        return signals

    def _find_macd_crosses(self, df):
        """
        找出 MACD 交叉點（向量化版本）

        條件：
        1. 黃金交叉: MACD 線從下方穿越信號線
           - 且 EMA100 > EMA50
        2. 死亡交叉: MACD 線從上方穿越信號線
           - 且 EMA100 < EMA50
        """
        required_cols = ["macd", "macd_signal", "ema_50", "ema_100"]
        if not all(col in df.columns for col in required_cols):
            return np.array([], dtype=int)

        macd = df["macd"].values
        signal = df["macd_signal"].values
        ema_50 = df["ema_50"].values
        ema_100 = df["ema_100"].values

        # 向量化計算交叉點
        # 黃金交叉: macd[t] < signal[t] 且 macd[t+1] > signal[t+1]
        golden_cross = (macd[:-1] < signal[:-1]) & (macd[1:] > signal[1:])

        # 死亡交叉: macd[t] > signal[t] 且 macd[t+1] < signal[t+1]
        death_cross = (macd[:-1] > signal[:-1]) & (macd[1:] < signal[1:])

        # 黃金交叉過濾條件: EMA100 > EMA50
        golden_filter = ema_100[:-1] > ema_50[:-1]

        # 死亡交叉過濾條件: EMA100 < EMA50
        death_filter = ema_100[:-1] < ema_50[:-1]

        # 應用過濾條件
        golden_cross_filtered = golden_cross & golden_filter
        death_cross_filtered = death_cross & death_filter

        # 合併兩種交叉
        crosses = golden_cross_filtered | death_cross_filtered

        # 獲取交叉點索引
        cross_indices = np.where(crosses)[0]

        return cross_indices

    def _extract_features(self, df, t_idx: int, entry_idx: int):
        """
        提取特徵向量

        特徵格式：
        - xxx_at_t: 從 xxx 欄位取 [t] 的值
        - xxx_at_entry: 從 xxx 欄位取 [entry] 的值

        Args:
            df: DataFrame
            t_idx: 交叉點索引 [t]
            entry_idx: 入場點索引 [t+1]

        Returns:
            特徵列表，如果缺少欄位則返回 None
        """
        features = []

        for feature_name in self.feature_columns:
            try:
                if feature_name.endswith("_at_t"):
                    # 從 [t] 取值
                    base_col = feature_name[:-5]  # 移除 "_at_t"
                    if base_col not in df.columns:
                        # print(f"警告: 缺少欄位 {base_col}")
                        return None
                    value = df.iloc[t_idx][base_col]

                elif feature_name.endswith("_at_entry"):
                    # 從 [entry] 取值
                    base_col = feature_name[:-9]  # 移除 "_at_entry"
                    if base_col not in df.columns:
                        # print(f"警告: 缺少欄位 {base_col}")
                        return None
                    value = df.iloc[entry_idx][base_col]

                else:
                    # 未知的特徵格式
                    # print(f"警告: 未知的特徵格式 {feature_name}")
                    return None

                # 檢查是否為 NaN
                if np.isnan(value):
                    return None

                features.append(value)

            except (KeyError, IndexError):
                # print(f"錯誤: 提取特徵 {feature_name} 時發生錯誤")
                return None

        return features

    def get_color(self):
        """返回可視化顏色"""
        return "cyan"

    def get_marker(self):
        """返回可視化標記"""
        return "^"  # 三角形
