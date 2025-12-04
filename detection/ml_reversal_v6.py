"""
ML Reversal V6 Detection
機器學習反轉檢測器 V6

基於 2024 年數據訓練，使用 RandomForest 和 Top 10 重要特徵
使用極值檢測 + 真實反轉點標籤

訓練邏輯：
- 使用 find_local_extremes 找出極值點 [t]
- 入場點為 [t+1]
- 正樣本：真實反轉點
- 負樣本：虧損超過 0.18% 且通過嚴格過濾條件
- 樣本平衡：1:1

Top 10 重要特徵（從 49 個中選出）：
1. sar_distance_pct_change_t_minus_1_to_t
2. avg_amplitude_t
3. sar_distance_pct_change_t_to_entry
4. avg_amplitude_entry
5. volatility_t
6. macd_histogram_change_t_to_entry
7. macd_2nd_change_t_minus_1_to_t
8. volatility_entry
9. amplitude_t
10. volume_change_t

檢測過濾條件（在 ML 預測前）：
通用條件：
- |sar_distance_pct| > 1.3%（價格與 SAR 距離足夠遠）
- [t-1] -> [t] 價格變化 < 0.5%（避免劇烈波動）

Peak（做空）：
- peak 在 SAR 上方
- peak 在 bb_middle 上方
- entry 價格 < peak 價格（確認下跌反轉開始）

Valley（做多）：
- valley 在 SAR 下方
- valley 在 bb_middle 下方
- entry 價格 > valley 價格（確認上漲反轉開始）

模型：
- 算法: RandomForest
- 特徵數: 10
- 測試準確率: 79.55%
- ROC AUC: 0.8784
- 正樣本召回率: 78%

回測結果（350天，2024-12-19 至 2025-12-04）：
- 交易次數: 95 次
- 總報酬: 11.62%
- 勝率: 33.68% ❌ 表現不佳
"""

import numpy as np
import pandas as pd
import joblib
import os
from .base import DetectionMethod
from utils.extremes import find_local_extremes


class MLReversalV6Detector(DetectionMethod):
    """
    機器學習反轉檢測器 V6

    使用 RandomForest 模型預測極值點後的反轉入場點
    """

    def __init__(
        self,
        model_path: str = "models/reversal/reversal_model_v6_rf.pkl",
        prob_threshold: float = 0.5,
    ):
        """
        初始化檢測器

        Args:
            model_path: 模型檔案路徑
            prob_threshold: 預測概率閾值（0-1）
        """
        super().__init__("ML Reversal V6 Detection")
        self.model_path = model_path
        self.prob_threshold = prob_threshold
        self.model = None
        self.feature_columns = None
        self._load_model()

    def _load_model(self):
        """載入訓練好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"找不到模型檔案: {self.model_path}\n"
                f"請先執行訓練: python model_training/reversal/train_ml_reversal_v6.py"
            )

        self.model = joblib.load(self.model_path)

        # 載入特徵列表
        feature_list_path = self.model_path.replace(".pkl", "_features.txt")
        if os.path.exists(feature_list_path):
            with open(feature_list_path, "r") as f:
                self.feature_columns = [line.strip() for line in f.readlines()]
        else:
            raise FileNotFoundError(f"找不到特徵列表檔案: {feature_list_path}")

        print(f"✅ ML Reversal V6 模型載入成功")
        print(f"   模型路徑: {self.model_path}")
        print(f"   特徵數: {len(self.feature_columns)}")
        print(f"   概率閾值: {self.prob_threshold}")

    def _extract_features(self, df: pd.DataFrame, t_idx: int, entry_idx: int) -> dict:
        """
        提取 [t-1], [t] 和 [entry] 的特徵

        Args:
            df: 包含所有指標的 DataFrame
            t_idx: [t] 極值點的索引
            entry_idx: [entry] 入場點的索引

        Returns:
            特徵字典（Top 10 重要特徵）
        """
        # 確保索引有效
        if t_idx < 1 or entry_idx >= len(df):
            return None

        features = {}

        try:
            # 1. sar_distance_pct 變化率
            sar_dist_t_minus_1 = df.iloc[t_idx - 1]["sar_distance_pct"]
            sar_dist_t = df.iloc[t_idx]["sar_distance_pct"]
            sar_dist_entry = df.iloc[entry_idx]["sar_distance_pct"]

            if abs(sar_dist_t_minus_1) > 1e-10:
                features["sar_distance_pct_change_t_minus_1_to_t"] = abs(
                    sar_dist_t - sar_dist_t_minus_1
                ) / abs(sar_dist_t_minus_1)
            else:
                features["sar_distance_pct_change_t_minus_1_to_t"] = 0.0

            if abs(sar_dist_t) > 1e-10:
                features["sar_distance_pct_change_t_to_entry"] = abs(
                    sar_dist_entry - sar_dist_t
                ) / abs(sar_dist_t)
            else:
                features["sar_distance_pct_change_t_to_entry"] = 0.0

            # 2. macd_histogram 變化率
            macd_hist_t = df.iloc[t_idx]["macd_histogram"]
            macd_hist_entry = df.iloc[entry_idx]["macd_histogram"]

            if abs(macd_hist_t) > 1e-10:
                features["macd_histogram_change_t_to_entry"] = abs(
                    macd_hist_entry - macd_hist_t
                ) / abs(macd_hist_t)
            else:
                features["macd_histogram_change_t_to_entry"] = 0.0

            # 3. macd_2nd 變化率
            macd_2nd_t_minus_1 = df.iloc[t_idx - 1]["macd_2nd"]
            macd_2nd_t = df.iloc[t_idx]["macd_2nd"]

            if abs(macd_2nd_t_minus_1) > 1e-10:
                features["macd_2nd_change_t_minus_1_to_t"] = abs(
                    macd_2nd_t - macd_2nd_t_minus_1
                ) / abs(macd_2nd_t_minus_1)
            else:
                features["macd_2nd_change_t_minus_1_to_t"] = 0.0

            # 4. 原始值特徵 [t] (Top 10 中的 [t] 特徵)
            features["avg_amplitude_t"] = df.iloc[t_idx]["avg_amplitude"]
            features["volatility_t"] = df.iloc[t_idx]["volatility"]
            features["amplitude_t"] = df.iloc[t_idx]["amplitude"]
            features["volume_change_t"] = df.iloc[t_idx]["volume_change"]

            # 5. 原始值特徵 [entry] (Top 10 中的 [entry] 特徵)
            features["avg_amplitude_entry"] = df.iloc[entry_idx]["avg_amplitude"]
            features["volatility_entry"] = df.iloc[entry_idx]["volatility"]

            # 檢查是否有 NaN
            for key, value in features.items():
                if pd.isna(value):
                    return None

            return features

        except (KeyError, IndexError):
            return None

    def detect(self, df: pd.DataFrame) -> np.ndarray:
        """
        檢測反轉點

        檢測策略：
        1. 使用 find_local_extremes 找出極值點 [t]
        2. 對每個極值點，使用 [t+1] 作為入場點
        3. 使用 ML 模型預測是否為好的入場點

        Args:
            df: 包含 OHLCV 數據和技術指標的 DataFrame

        Returns:
            整數數組：
            - 0: 無信號
            - 1: 做空信號（peak 後的入場點）
            - 2: 做多信號（valley 後的入場點）
        """
        n = len(df)
        if n < 20:
            return np.zeros(n, dtype=int)

        signals = np.zeros(n, dtype=int)

        # 1. 找出極值點
        prices = df["close"].values
        extremes, peaks, valleys = find_local_extremes(
            prices, window=20, prominence_factor=0.3
        )

        if len(peaks) == 0 and len(valleys) == 0:
            return signals

        # 統計
        candidates_count = 0
        peak_predictions = 0
        valley_predictions = 0

        # Debug: 收集預測概率
        peak_probas = []
        valley_probas = []

        # 2. 處理 peak（做空信號）
        for t_idx in sorted(peaks):
            entry_idx = t_idx + 1

            # 確保入場點有效
            if entry_idx >= n:
                continue

            t_price = df.iloc[t_idx]["close"]
            entry_price = df.iloc[entry_idx]["close"]
            bb_middle = df.iloc[t_idx]["bb_middle"]
            sar_distance_pct = df.iloc[t_idx]["sar_distance_pct"]
            sar_value = df.iloc[t_idx]["sar"]

            # 計算 [t-1] -> [t] 的價格變化
            if t_idx >= 1:
                t_minus_1_price = df.iloc[t_idx - 1]["close"]
                price_change_t = abs((t_price - t_minus_1_price) / t_minus_1_price)
            else:
                price_change_t = 0.0

            # 過濾條件（在 ML 預測前）：
            # 1. |sar_distance_pct| > 1.3%
            if abs(sar_distance_pct) <= 1.3:
                continue

            # 2. peak 在 SAR 上方
            if t_price <= sar_value:
                continue

            # 3. peak 在 bb_middle 上方
            if t_price <= bb_middle:
                continue

            # 4. [t-1] -> [t] 價格變化 < 0.5%
            if price_change_t >= 0.005:
                continue

            # 5. entry < t_price（確認下跌反轉開始）
            if entry_price >= t_price:
                continue

            candidates_count += 1

            # 提取特徵
            features = self._extract_features(df, t_idx, entry_idx)

            if features is None:
                continue

            # 準備特徵向量（按照訓練時的順序）
            try:
                feature_vector = np.array(
                    [features[col] for col in self.feature_columns]
                ).reshape(1, -1)

                # 檢查是否有 NaN
                if np.isnan(feature_vector).any():
                    continue

                # 模型預測
                proba = self.model.predict_proba(feature_vector)[0]
                peak_probas.append(proba[1])

                # 使用概率閾值
                if proba[1] >= self.prob_threshold:
                    signals[entry_idx] = 1  # 做空信號
                    peak_predictions += 1

            except Exception:
                # 跳過有問題的點
                continue

        # 3. 處理 valley（做多信號）
        for t_idx in sorted(valleys):
            entry_idx = t_idx + 1

            # 確保入場點有效
            if entry_idx >= n:
                continue

            t_price = df.iloc[t_idx]["close"]
            entry_price = df.iloc[entry_idx]["close"]
            bb_middle = df.iloc[t_idx]["bb_middle"]
            sar_distance_pct = df.iloc[t_idx]["sar_distance_pct"]
            sar_value = df.iloc[t_idx]["sar"]

            # 計算 [t-1] -> [t] 的價格變化
            if t_idx >= 1:
                t_minus_1_price = df.iloc[t_idx - 1]["close"]
                price_change_t = abs((t_price - t_minus_1_price) / t_minus_1_price)
            else:
                price_change_t = 0.0

            # 過濾條件（在 ML 預測前）：
            # 1. |sar_distance_pct| > 1.3%
            if abs(sar_distance_pct) <= 1.3:
                continue

            # 2. valley 在 SAR 下方
            if t_price >= sar_value:
                continue

            # 3. valley 在 bb_middle 下方
            if t_price >= bb_middle:
                continue

            # 4. [t-1] -> [t] 價格變化 < 0.5%
            if price_change_t >= 0.005:
                continue

            # 5. entry > t_price（確認上漲反轉開始）
            if entry_price <= t_price:
                continue

            candidates_count += 1

            # 提取特徵
            features = self._extract_features(df, t_idx, entry_idx)

            if features is None:
                continue

            # 準備特徵向量（按照訓練時的順序）
            try:
                feature_vector = np.array(
                    [features[col] for col in self.feature_columns]
                ).reshape(1, -1)

                # 檢查是否有 NaN
                if np.isnan(feature_vector).any():
                    continue

                # 模型預測
                proba = self.model.predict_proba(feature_vector)[0]
                valley_probas.append(proba[1])

                # 使用概率閾值
                if proba[1] >= self.prob_threshold:
                    signals[entry_idx] = 2  # 做多信號
                    valley_predictions += 1

            except Exception:
                # 跳過有問題的點
                continue

        # Debug 信息
        if candidates_count > 0:
            print(f"   [ML V6] 極值點: {len(peaks)} 個 peak + {len(valleys)} 個 valley")
            print(f"   [ML V6] 候選入場點: {candidates_count} 個")
            print(
                f"   [ML V6] 預測信號: {peak_predictions} 個做空 + {valley_predictions} 個做多"
            )

            # 顯示概率分佈
            if len(peak_probas) > 0:
                print(
                    f"   [ML V6] Peak 概率: min={min(peak_probas):.3f}, max={max(peak_probas):.3f}, avg={np.mean(peak_probas):.3f}"
                )
            if len(valley_probas) > 0:
                print(
                    f"   [ML V6] Valley 概率: min={min(valley_probas):.3f}, max={max(valley_probas):.3f}, avg={np.mean(valley_probas):.3f}"
                )

        return signals

    def get_color(self):
        """返回檢測方法的顏色（用於可視化）"""
        return "purple"
