"""
ML Reversal V3 Detection
機器學習反轉檢測器 V3

基於 2024 年數據訓練，使用 XGBoost 和 Top 15 特徵
使用極值檢測 + 止損止盈邏輯生成標籤

訓練邏輯：
- 使用 find_local_extremes 找出極值點 [t]
- 入場點為 [t+1] 或 [t+2]
- 止損：0.18%
- 止盈：2.5小時內 >= 0.4%
- 正樣本：達到止盈
- 負樣本：觸及止損
- 樣本平衡：1:1

Top 15 特徵：
1. ema_50_at_entry - 入場點 EMA50
2. bb_upper_at_t - 極值點布林帶上軌
3. price_change_pct_at_entry - 入場點價格變化
4. low_at_t - 極值點最低價
5. bb_upper_at_entry - 入場點布林帶上軌
6. bb_middle_at_entry - 入場點布林帶中軌
7. sar_at_entry - 入場點 SAR
8. avg_amplitude_at_entry - 入場點平均振幅
9. bb_lower_at_entry - 入場點布林帶下軌
10. macd_signal_at_entry - 入場點 MACD 信號線
11. atr_at_entry - 入場點 ATR
12. volume_sma_at_t - 極值點成交量均線
13. bb_middle_at_t - 極值點布林帶中軌
14. open_at_t - 極值點開盤價
15. low_at_entry - 入場點最低價

模型：
- 算法: XGBoost
- 特徵數: 15
- 測試準確率: 65.17%
- ROC AUC: 0.7083
"""

import numpy as np
import pandas as pd
import joblib
import os
from .base import DetectionMethod
from utils.extremes import find_local_extremes


class MLReversalV3Detector(DetectionMethod):
    """
    機器學習反轉檢測器 V3

    使用 XGBoost 模型預測極值點後的反轉入場點
    """

    def __init__(
        self,
        model_path: str = "models/reversal/reversal_model_v3_top15.pkl",
        prob_threshold: float = 0.5,
    ):
        """
        初始化檢測器

        Args:
            model_path: 模型檔案路徑
            prob_threshold: 預測概率閾值（0-1）
        """
        super().__init__("ML Reversal V3 Detection")
        self.model_path = model_path
        self.prob_threshold = prob_threshold
        self.model_data = None
        self.model = None
        self.feature_columns = None
        self._load_model()

    def _load_model(self):
        """載入訓練好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"找不到模型檔案: {self.model_path}\n"
                f"請先執行訓練: python model_training/reversal/train_ml_reversal_v3.py"
            )

        self.model_data = joblib.load(self.model_path)
        self.model = self.model_data["model"]
        self.feature_columns = self.model_data["feature_columns"]

        print(f"✅ ML Reversal V3 模型載入成功")
        print(f"   模型路徑: {self.model_path}")
        print(f"   特徵數: {len(self.feature_columns)}")
        print(f"   概率閾值: {self.prob_threshold}")

    def _extract_features(self, df: pd.DataFrame, t_idx: int, entry_idx: int) -> dict:
        """
        提取 [t] 和 [entry] 的特徵

        Args:
            df: 包含所有指標的 DataFrame
            t_idx: [t] 極值點的索引
            entry_idx: [entry] 入場點的索引

        Returns:
            特徵字典（Top 15 特徵）
        """
        # 確保索引有效
        if t_idx < 0 or entry_idx >= len(df):
            return None

        t_data = df.iloc[t_idx]
        entry_data = df.iloc[entry_idx]

        # 從特徵名稱中提取基礎欄位名稱
        features = {}

        for feat_name in self.feature_columns:
            # 解析特徵名稱: {column}_at_{t|entry}
            if "_at_entry" in feat_name:
                base_col = feat_name.replace("_at_entry", "")
                if base_col not in df.columns:
                    return None
                if pd.isna(entry_data[base_col]):
                    return None
                features[feat_name] = entry_data[base_col]
            elif "_at_t" in feat_name:
                base_col = feat_name.replace("_at_t", "")
                if base_col not in df.columns:
                    return None
                if pd.isna(t_data[base_col]):
                    return None
                features[feat_name] = t_data[base_col]
            else:
                return None

        return features

    def detect(self, df: pd.DataFrame) -> np.ndarray:
        """
        檢測反轉點

        檢測策略：
        1. 使用 find_local_extremes 找出極值點 [t]
        2. 對每個極值點，嘗試 [t+1] 和 [t+2] 作為入場點
        3. 使用 ML 模型預測是否為好的入場點

        Args:
            df: 包含 OHLCV 數據和技術指標的 DataFrame

        Returns:
            布爾數組，True 表示該點為入場點
        """
        n = len(df)
        if n < 20:
            return np.zeros(n, dtype=bool)

        signals = np.zeros(n, dtype=bool)

        # 1. 找出極值點
        prices = df["close"].values
        extremes, peaks, valleys = find_local_extremes(
            prices, window=10, prominence_factor=0
        )

        # 極值點索引
        extreme_indices = sorted(peaks | valleys)

        if len(extreme_indices) == 0:
            return signals

        # 統計
        candidates_count = 0
        predictions_count = 0

        # 2. 對每個極值點，嘗試 [t+1] 和 [t+2] 作為入場點
        for t_idx in extreme_indices:
            # 嘗試 t+1 和 t+2
            for offset in [1, 2]:
                entry_idx = t_idx + offset

                # 確保入場點有效
                if entry_idx >= n:
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

                    # 使用概率閾值
                    if proba[1] >= self.prob_threshold:
                        signals[entry_idx] = True
                        predictions_count += 1

                except Exception:
                    # 跳過有問題的點
                    continue

        # Debug 信息
        if candidates_count > 0:
            print(f"   [ML V3] 極值點: {len(extreme_indices)} 個")
            print(f"   [ML V3] 候選入場點: {candidates_count} 個")
            print(f"   [ML V3] 預測反轉信號: {predictions_count} 個")

        return signals

    def get_color(self):
        """返回檢測方法的顏色（用於可視化）"""
        return "orange"
