#!/usr/bin/env python3
"""
ML Reversal V4 反轉檢測器

基於 V3 改進：
- 使用 Random Forest / LightGBM / CatBoost 三模型集成
- 根據 ROC AUC 加權投票
- Top 20 特徵，性能與速度平衡

模型權重：
- Random Forest: 35.2% (ROC AUC 0.6901)
- LightGBM: 33.2% (ROC AUC 0.6504)
- CatBoost: 33.1% (ROC AUC 0.6495)
"""

import numpy as np
import joblib
from pathlib import Path


class MLReversalV4Detector:
    """
    ML Reversal V4 檢測器 - 三模型加權投票

    使用 Random Forest, LightGBM, CatBoost 三個模型
    根據 ROC AUC 加權投票
    """

    def __init__(
        self,
        rf_model_path: str = "models/reversal/reversal_model_v4_rf_top20.pkl",
        lgb_model_path: str = "models/reversal/reversal_model_v4_lgb_top20.pkl",
        cat_model_path: str = "models/reversal/reversal_model_v4_cat_top20.pkl",
    ):
        """
        初始化檢測器

        Args:
            rf_model_path: Random Forest 模型路徑
            lgb_model_path: LightGBM 模型路徑
            cat_model_path: CatBoost 模型路徑
        """
        self.name = "ML Reversal V4 Ensemble"
        self.models = {}
        self.weights = {}

        # 載入三個模型
        self._load_model("Random Forest", rf_model_path)
        self._load_model("LightGBM", lgb_model_path)
        self._load_model("CatBoost", cat_model_path)

        # 計算權重（基於 ROC AUC）
        self._calculate_weights()

    def _load_model(self, name: str, model_path: str):
        """載入單個模型"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{name} 模型文件不存在: {model_path}")

        model_data = joblib.load(model_path)
        self.models[name] = {
            "model": model_data["model"],
            "features": model_data["feature_columns"],
            "roc_auc": model_data.get("roc_auc", 0.5),
        }

        print(f"✅ 載入 {name} 模型:")
        print(f"   特徵數: {len(model_data['feature_columns'])}")
        print(f"   ROC AUC: {model_data.get('roc_auc', 'N/A'):.4f}")

    def _calculate_weights(self):
        """根據 ROC AUC 計算權重"""
        total_auc = sum(m["roc_auc"] for m in self.models.values())

        for name, model_data in self.models.items():
            weight = model_data["roc_auc"] / total_auc
            self.weights[name] = weight

        print(f"\n📊 模型權重:")
        for name, weight in self.weights.items():
            print(f"   {name}: {weight:.3f} ({weight*100:.1f}%)")

    def detect(
        self, df, window: int = 10, prob_threshold: float = 0.5, debug: bool = False
    ):
        """
        使用加權投票檢測反轉信號

        Args:
            df: 包含價格和技術指標的 DataFrame
            window: 極值檢測窗口
            prob_threshold: 加權概率閾值
            debug: 是否輸出調試信息

        Returns:
            買入信號數組（布爾值）
        """
        from utils.extremes import find_local_extremes

        if len(df) < window * 2:
            return np.zeros(len(df), dtype=bool)

        # 找出極值點
        prices = df["close"].values
        extremes, peaks, valleys = find_local_extremes(
            prices, window=window, prominence_factor=0.1
        )

        extreme_indices = sorted(peaks | valleys)

        # 初始化信號數組
        signals = np.zeros(len(df), dtype=bool)

        if debug:
            print(f"\n[ML V4 Debug] 找到 {len(extreme_indices)} 個極值點")

        predictions_count = 0
        candidates_count = 0

        # 對每個極值點進行預測
        for t_idx in extreme_indices:
            for offset in [1, 2]:
                entry_idx = t_idx + offset

                if entry_idx >= len(df):
                    continue

                candidates_count += 1

                try:
                    # 獲取三個模型的加權預測
                    weighted_proba = 0.0

                    for name, model_data in self.models.items():
                        features = self._extract_features(
                            df, t_idx, entry_idx, model_data["features"]
                        )

                        if features is None:
                            continue

                        features_array = np.array([features])
                        proba = model_data["model"].predict_proba(features_array)[0]

                        # 加權
                        weighted_proba += proba[1] * self.weights[name]

                    # 生成信號
                    if weighted_proba >= prob_threshold:
                        signals[entry_idx] = True
                        predictions_count += 1

                except Exception:
                    continue

        # Debug 信息
        if candidates_count > 0:
            print(f"   [ML V4] 極值點: {len(extreme_indices)} 個")
            print(f"   [ML V4] 候選點: {candidates_count} 個")
            print(f"   [ML V4] 預測信號: {predictions_count} 個")
            print(f"   [ML V4] 信號率: {predictions_count/candidates_count*100:.1f}%")

        return signals

    def get_color(self):
        """返回可視化顏色"""
        return "purple"

    def get_marker(self):
        """返回可視化標記"""
        return "D"  # 菱形

    def _extract_features(self, df, t_idx: int, entry_idx: int, feature_columns: list):
        """提取特徵向量"""
        features = []

        for col in feature_columns:
            if col.endswith("_at_t"):
                base_col = col[:-5]
                if base_col in df.columns:
                    features.append(df.iloc[t_idx][base_col])
                else:
                    return None
            elif col.endswith("_at_entry"):
                base_col = col[:-9]
                if base_col in df.columns:
                    features.append(df.iloc[entry_idx][base_col])
                else:
                    return None
            else:
                return None

        return features
