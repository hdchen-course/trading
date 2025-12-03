#!/usr/bin/env python3
"""
ML Reversal V5 反轉檢測器

基於訓練結果優化：
- 使用 Random Forest 模型（RF Top 16 特徵）
- 16 個核心特徵（從 138 個全特徵中精選）
- 簡化的標籤生成邏輯（峰值/谷值分開處理）
- 極值檢測：window=20, prominence_factor=0.3
- ROC AUC: 0.6471，準確率: 61.71%

16 個核心特徵：
1. bb_position_t2 - 布林帶位置
2. rsi_t2 - RSI
3. volatility [t, t+2] - 波動率
4. amplitude [t+1, t+2] - 振幅
5. macd_2nd_t2 - MACD 二階導數
6. volume_ratio_t2 - 成交量比率
7. price_2nd [t, t+1] - 價格二階導數
8. atr_2nd [t, t+2] - ATR 二階導數
9. volume_change_t1 - 成交量變化
10. volume [t, t+2] - 成交量
11. price_change_3_t2 - 3根K線價格變化

時間點：[t], [t+1], [t+2]
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path


class MLReversalV5Detector:
    """
    ML Reversal V5 檢測器 - Random Forest 模型

    使用 16 個核心特徵（RF Top 16）
    基於峰值/谷值分開處理的標籤生成邏輯
    """

    def __init__(
        self,
        model_path: str = "models/reversal/reversal_model_v5_rf_top16.pkl",
    ):
        """
        初始化檢測器

        Args:
            model_path: Random Forest 模型路徑（Top 16 特徵）
        """
        self.name = "ML Reversal V5 (Random Forest Top 16)"
        self.model_path = Path(model_path)

        # 載入模型
        self._load_model()

        # 從模型中獲取實際使用的特徵列表
        # 這些特徵已經在 _load_model 中設置為 self.features

    def _load_model(self):
        """載入 Random Forest 模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        model_data = joblib.load(self.model_path)
        self.model = model_data["model"]

        # 兼容不同的特徵欄位名稱
        if "features" in model_data:
            self.features = model_data["features"]
        elif "feature_columns" in model_data:
            self.features = model_data["feature_columns"]
        else:
            # 使用預設的 20 個核心特徵
            self.features = self.core_features

        self.accuracy = model_data.get("accuracy", 0)
        self.roc_auc = model_data.get("roc_auc", 0)

        print(f"✅ 載入 {self.name} 模型:")
        print(f"   模型路徑: {self.model_path}")
        print(f"   特徵數: {len(self.features)}")
        print(f"   準確率: {self.accuracy:.4f}")
        print(f"   ROC AUC: {self.roc_auc:.4f}")

    def detect(
        self,
        df: pd.DataFrame,
        window: int = 20,
        prob_threshold: float = 0.5,
        stochrsi_threshold_high: float = 90,
        stochrsi_threshold_low: float = 10,
        debug: bool = False,
    ):
        """
        檢測反轉信號

        Args:
            df: 包含價格和技術指標的 DataFrame
            window: 極值檢測窗口（默認20，與訓練一致）
            prob_threshold: 概率閾值
            stochrsi_threshold_high: StochRSI 高閾值（峰值過濾）
            stochrsi_threshold_low: StochRSI 低閾值（谷值過濾）
            debug: 是否輸出調試信息

        Returns:
            買入信號數組（布爾值）
        """
        from utils.extremes import find_local_extremes

        # 初始化信號數組（整數格式：0=無信號, 1=做空, 2=做多）
        signals = np.zeros(len(df), dtype=int)

        # 找出局部極值點
        extremes, peaks, valleys = find_local_extremes(
            df["close"].values, window=window, prominence_factor=0.3
        )

        if debug:
            print(f"\n🔍 找到極值點:")
            print(f"   峰值點: {len(peaks)} 個")
            print(f"   谷值點: {len(valleys)} 個")

        # 應用 StochRSI 過濾
        filtered_peaks = set()
        filtered_valleys = set()

        for peak_idx in peaks:
            if peak_idx < len(df):
                stochrsi_k = df.iloc[peak_idx].get("stochrsi_k", 50)
                if stochrsi_k > stochrsi_threshold_high:
                    filtered_peaks.add(peak_idx)

        for valley_idx in valleys:
            if valley_idx < len(df):
                stochrsi_k = df.iloc[valley_idx].get("stochrsi_k", 50)
                if stochrsi_k < stochrsi_threshold_low:
                    filtered_valleys.add(valley_idx)

        if debug:
            print(f"\n🔍 過濾後（StochRSI）:")
            print(f"   峰值點: {len(filtered_peaks)} 個")
            print(f"   谷值點: {len(filtered_valleys)} 個")

        # 處理峰值和谷值（訓練時包含兩者）
        if len(filtered_peaks) == 0 and len(filtered_valleys) == 0:
            if debug:
                print("⚠️  沒有找到符合條件的極值點")
            return signals

        # 提取特徵並預測
        peak_predictions = []
        valley_predictions = []

        # 處理峰值點（做空）
        for t_idx in filtered_peaks:
            if t_idx + 2 >= len(df):
                continue

            features = self._extract_features(df, t_idx)
            if features is None:
                continue

            prob = self.model.predict_proba([features])[0][1]
            peak_predictions.append((t_idx, prob))

        # 處理谷值點（做多）
        for t_idx in filtered_valleys:
            if t_idx + 2 >= len(df):
                continue

            features = self._extract_features(df, t_idx)
            if features is None:
                continue

            prob = self.model.predict_proba([features])[0][1]
            valley_predictions.append((t_idx, prob))

        if debug:
            print(f"\n🤖 模型預測:")
            print(f"   峰值點樣本: {len(peak_predictions)} 個")
            print(f"   谷值點樣本: {len(valley_predictions)} 個")

        # 根據概率閾值生成信號
        short_count = 0
        long_count = 0

        # 峰值 → 做空信號（1）
        for t_idx, prob in peak_predictions:
            if prob >= prob_threshold:
                entry_idx = t_idx + 2
                if entry_idx < len(df):
                    # 做空條件：[t+2] 價格要低於 [t]（價格已經下跌）
                    t_price = df.iloc[t_idx]["close"]
                    entry_price = df.iloc[entry_idx]["close"]
                    if entry_price < t_price:
                        signals[entry_idx] = 1  # 做空信號
                        short_count += 1

        # 谷值 → 做多信號（2）
        for t_idx, prob in valley_predictions:
            if prob >= prob_threshold:
                entry_idx = t_idx + 2
                if entry_idx < len(df):
                    # 做多條件：[t+2] 價格要高於 [t]（價格已經上漲）
                    t_price = df.iloc[t_idx]["close"]
                    entry_price = df.iloc[entry_idx]["close"]
                    if entry_price > t_price:
                        signals[entry_idx] = 2  # 做多信號
                        long_count += 1

        if debug:
            print(f"   做空信號: {short_count} 個")
            print(f"   做多信號: {long_count} 個")
            print(f"   總信號數: {short_count + long_count} 個")
            print(f"   概率閾值: {prob_threshold:.2f}")

        return signals

    def _extract_features(self, df: pd.DataFrame, t_idx: int):
        """
        提取 16 個核心特徵

        Args:
            df: 數據 DataFrame
            t_idx: 反轉點索引

        Returns:
            特徵列表，如果缺失則返回 None
        """
        # 建立特徵字典
        features_dict = {}

        # 提取三個時間點 [t], [t+1], [t+2] 的數據
        for offset, suffix in [(0, "t"), (1, "t1"), (2, "t2")]:
            idx = t_idx + offset
            if idx >= len(df):
                return None

            row = df.iloc[idx]

            # 直接從 DataFrame 提取所有欄位，加上時間後綴
            for col in df.columns:
                feat_name = f"{col}_{suffix}"
                if not pd.isna(row[col]):
                    features_dict[feat_name] = row[col]

        # 檢查所有需要的特徵是否都存在
        for feat in self.features:
            if feat not in features_dict:
                return None

        # 轉換為按照 self.features 順序的列表
        features = [features_dict[feat] for feat in self.features]

        return features

    def get_color(self):
        """返回可視化顏色"""
        return "magenta"

    def get_marker(self):
        """返回可視化標記"""
        return "s"  # 正方形

    def get_feature_importance(self):
        """
        獲取特徵重要性

        Returns:
            特徵重要性字典
        """
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            return dict(zip(self.features, importance))
        return {}
