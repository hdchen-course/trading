#!/usr/bin/env python3
"""交易品質 meta 模型(meta-labeling):對『順勢進場訊號』用進場當下的市場結構特徵,預測這筆
會不會賺,轉成『部位倍率』做軟性傾斜(高把握加碼、低把握減碼,不丟單→保留分散)。

研究驗證(scripts/research_scratch/meta_portfolio_study.py):軟傾斜(size∝proba/均值,clip)
在 end-anchored 多窗一致勝 baseline,且 GBM 與 Logistic 雙模型確認=非過擬合。

設計:
- **離線訓練 + 持久化**,live 只載入評分(ML 不在可攜硬約束內了,但仍維持引擎 import-safe:
  sklearn 延遲 import,沒裝也能 import 本模組、只是不能 fit/score)。
- **無前視**:特徵來自進場『決策根』(engine.features);size 倍率用『訓練時』的參考均值置中
  (ref_mean,存在模型裡),不用未來資訊。
- 倍率 = clip(proba / ref_mean, lo, hi):平均曝險≈1,只重分配。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 與 engine.features.build_feature_frame 對齊的特徵欄(順序固定,評分時據此取值)
FEATURE_COLS = [
    "rsi14", "roc20", "dist_ema200", "dist_ema50", "ema_stack", "atr_pct100",
    "dist_resist", "dist_support", "trend_age200", "dist_daily_axis", "dist_weekly_axis",
]
DEFAULT_CLIP = (0.3, 2.0)


@dataclass
class MetaModel:
    """封裝已訓練的分類器 + 參考均值,提供 proba 與 size 倍率。"""

    model: object            # sklearn 分類器(predict_proba)
    ref_mean: float          # 訓練集平均 proba(置中用,無前視)
    clip: tuple = DEFAULT_CLIP

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        """回傳 P(win) ∈ [0,1]。X 需含 FEATURE_COLS。"""
        arr = X[FEATURE_COLS].to_numpy(dtype=float)
        return self.model.predict_proba(arr)[:, 1]

    def size_multiplier(self, X: pd.DataFrame) -> np.ndarray:
        """部位倍率 = clip(proba / 訓練參考均值, lo, hi);平均≈1,高把握加碼、低把握減碼。"""
        p = self.proba(X)
        ref = self.ref_mean if self.ref_mean and self.ref_mean > 0 else 0.5
        lo, hi = self.clip
        return np.clip(p / ref, lo, hi)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "ref_mean": self.ref_mean,
                         "clip": self.clip, "features": FEATURE_COLS}, f)

    @staticmethod
    def load(path: str | Path) -> "MetaModel":
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d.get("features") != FEATURE_COLS:
            raise ValueError("模型特徵欄與目前 FEATURE_COLS 不符,請重新訓練")
        return MetaModel(model=d["model"], ref_mean=d["ref_mean"], clip=d.get("clip", DEFAULT_CLIP))


def train(X: pd.DataFrame, y, clip: tuple = DEFAULT_CLIP, kind: str = "gbm") -> MetaModel:
    """離線訓練 meta 模型。X 含 FEATURE_COLS,y=該筆是否獲利(1/0)。
    kind='gbm'(HistGradientBoosting,容忍 NaN)或 'logit'(Logistic+impute+scale)。"""
    Xa = X[FEATURE_COLS].to_numpy(dtype=float)
    ya = np.asarray(y).astype(int)
    if kind == "logit":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=0.3, max_iter=500))
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.05,
            l2_regularization=1.0, min_samples_leaf=40, random_state=0)
    model.fit(Xa, ya)
    ref_mean = float(model.predict_proba(Xa)[:, 1].mean())
    return MetaModel(model=model, ref_mean=ref_mean, clip=clip)
