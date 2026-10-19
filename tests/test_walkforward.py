#!/usr/bin/env python3
"""Walk-forward 引擎的單元測試。最重要的是「防洩漏」:測試折疊永遠在訓練之後,
且訓練尾端有 embargo 個 bar 被 purge(避免前視標籤洩漏)。"""

import numpy as np
import pandas as pd

from engine.walkforward import (
    evaluate_predictions,
    forward_return,
    walk_forward_predict,
    walk_forward_splits,
)


def test_no_leakage_and_embargo():
    """每折:max(train) 與 min(test) 之間至少隔 embargo 根(purge),且 train 全在 test 之前。"""
    emb = 5
    folds = list(walk_forward_splits(n=1000, test_size=100, min_train=300, embargo=emb))
    assert len(folds) > 0
    for tr, te in folds:
        assert tr.max() < te.min()                      # 訓練全在測試之前
        assert te.min() - tr.max() - 1 >= emb           # 中間至少 purge embargo 根
        assert len(tr) >= 300                            # 滿足 min_train


def test_test_blocks_tile_without_overlap():
    """預設 step=test_size → 測試窗不重疊、連續鋪滿。"""
    folds = list(walk_forward_splits(n=1000, test_size=100, min_train=300, embargo=0))
    test_starts = [te.min() for _, te in folds]
    assert test_starts[0] == 300
    for a, b in zip(test_starts[:-1], test_starts[1:]):
        assert b - a == 100  # 相鄰測試窗起點差 = test_size


def test_rolling_max_train():
    """設 max_train → 訓練窗長度被限制(rolling)。"""
    folds = list(walk_forward_splits(1000, 100, 300, 0, max_train=200))
    for tr, _ in folds:
        assert len(tr) <= 200


def test_walk_forward_predict_is_oos_only():
    """預測只出現在測試區(第一個測試窗之前全為 NaN),且用 trivial 模型值正確。"""
    n = 600
    X = pd.DataFrame({"f": np.arange(n, dtype=float)})
    y = pd.Series(np.arange(n, dtype=float))

    # trivial 模型:預測 = 訓練 y 的平均(與 X_test 無關)→ 可驗證它只用訓練資料
    def fit_predict(Xtr, ytr, Xte):
        return np.full(len(Xte), ytr.mean())

    preds = walk_forward_predict(X, y, fit_predict, test_size=100, min_train=300)
    assert preds.iloc[:300].isna().all()      # 第一個測試窗(從 300)之前無預測
    assert preds.iloc[300:].notna().all()      # 之後每根都有 OOS 預測
    # 第一折訓練 y=[0..299],平均=149.5 → 測試區 [300,400) 的預測應為 149.5
    assert abs(preds.iloc[300] - 149.5) < 1e-9


def test_forward_return_and_eval():
    close = pd.Series([100, 110, 121, 133.1])  # 每步 +10%
    fr = forward_return(close, 1)
    assert abs(fr.iloc[0] - 0.10) < 1e-9
    assert pd.isna(fr.iloc[-1])                # 最後一根無未來 → NaN
    # evaluate:預測全部 ≥ 門檻 → 全做多,命中率 100%
    preds = pd.Series([1.0, 1.0, 1.0, 1.0])
    res = evaluate_predictions(preds, fr, long_thr=0.5)
    assert res["n_trades"] == 3 and res["hit_rate"] == 1.0
