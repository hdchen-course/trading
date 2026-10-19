#!/usr/bin/env python3
"""meta 模型單元測試:可分資料上 proba 區分正負、size 倍率單調且置中於 1、存讀一致。"""

import numpy as np
import pandas as pd
import pytest

from engine.meta_model import FEATURE_COLS, MetaModel, train


def _data(n=800, seed=0):
    """造可分資料:rsi14 低 → 較可能贏(人工訊號),其餘雜訊。"""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLS})
    # 讓 dist_ema200 越正(強趨勢)越容易贏
    prob = 1 / (1 + np.exp(-0.9 * X["dist_ema200"]))
    y = (rng.uniform(0, 1, n) < prob).astype(int)
    return X, y


@pytest.mark.parametrize("kind", ["gbm", "logit"])
def test_proba_separates_and_multiplier_centered(kind):
    X, y = _data()
    m = train(X, y, kind=kind)
    # 強趨勢樣本的平均 proba 應高於弱趨勢樣本
    hi = X["dist_ema200"] > 1
    lo = X["dist_ema200"] < -1
    assert m.proba(X[hi]).mean() > m.proba(X[lo]).mean()
    # size 倍率在 clip 範圍內,且整體平均接近 1(置中,平均曝險不變)
    mult = m.size_multiplier(X)
    assert mult.min() >= m.clip[0] - 1e-9 and mult.max() <= m.clip[1] + 1e-9
    assert 0.8 < float(np.mean(mult)) < 1.2


def test_multiplier_monotonic_in_proba():
    X, y = _data()
    m = train(X, y, kind="gbm")
    p = m.proba(X)
    mult = m.size_multiplier(X)
    order = np.argsort(p)
    # proba 升序 → 倍率非遞減(clip 內單調)
    assert np.all(np.diff(mult[order]) >= -1e-9)


def test_save_load_roundtrip(tmp_path):
    X, y = _data()
    m = train(X, y, kind="gbm")
    fp = tmp_path / "meta.pkl"
    m.save(fp)
    m2 = MetaModel.load(fp)
    assert np.allclose(m.size_multiplier(X), m2.size_multiplier(X))
    assert m2.ref_mean == pytest.approx(m.ref_mean)
