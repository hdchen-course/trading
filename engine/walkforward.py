#!/usr/bin/env python3
"""Walk-forward 驗證引擎(守門員)——比單一切點更嚴、對資料 vintage 不敏感的時間序列驗證。

為何需要:單一「後30%」切點對 cache 刷新/資料回填敏感(見研究教訓),且只測一個時期。
walk-forward 把時間切成多個連續的「訓練窗→(purge)→測試窗」,模型只用過去擬合、對緊接的未來
預測,把所有測試窗的 OOS 預測串起來評估。這才敢信「更複雜的情境模型」不是在背答案。

設計:
- **防洩漏(no leakage)是鐵律**:測試索引永遠在訓練索引之後;且訓練窗尾端留 embargo(purge)個
  bar 不用,避免「標籤是未來 H 根報酬」時訓練標籤的視窗與測試期重疊而洩漏。
- **模型無關**:呼叫端提供 fit_predict(X_train, y_train, X_test)->preds;引擎本身不依賴 sklearn,
  維持可攜(純 pandas/numpy)。ML 與否、用什麼模型,都由呼叫端決定。
- expanding(預設,用全部過去)或 rolling(限 max_train 窗)。
"""

from __future__ import annotations

from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


def walk_forward_splits(
    n: int,
    test_size: int,
    min_train: int,
    embargo: int = 0,
    step: Optional[int] = None,
    max_train: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """產生 (train_idx, test_idx) 整數索引序列(0..n-1)。

    參數:
        test_size: 每個測試窗長度。
        min_train: 第一個測試窗前至少要有的訓練根數。
        embargo: 訓練窗尾端要「purge」掉的根數(通常=標籤的前視視窗 H),防洩漏。
        step: 測試窗前進步長,預設=test_size(不重疊)。
        max_train: 設了則為 rolling 窗(訓練只取最近 max_train 根);否則 expanding(用全部過去)。
    """
    if step is None:
        step = test_size
    test_start = min_train + embargo
    while test_start + test_size <= n:
        train_end = test_start - embargo  # 訓練只到 test_start-embargo(purge 掉 embargo 根)
        train_lo = 0 if max_train is None else max(0, train_end - max_train)
        train_idx = np.arange(train_lo, train_end)
        test_idx = np.arange(test_start, test_start + test_size)
        if len(train_idx) >= min_train:
            yield train_idx, test_idx
        test_start += step


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    fit_predict: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray],
    test_size: int,
    min_train: int,
    embargo: int = 0,
    step: Optional[int] = None,
    max_train: Optional[int] = None,
) -> pd.Series:
    """跨 walk-forward 折疊累積 OOS 預測。回傳與 X 同索引的預測 Series(未被預測處為 NaN)。

    fit_predict(X_train, y_train, X_test) 回傳對 X_test 每列的預測(長度=len(X_test))。
    只會用到訓練折疊(過去+purge)資料,故串起來的預測是純樣本外。
    """
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y), index=X.index)
    preds = pd.Series(np.nan, index=X.index, dtype=float)
    for train_idx, test_idx in walk_forward_splits(
        len(X), test_size, min_train, embargo, step, max_train
    ):
        Xtr, ytr = X.iloc[train_idx], y.iloc[train_idx]
        mask = ytr.notna()
        if mask.sum() < min_train // 2 or Xtr.loc[mask].isna().all(axis=None):
            continue
        out = fit_predict(Xtr.loc[mask], ytr.loc[mask], X.iloc[test_idx])
        preds.iloc[test_idx] = np.asarray(out, dtype=float)
    return preds


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """標籤輔助:未來 horizon 根的報酬(close_{t+h}/close_t - 1)。
    注意:這是『前視標籤』,只能當 y 訓練用,最後 horizon 根為 NaN;配合 embargo=horizon 防洩漏。"""
    return close.shift(-horizon) / close - 1.0


def evaluate_predictions(
    preds: pd.Series, fwd_ret: pd.Series, long_thr: float, short_thr: Optional[float] = None
) -> dict:
    """把 OOS 預測轉成『高把握才出手』的動作並評估(precision 重於 recall)。
    preds≥long_thr→做多、≤short_thr→做空(None 表示只做多);用對應 fwd_ret 結算(需同索引)。
    回傳 n_trades/命中率/平均報酬/期望等。"""
    preds, fwd_ret = preds.align(fwd_ret, join="inner")
    valid = preds.notna() & fwd_ret.notna()
    preds, fwd_ret = preds[valid], fwd_ret[valid]
    pos = pd.Series(0, index=preds.index)
    pos[preds >= long_thr] = 1
    if short_thr is not None:
        pos[preds <= short_thr] = -1
    acted = pos != 0
    trade_ret = (pos * fwd_ret)[acted]
    n = int(acted.sum())
    return {
        "n_trades": n,
        "coverage": float(acted.mean()) if len(acted) else 0.0,
        "hit_rate": float((trade_ret > 0).mean()) if n else float("nan"),
        "avg_ret": float(trade_ret.mean()) if n else float("nan"),
        "total_ret": float(trade_ret.sum()) if n else 0.0,
    }
