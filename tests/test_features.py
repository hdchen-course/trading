#!/usr/bin/env python3
"""市場結構特徵庫的單元測試。

最重要的是「無前視(no-lookahead)性質測試」:把資料截斷到第 k 根再算,末值必須與用
完整資料算出的第 k 個值相同。任何前視偏誤都會讓這個測試失敗——因為使用者無法自行 audit,
這個性質測試就是特徵可信的保證。
"""

import numpy as np
import pandas as pd

from engine import features as F


def _synthetic(n=400, seed=0):
    """造一段隨機漫步 OHLCV(足夠長以通過 EMA200/atr_pct 暖身)。"""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    steps = rng.normal(0, 1, n).cumsum()
    close = 100 + steps
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close - rng.normal(0, 0.3, n)
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high,
                         "low": low, "close": close, "volume": 1.0})


_FEATURES = {
    "rsi14": lambda d: F.rsi(d["close"], 14),
    "roc20": lambda d: F.roc(d["close"], 20),
    "dist_ema200": lambda d: F.dist_from_ema(d, 200),
    "ema_stack": lambda d: F.ema_stack_score(d, (20, 50, 100, 200)),
    "atr_pct100": lambda d: F.atr_percentile(d, 14, 100),
    "recent_swing_high": lambda d: F.recent_swing_high(d, 3, 3),
    "recent_swing_low": lambda d: F.recent_swing_low(d, 3, 3),
    "dist_resist": lambda d: F.dist_to_resistance(d),
    "dist_support": lambda d: F.dist_to_support(d),
    "trend_age200": lambda d: F.trend_age(d, 200),
}


def test_no_lookahead_truncation_invariance():
    """核心:每個特徵在第 k 根的值,只能用到第 k 根(含)以前的資料。
    → 把 df 截到 [:k+1] 重算,末值必須等於用完整 df 算出的第 k 個值。"""
    df = _synthetic()
    full = {name: fn(df) for name, fn in _FEATURES.items()}
    for k in (250, 300, 350, 399):  # 暖身後的數個位置
        trunc = df.iloc[: k + 1].reset_index(drop=True)
        for name, fn in _FEATURES.items():
            v_trunc = fn(trunc).iloc[-1]
            v_full = full[name].iloc[k]
            if pd.isna(v_trunc) and pd.isna(v_full):
                continue
            assert v_trunc == v_full or abs(v_trunc - v_full) < 1e-9, (
                f"{name} 第 {k} 根前視偏誤:截斷={v_trunc} 完整={v_full}"
            )


def test_rsi_bounds_and_all_up():
    df = _synthetic()
    r = F.rsi(df["close"], 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()
    # 一路上漲 → RSI 應趨近 100(avg_loss=0 時定義為 100)
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert F.rsi(up, 14).iloc[-1] == 100.0


def test_ema_stack_bounds_and_direction():
    df = _synthetic()
    s = F.ema_stack_score(df, (20, 50, 100, 200)).dropna()
    assert (s >= -1).all() and (s <= 1).all()
    # 嚴格遞增序列 → 完全多頭排列 = +1
    up = pd.DataFrame({"close": np.arange(1, 400, dtype=float)})
    assert F.ema_stack_score(up, (20, 50, 100, 200)).iloc[-1] == 1.0


def test_swing_high_is_a_past_local_max():
    """recent_swing_high 應回報一個「過去出現過」的高點(不超過到當下為止的最高 high)。"""
    df = _synthetic()
    rsh = F.recent_swing_high(df, 3, 3)
    running_max_high = df["high"].cummax()
    ok = rsh.dropna()
    # 已確認的波段高必 ≤ 當下為止的歷史最高(不可能來自未來)
    assert (ok <= running_max_high.loc[ok.index] + 1e-9).all()


def test_build_feature_frame_shape():
    df = _synthetic()
    feats = F.build_feature_frame(df)
    assert len(feats) == len(df)
    assert "rsi14" in feats.columns and "dist_resist" in feats.columns
