#!/usr/bin/env python3
"""離線訓練交易品質 meta 模型(全史池化交易)並持久化,供 live 對新訊號評分→部位傾斜。

驗證在 scripts/research_scratch/meta_portfolio_study.py(walk-forward 多窗確認軟傾斜勝 baseline)。
這裡用『全部歷史交易』訓練最終部署模型(標準做法),存到 models/meta/meta_model.pkl。
重新訓練:定期跑此腳本(例如每月)即可。

執行:./venv/bin/python scripts/train_meta_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import research_combined as rc  # noqa: E402
from engine.backtester import EventBacktester  # noqa: E402
from engine.data import resample_ohlcv  # noqa: E402
from engine.features import build_feature_frame  # noqa: E402
from engine.fetch_binance import load_or_fetch  # noqa: E402
from engine.fetch_equity import load_or_fetch_daily  # noqa: E402
from engine.meta_model import FEATURE_COLS, train  # noqa: E402

OUT = ROOT / "models" / "meta" / "meta_model.pkl"


def collect_trades():
    rows = []

    def do(d, sig, bt):
        res = EventBacktester(**bt, **rc.BT_COMMON).run(d, sig)
        if not res.trades:
            return
        feats = build_feature_frame(d)
        for t in res.trades:
            fi = max(0, t.entry_idx - 1)
            r = feats.iloc[fi]
            rows.append({**{c: r[c] for c in FEATURE_COLS}, "win": int(t.net_return_pct > 0)})

    for c in rc.CRYPTO:
        raw = load_or_fetch(c, "1h")
        if len(raw) >= 2000:
            for tf, fn in rc.CRYPTO_SIGS:
                d = resample_ohlcv(raw, tf)
                if len(d) >= 400:
                    do(d, rc._gate(d, fn(d)), rc.BT_CRYPTO)
    for s in rc.EQUITY:
        d = load_or_fetch_daily(s)
        if len(d) >= 400:
            for fn in rc.EQUITY_SIGS:
                do(d, rc._gate(d, fn(d)), rc.BT_EQUITY)
    return pd.DataFrame(rows)


def main():
    df = collect_trades().dropna()
    print(f"池化交易 {len(df)} 筆,勝率 {df['win'].mean():.1%}")
    model = train(df[FEATURE_COLS], df["win"], kind="gbm")
    model.save(OUT)
    print(f"已存模型 → {OUT}(ref_mean={model.ref_mean:.3f},clip={model.clip})")
    # 快速自檢:高/低把握樣本的建議倍率
    mult = model.size_multiplier(df[FEATURE_COLS])
    print(f"部位倍率分布:min {mult.min():.2f} / 中位 {pd.Series(mult).median():.2f} / "
          f"max {mult.max():.2f} / 均 {mult.mean():.2f}")


if __name__ == "__main__":
    main()
