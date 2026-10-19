#!/usr/bin/env python3
"""前向紙上交易帳本:把每次觸發的訊號落地存檔,供日後「實盤 vs 回測」比對。

這是檢驗系統「是否真的如回測預期」的唯一誠實方法 —— 先紙上記錄幾週,
再看實際訊號的時點/價格/後續走勢是否與回測一致,確認無誤才放真錢。
純 pandas,冪等(idempotent):同一根訊號重複執行 run_live 不會重複記錄。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

# 唯一鍵:同一標的、週期、訊號、K 線時間、方向 只記一次
_KEY = ["coin", "timeframe", "signal_name", "time", "direction"]
_COLS = _KEY + ["price", "init_stop", "notional_pct_of_equity", "logged_at"]


def log_signals(events: List[dict], path: str = "output/paper_trades.csv") -> int:
    """把訊號事件附加到帳本 CSV,回傳「這次新增的筆數」(冪等去重)。"""
    if not events:
        return 0
    now = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        {
            "coin": e["coin"],
            "timeframe": e["timeframe"],
            "signal_name": e["signal_name"],
            "time": str(e["time"]),
            "direction": e["direction"],
            "price": e["price"],
            "init_stop": e.get("init_stop"),
            "notional_pct_of_equity": e.get("notional_pct_of_equity"),
            "logged_at": now,
        }
        for e in events
    ]
    new = pd.DataFrame(rows, columns=_COLS)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        existing = pd.read_csv(p)
        combined = pd.concat([existing, new], ignore_index=True)
        before = len(existing)
        combined = combined.drop_duplicates(subset=_KEY, keep="first")
        added = len(combined) - before
    else:
        combined = new.drop_duplicates(subset=_KEY, keep="first")
        added = len(combined)
    combined.to_csv(p, index=False)
    return added
