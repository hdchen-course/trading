#!/usr/bin/env python3
"""總經風險 overlay(美股用):VIX 飆升時對美股『做多』風險off。

只作用於美股腿(合約跟著現貨美股走,總經/風險事件是美股的濾網);加密腿不受影響。
已在 scripts/research_scratch/risk_overlay.py 多窗驗證:VIX>30 全平美股多單,
樣本外(後30%)Sharpe 1.11→1.18、近4年 1.64→1.71、最大回撤 17.0%→16.8%——
改善集中在樣本外(反過擬合特徵),且 VIX 25/30 門檻皆成立(門檻穩健)。

無前視鐵則:某日的風險off判定只看『當日可得』的 VIX(預設落後 1 天=用昨收),
故 live 當日用昨日 VIX 收盤決策,絕不用當日未來值。純函式,便於測試。
"""

from __future__ import annotations

import pandas as pd

DEFAULT_VIX_THRESHOLD = 30.0  # 恐慌級;研究中 25 與 30 皆改善,取較保守的 30(干預少、回撤最佳)
DEFAULT_LAG_DAYS = 1  # VIX 日頻當日收盤才定 → 隔日才可用,避免前視
EQUITY_TIMEFRAME = "1d"  # 美股腿以日線辨識(加密腿為 4h/8h/12h)


def vix_risk_off(vix_close: pd.Series, threshold: float = DEFAULT_VIX_THRESHOLD,
                 lag_days: int = DEFAULT_LAG_DAYS) -> pd.Series:
    """回傳『風險off』布林序列(index 對齊輸入日期):VIX(落後 lag_days)> threshold。

    用 shift(lag_days) 把判定右移,確保 t 日只用 t-lag 的 VIX;無前視。
    """
    s = pd.Series(vix_close).astype(float)
    return (s.shift(lag_days) > threshold).fillna(False)


def is_risk_off_now(vix_df: pd.DataFrame, threshold: float = DEFAULT_VIX_THRESHOLD,
                    lag_days: int = DEFAULT_LAG_DAYS) -> bool:
    """由 VIX 日線 DataFrame(需含 close)判定『現在』是否風險off(用最後一筆的 as-of 值)。

    等同取 vix_risk_off(...) 的最後一個值;資料不足時回傳 False(不干預)。
    """
    if vix_df is None or len(vix_df) <= lag_days or "close" not in vix_df:
        return False
    return bool(vix_risk_off(vix_df["close"], threshold, lag_days).iloc[-1])


def filter_equity_longs(events: list[dict], risk_off: bool) -> tuple[list[dict], list[dict]]:
    """風險off 時濾掉『美股新做多』進場事件(加密與所有出場/美股做空不動)。

    回傳 (保留的事件, 被濾掉的美股做多事件)。risk_off=False 時原樣返回。
    """
    if not risk_off:
        return events, []
    kept, suppressed = [], []
    for e in events:
        is_equity = e.get("timeframe") == EQUITY_TIMEFRAME
        is_long = str(e.get("direction", "")).upper() == "LONG"
        (suppressed if (is_equity and is_long) else kept).append(e)
    return kept, suppressed
