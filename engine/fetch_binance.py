#!/usr/bin/env python3
"""幣安 USDT-M 永續合約 K 線抓取(純標準庫 urllib,無額外依賴,可攜)。

用於多幣種研究:抓取並快取各永續合約的 K 線到 data/cache/<symbol>_<interval>.csv。
資料來源:https://fapi.binance.com/fapi/v1/klines(USDT 本位永續)。
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

_FAPI = "https://fapi.binance.com/fapi/v1/klines"
_LIMIT = 1500  # 每次請求最多根數
_CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"


def _interval_ms(interval: str) -> int:
    unit = interval[-1]
    num = int(interval[:-1])
    return num * {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit]


def fetch_klines(symbol: str, interval: str = "1h", start_ms: int | None = None,
                 end_ms: int | None = None) -> pd.DataFrame:
    """抓取單一永續合約 K 線,自動分頁直到抓完整段。回傳標準 OHLCV DataFrame。"""
    if start_ms is None:
        start_ms = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp() * 1000)
    if end_ms is None:
        end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    step = _interval_ms(interval)
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        params = urllib.parse.urlencode(
            {"symbol": symbol, "interval": interval, "startTime": cursor,
             "limit": _LIMIT}
        )
        with urllib.request.urlopen(f"{_FAPI}?{params}", timeout=20) as resp:
            batch = json.loads(resp.read())
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        cursor = last_open + step
        if len(batch) < _LIMIT:
            break
        time.sleep(0.25)  # 禮貌限速
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_recent(symbol: str, interval: str = "1h", bars: int = 6000) -> pd.DataFrame:
    """抓取最近 bars 根 K 線(供即時監控用,不讀快取,總是取最新)。

    bars 預設 6000:足夠覆蓋 12h ribbon 的 420 EMA(420*12h ≈ 5040 根 1h)+ 緩衝。
    """
    step = _interval_ms(interval)
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_ms = end_ms - bars * step
    return fetch_klines(symbol, interval, start_ms=start_ms, end_ms=end_ms)


def load_or_fetch(symbol: str, interval: str = "1h") -> pd.DataFrame:
    """優先讀快取,沒有才抓並寫快取。"""
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / f"{symbol}_{interval}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    df = fetch_klines(symbol, interval)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    import sys

    symbols = sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    for sym in symbols:
        d = load_or_fetch(sym, "1h")
        rng = f"{d.timestamp.min()} → {d.timestamp.max()}" if len(d) else "(空)"
        print(f"{sym}: {len(d)} 根 1h | {rng}")
