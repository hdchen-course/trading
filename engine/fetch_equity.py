#!/usr/bin/env python3
"""美股 / ETF 日線抓取(yfinance,研究用資料源)。

注意分工:yfinance 只在「研究 / 抓資料」階段使用(這台機器),抓完後正規化成標準 OHLCV CSV
快取於 data/cache/<SYMBOL>_1d.csv。**可攜的即時核心只讀這些 CSV,不依賴 yfinance**,因此
老 macOS(無法裝 yfinance)仍可跑即時訊號。

使用 auto_adjust=True 取得「還原分割/配息後」的 OHLC,避免槓桿 ETF 分割造成的價格跳空假訊號。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"


def fetch_equity_daily(symbol: str) -> pd.DataFrame:
    """抓單一美股/ETF 日線,正規化為標準 OHLCV(timestamp/open/high/low/close/volume)。"""
    import yfinance as yf

    raw = yf.download(symbol, period="max", interval="1d",
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    # yfinance 多層欄位 → 取單一標的那層
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(
        columns={"Open": "open", "High": "high", "Low": "low",
                 "Close": "close", "Volume": "volume"}
    )
    out = raw.reset_index().rename(columns={"Date": "timestamp"})
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    out = out[cols].dropna(subset=["open", "high", "low", "close"])
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out.reset_index(drop=True)


def load_or_fetch_daily(symbol: str) -> pd.DataFrame:
    """優先讀快取,沒有才抓並寫快取。"""
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / f"{symbol.upper()}_1d.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    df = fetch_equity_daily(symbol)
    if len(df):
        df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    import sys

    syms = sys.argv[1:] or ["SOXL", "TQQQ", "SOXX", "KORU", "MSTR", "INTC", "NVDA", "SPXL"]
    for s in syms:
        d = load_or_fetch_daily(s)
        rng = f"{d.timestamp.min().date()} → {d.timestamp.max().date()}" if len(d) else "(空)"
        print(f"{s}: {len(d)} 日 | {rng}")
