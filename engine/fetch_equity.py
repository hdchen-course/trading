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


def merge_ohlcv(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """合併兩段 OHLCV,依 timestamp 去重(保留較新那筆)、排序。純函式便於測試。"""
    if old is None or len(old) == 0:
        return new.reset_index(drop=True)
    if new is None or len(new) == 0:
        return old.reset_index(drop=True)
    both = pd.concat([old, new], ignore_index=True)
    both = both.drop_duplicates(subset="timestamp", keep="last")
    return both.sort_values("timestamp").reset_index(drop=True)


def fetch_equity_daily(symbol: str, period: str = "max") -> pd.DataFrame:
    """抓單一美股/ETF 日線,正規化為標準 OHLCV(timestamp/open/high/low/close/volume)。"""
    import yfinance as yf

    raw = yf.download(symbol, period=period, interval="1d",
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


def load_or_fetch_daily(symbol: str, refresh: bool = False) -> pd.DataFrame:
    """讀快取;refresh=True 時再抓最近 6 個月與快取合併(補上最新交易日),回寫快取。

    refresh 讓 live 的美股日線即時到當日(否則讀快取會差一天)。老機器無 yfinance 時,
    呼叫端已以 try/except 包起,會退回純快取或略過美股。
    """
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / f"{symbol.upper()}_1d.csv"
    cache = pd.read_csv(path, parse_dates=["timestamp"]) if path.exists() else None

    if cache is not None and not refresh:
        return cache

    recent_period = "6mo" if cache is not None else "max"
    fresh = fetch_equity_daily(symbol, period=recent_period)
    merged = merge_ohlcv(cache, fresh)
    if len(merged):
        merged.to_csv(path, index=False)
    return merged


if __name__ == "__main__":
    import sys

    syms = sys.argv[1:] or ["SOXL", "TQQQ", "SOXX", "KORU", "MSTR", "INTC", "NVDA", "SPXL"]
    for s in syms:
        d = load_or_fetch_daily(s)
        rng = f"{d.timestamp.min().date()} → {d.timestamp.max().date()}" if len(d) else "(空)"
        print(f"{s}: {len(d)} 日 | {rng}")
