#!/usr/bin/env python3
"""幣安 funding 監控儀表板(只顯示、不自動下單)。

用途:funding carry(空永續+等額現貨)是與順勢組合零相關的中性收益(研究:~年化6-8%淨),
但其爆倉尾巴(逼空強平/ADL/交易所停機/USDT脫鉤)模擬抓不到、價格止損擋不住,故**不自動交易**,
只把當前 funding 攤在眼前讓使用者手動判斷。

顯示每個永續的:當前 funding 年化、近 N 根(預設21×8h≈7天)平均年化、正 funding 比例。
純 urllib + pandas(可攜,老機也能跑)。執行:./venv/bin/python scripts/funding_monitor.py
"""

from __future__ import annotations

import json
import urllib.request

import pandas as pd

SYMBOLS = ["BTCUSDT", "AAVEUSDT"]
PERIODS_PER_YEAR = 3 * 365  # funding 每 8h 結算一次
SMOOTH = 21                 # 近 ~7 天


def annualize(rate_8h: float) -> float:
    """單根 8h funding rate → 年化。"""
    return rate_8h * PERIODS_PER_YEAR


def fetch_funding(symbol: str, limit: int = 60) -> pd.DataFrame:
    url = (f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit={limit}")
    d = json.load(urllib.request.urlopen(url, timeout=30))
    return pd.DataFrame({
        "time": pd.to_datetime([int(r["fundingTime"]) for r in d], unit="ms"),
        "rate": [float(r["fundingRate"]) for r in d],
    })


def main() -> None:
    print("=== 幣安 funding 監控(中性 carry 參考;不自動下單)===")
    print(f"{'標的':<10}{'現funding年化':>14}{'近7天平均年化':>16}{'近60根正funding':>16}")
    for s in SYMBOLS:
        try:
            df = fetch_funding(s)
        except Exception as exc:
            print(f"{s:<10}  抓取失敗:{exc}")
            continue
        cur = annualize(df["rate"].iloc[-1])
        avg = annualize(df["rate"].tail(SMOOTH).mean())
        pos = (df["rate"] > 0).mean()
        flag = "✅可考慮中性carry" if avg > 0.05 else ("⚠️funding偏弱/翻負" if avg <= 0 else "—")
        print(f"{s:<10}{cur:>13.1%}{avg:>16.1%}{pos:>15.0%}  {flag}")
    print("\n做法:avg 年化夠正(如>5%)才考慮『空永續+等額現貨多』收 funding;"
          "funding 轉弱/翻負就退。務必低永續槓桿+保證金充足。")
    print("⚠️ 極端尾險(逼空強平空單腿/ADL/交易所停機/USDT脫鉤)此工具與回測都抓不到,"
          "價格止損也擋不住 → 只用小額、能承受該 sleeve 短時間 -20~50% 才做。")


if __name__ == "__main__":
    main()
