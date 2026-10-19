#!/usr/bin/env python3
"""生產用的順勢組合策略:輸出「當下該不該進場」的可執行訊號。

這是研究結論(高週期順勢 + ATR 移動停損 + 風險控管 + 多幣種分散)的落地層。
不含任何 ML、不依賴 TA-Lib,只用 pandas/numpy,能在老機器上跑。

current_signals() 只回報「這一根新出現」的訊號(由 0 或反向翻成多/空的那一刻),
避免狀態型訊號(如 ribbon)每根重複通知。每個事件附帶建議初始停損與當前 ATR,
供下游(監控/Telegram)直接呈現給使用者。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import pandas as pd

from engine.data import atr, resample_ohlcv
from signals.library import (
    donchian_breakout,
    ema_cross,
    ema_ribbon_trend,
    regime_filter,
    trend_distance_gate,
)


@dataclass
class Leg:
    """一條策略腿:某幣種、某週期、某訊號。"""

    coin: str
    timeframe: str
    name: str
    signal_fn: Callable[[pd.DataFrame], pd.Series]


def default_legs(coins: List[str]) -> List[Leg]:
    """對每個幣種套用三個通過樣本外驗證的順勢訊號。"""
    legs: List[Leg] = []
    for coin in coins:
        legs.append(Leg(coin, "4h", "EMA50/200", lambda d: ema_cross(d, 50, 200)))
        legs.append(Leg(coin, "8h", "Donchian20", lambda d: donchian_breakout(d, 20)))
        legs.append(
            Leg(coin, "12h", "ribbon",
                lambda d: ema_ribbon_trend(d, (90, 120, 240, 300, 420)))
        )
    return legs


@dataclass
class TrendPortfolio:
    """順勢組合策略。

    參數:
        legs: 策略腿清單(幣種×週期×訊號)。
        stop_loss_pct: 建議初始硬停損(價格比率)。
        atr_period: ATR 週期,供移動停損參考值使用。
        trailing_atr_mult: 移動停損 ATR 倍數(僅作為通知時的建議,實際追蹤由監控/下單端執行)。
    """

    legs: List[Leg]
    stop_loss_pct: float = 0.06
    atr_period: int = 14
    trailing_atr_mult: float = 4.0
    risk_per_trade: float = 0.02  # 穩健檔:每筆最多賠權益 2%
    regime_ema: Optional[int] = 200  # 市場狀態濾網:只在 close 站上/跌破 EMA200 時順勢進場(None=關閉)
    dist_gate_atr: Optional[float] = 1.0  # 趨勢強度 gate:|close-EMA200|/ATR 須 ≥ 此值才進場(None=關閉)

    def current_signals(self, data_by_coin: Dict[str, pd.DataFrame]) -> List[dict]:
        """回傳「這一根新出現」的可執行訊號事件清單。

        data_by_coin: {coin: 1h(或更細) OHLCV DataFrame}。各腿會自行 resample 到其週期。
        """
        events: List[dict] = []
        for leg in self.legs:
            raw = data_by_coin.get(leg.coin)
            if raw is None or len(raw) < 50:
                continue
            d = resample_ohlcv(raw, leg.timeframe)
            sig = leg.signal_fn(d)
            if self.regime_ema is not None:
                sig = regime_filter(d, sig, self.regime_ema)
            if self.dist_gate_atr is not None:
                sig = trend_distance_gate(
                    d, sig, ema_span=self.regime_ema or 200,
                    min_atr_mult=self.dist_gate_atr,
                )
            if len(sig) < 2:
                continue
            now = int(sig.iloc[-1])
            prev = int(sig.iloc[-2])
            # 只在「新翻成多/空」的那一根通知(0或反向 → 多/空)
            if now != 0 and now != prev:
                price = float(d["close"].iloc[-1])
                atr_now = float(atr(d, self.atr_period).iloc[-1])
                init_stop = (
                    price * (1 - self.stop_loss_pct)
                    if now > 0
                    else price * (1 + self.stop_loss_pct)
                )
                # 建議名目部位(佔權益比例)= 每筆風險 ÷ 停損距離,使停損被打到時恰賠 risk_per_trade
                notional_pct = round(self.risk_per_trade / self.stop_loss_pct, 2)
                events.append(
                    {
                        "coin": leg.coin,
                        "timeframe": leg.timeframe,
                        "signal_name": leg.name,
                        "direction": "LONG" if now > 0 else "SHORT",
                        "price": price,
                        "time": d["timestamp"].iloc[-1],
                        "init_stop": round(init_stop, 6),
                        "atr": round(atr_now, 6),
                        "risk_pct": self.risk_per_trade,
                        "notional_pct_of_equity": notional_pct,
                        "trailing_hint": f"{self.trailing_atr_mult}×ATR 移動停損",
                    }
                )
        return events
