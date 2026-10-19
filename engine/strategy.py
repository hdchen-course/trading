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
from typing import Callable, Dict, List, Optional

import pandas as pd

from engine.data import atr, resample_ohlcv
from engine.features import build_feature_frame
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
    """加密標的(24h):每個幣套用三個通過樣本外驗證的高週期順勢訊號(4h/8h/12h)。"""
    legs: List[Leg] = []
    for coin in coins:
        legs.append(Leg(coin, "4h", "EMA50/200", lambda d: ema_cross(d, 50, 200)))
        legs.append(Leg(coin, "8h", "Donchian20", lambda d: donchian_breakout(d, 20)))
        legs.append(
            Leg(coin, "12h", "ribbon",
                lambda d: ema_ribbon_trend(d, (90, 120, 240, 300, 420)))
        )
    return legs


def equity_legs(symbols: List[str]) -> List[Leg]:
    """美股/ETF(非 24h):日線三個順勢訊號(EMA50/200、EMA20/50、Donchian20)。"""
    legs: List[Leg] = []
    for s in symbols:
        legs.append(Leg(s, "1d", "EMA50/200", lambda d: ema_cross(d, 50, 200)))
        legs.append(Leg(s, "1d", "EMA20/50", lambda d: ema_cross(d, 20, 50)))
        legs.append(Leg(s, "1d", "Donchian20", lambda d: donchian_breakout(d, 20)))
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
    max_hold_bars: Optional[int] = 80  # 時間停損:持倉滿此根數(該腿週期)建議平倉(回測驗證的 edge,None=關閉)
    meta_model_path: Optional[str] = None  # 交易品質 meta 模型路徑;設了則用它對訊號做部位倍率傾斜(None=關閉)
    _meta: object = field(default=None, init=False, repr=False, compare=False)
    _meta_loaded: bool = field(default=False, init=False, repr=False, compare=False)

    def _get_meta(self):
        """延遲載入 meta 模型;載入失敗(無檔/無 sklearn)→ None,不影響主流程。"""
        if not self._meta_loaded:
            self._meta_loaded = True
            if self.meta_model_path:
                try:
                    from engine.meta_model import MetaModel
                    self._meta = MetaModel.load(self.meta_model_path)
                except Exception:
                    self._meta = None
        return self._meta

    def _gated_signal(self, leg: Leg, raw: pd.DataFrame):
        """對單一腿算出「套完 regime + 距離 gate」的訊號序列。回傳 (d, sig) 或 None。

        current_signals(新觸發事件)與 current_states(當前狀態)共用,避免 gate 邏輯重複。
        """
        if raw is None or len(raw) < 50:
            return None
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
            return None
        return d, sig

    def current_states(self, data_by_coin: Dict[str, pd.DataFrame]) -> List[dict]:
        """回傳每條腿「當前 gated 訊號狀態」(1/0/-1),供儀表板總覽(非只看新觸發)。"""
        states: List[dict] = []
        for leg in self.legs:
            res = self._gated_signal(leg, data_by_coin.get(leg.coin))
            if res is None:
                continue
            d, sig = res
            states.append({
                "coin": leg.coin, "timeframe": leg.timeframe,
                "signal_name": leg.name, "state": int(sig.iloc[-1]),
                "price": float(d["close"].iloc[-1]),
            })
        return states

    @staticmethod
    def _bars_in_state(sig: pd.Series) -> int:
        """當前(末根)非零狀態已持續幾根(尾端同號連續數)。末根為 0 回傳 0。"""
        vals = sig.to_numpy()
        last = vals[-1]
        if last == 0:
            return 0
        n = 0
        for v in vals[::-1]:
            if v == last:
                n += 1
            else:
                break
        return n

    def current_exits(self, data_by_coin: Dict[str, pd.DataFrame]) -> List[dict]:
        """回傳「持倉已滿 max_hold_bars 根、建議時間平倉」的腿(落地回測的時間停損 edge)。

        以訊號序列尾端同向連續根數當持倉年齡(無狀態、無前視)。到期發平倉提醒;若趨勢續行,
        下一次新翻轉會再由 current_signals 給進場訊號(與回測 max_hold 的「到期換倉」一致)。
        """
        if self.max_hold_bars is None:
            return []
        exits: List[dict] = []
        for leg in self.legs:
            res = self._gated_signal(leg, data_by_coin.get(leg.coin))
            if res is None:
                continue
            d, sig = res
            state = int(sig.iloc[-1])
            if state == 0:
                continue
            held = self._bars_in_state(sig)
            # 只在「換倉點」(持倉=max_hold 的整數倍:80/160/240…)各發一次,避免持續趨勢每根狂噴;
            # 對應回測「到期平倉+重進、計時歸零」語義。假設 live 至少每根(該週期)跑一次。
            if held >= self.max_hold_bars and held % self.max_hold_bars == 0:
                exits.append({
                    "coin": leg.coin, "timeframe": leg.timeframe,
                    "signal_name": leg.name,
                    "direction": "LONG" if state > 0 else "SHORT",
                    "action": "TIME_EXIT",
                    "held_bars": held,
                    "price": float(d["close"].iloc[-1]),
                    "time": d["timestamp"].iloc[-1],
                    "note": f"持倉已滿 {held} 根({leg.timeframe})≥{self.max_hold_bars},"
                            f"建議平倉;趨勢若續行待新訊號再進",
                })
        return exits

    def current_signals(self, data_by_coin: Dict[str, pd.DataFrame]) -> List[dict]:
        """回傳「這一根新出現」的可執行訊號事件清單。

        data_by_coin: {coin: 1h(或更細) OHLCV DataFrame}。各腿會自行 resample 到其週期。
        """
        events: List[dict] = []
        for leg in self.legs:
            res = self._gated_signal(leg, data_by_coin.get(leg.coin))
            if res is None:
                continue
            d, sig = res
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
                # 交易品質 meta 傾斜:高把握加碼、低把握減碼(軟性,平均≈1;驗證見 meta_portfolio_study)
                quality_mult = 1.0
                meta = self._get_meta()
                if meta is not None:
                    try:
                        row = build_feature_frame(d).iloc[[-1]]
                        quality_mult = round(float(meta.size_multiplier(row)[0]), 3)
                    except Exception:
                        quality_mult = 1.0
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
                        "quality_mult": quality_mult,
                        "notional_pct_of_equity": round(notional_pct * quality_mult, 3),
                        "trailing_hint": f"{self.trailing_atr_mult}×ATR 移動停損",
                    }
                )
        return events
