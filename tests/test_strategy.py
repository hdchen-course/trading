#!/usr/bin/env python3
"""TrendPortfolio.current_signals 的轉換邏輯測試:只在「新翻多/空」那根才通知。"""

import numpy as np
import pandas as pd
import pytest

from engine.strategy import Leg, TrendPortfolio, default_legs, equity_legs


def _raw(n=60):
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    price = np.linspace(100, 110, n)
    return pd.DataFrame(
        {"timestamp": ts, "open": price, "high": price + 1, "low": price - 1,
         "close": price, "volume": 1.0}
    )


def _portfolio_with_tail(last, prev):
    """造一個固定訊號(最後兩根為 prev,last)的 Leg,其餘為 0。"""
    def fn(d):
        s = pd.Series(0, index=d.index, dtype=int)
        if len(s) >= 2:
            s.iloc[-2] = prev
            s.iloc[-1] = last
        return s

    leg = Leg(coin="BTCUSDT", timeframe="1h", name="test", signal_fn=fn)
    # 關閉 regime 與距離 gate 以隔離「轉換偵測」邏輯(gate 另有專屬測試)
    return TrendPortfolio(legs=[leg], stop_loss_pct=0.06, regime_ema=None, dist_gate_atr=None)


def test_fresh_long_triggers_event():
    p = _portfolio_with_tail(last=1, prev=0)
    ev = p.current_signals({"BTCUSDT": _raw()})
    assert len(ev) == 1
    assert ev[0]["direction"] == "LONG"
    assert ev[0]["init_stop"] < ev[0]["price"]  # 多單停損在下方


def test_continuing_signal_no_event():
    """訊號延續(last==prev)不應重複通知。"""
    p = _portfolio_with_tail(last=1, prev=1)
    assert p.current_signals({"BTCUSDT": _raw()}) == []


def test_flip_to_short_triggers_event():
    p = _portfolio_with_tail(last=-1, prev=1)
    ev = p.current_signals({"BTCUSDT": _raw()})
    assert len(ev) == 1
    assert ev[0]["direction"] == "SHORT"
    assert ev[0]["init_stop"] > ev[0]["price"]  # 空單停損在上方


def test_no_signal_no_event():
    p = _portfolio_with_tail(last=0, prev=0)
    assert p.current_signals({"BTCUSDT": _raw()}) == []


def test_missing_coin_is_skipped():
    p = _portfolio_with_tail(last=1, prev=0)
    assert p.current_signals({}) == []  # 沒資料不報錯


def test_current_states_reports_gated_state():
    """current_states 回報每條腿當前 gated 狀態(此處關 gate,直接反映訊號末值)。"""
    def fn(d):
        s = pd.Series(0, index=d.index, dtype=int)
        s.iloc[-1] = 1
        return s

    leg = Leg(coin="BTCUSDT", timeframe="1h", name="t", signal_fn=fn)
    p = TrendPortfolio(legs=[leg], regime_ema=None, dist_gate_atr=None)
    states = p.current_states({"BTCUSDT": _raw()})
    assert len(states) == 1
    assert states[0]["state"] == 1
    assert states[0]["coin"] == "BTCUSDT"


def test_default_and_equity_legs_shapes():
    """加密腿 3×幣(4h/8h/12h);美股腿 3×標的(全日線)。"""
    c = default_legs(["BTCUSDT", "AAVEUSDT"])
    assert len(c) == 6
    assert {leg.timeframe for leg in c} == {"4h", "8h", "12h"}
    e = equity_legs(["SOXL", "TQQQ"])
    assert len(e) == 6
    assert {leg.timeframe for leg in e} == {"1d"}
    assert {leg.coin for leg in e} == {"SOXL", "TQQQ"}


def _const_portfolio(state, n_tail, max_hold):
    """造一條『末 n_tail 根都是 state』的腿,關 gate,設 max_hold。"""
    def fn(d):
        s = pd.Series(0, index=d.index, dtype=int)
        s.iloc[-n_tail:] = state
        return s
    leg = Leg(coin="BTCUSDT", timeframe="1h", name="t", signal_fn=fn)
    return TrendPortfolio(legs=[leg], regime_ema=None, dist_gate_atr=None,
                          max_hold_bars=max_hold)


def test_current_exits_fires_when_held_exceeds_max_hold():
    """持倉同向根數 ≥ max_hold → 發時間平倉提醒。"""
    p = _const_portfolio(state=1, n_tail=30, max_hold=10)
    ex = p.current_exits({"BTCUSDT": _raw()})
    assert len(ex) == 1
    assert ex[0]["action"] == "TIME_EXIT"
    assert ex[0]["direction"] == "LONG"
    assert ex[0]["held_bars"] >= 10


def test_current_exits_silent_when_held_below_max_hold():
    """持倉未滿 max_hold → 不發。"""
    p = _const_portfolio(state=1, n_tail=4, max_hold=10)
    assert p.current_exits({"BTCUSDT": _raw()}) == []


def test_current_exits_only_at_rollover_multiples():
    """只在換倉點(max_hold 整數倍)發,非整數倍不發(避免持續趨勢每根狂噴)。"""
    # held=13, max_hold=10 → 13%10≠0 → 不發
    assert _const_portfolio(state=1, n_tail=13, max_hold=10).current_exits(
        {"BTCUSDT": _raw()}) == []
    # held=20, max_hold=10 → 20%10==0 → 發
    assert len(_const_portfolio(state=1, n_tail=20, max_hold=10).current_exits(
        {"BTCUSDT": _raw()})) == 1


def test_current_exits_none_when_flat_or_disabled():
    p_flat = _const_portfolio(state=0, n_tail=30, max_hold=10)
    assert p_flat.current_exits({"BTCUSDT": _raw()}) == []
    p_off = _const_portfolio(state=1, n_tail=30, max_hold=None)
    assert p_off.current_exits({"BTCUSDT": _raw()}) == []


def test_meta_model_tilts_position_size(tmp_path):
    """掛上 meta 模型後,current_signals 事件應帶 quality_mult 並據此傾斜建議部位。"""
    import numpy as np
    from engine.meta_model import FEATURE_COLS, train

    rng = np.random.default_rng(0)
    Xtr = pd.DataFrame({c: rng.normal(0, 1, 500) for c in FEATURE_COLS})
    ytr = (rng.uniform(0, 1, 500) < 1 / (1 + np.exp(-Xtr["dist_ema200"]))).astype(int)
    mp = tmp_path / "m.pkl"
    train(Xtr, ytr, kind="gbm").save(mp)

    def fn(d):
        s = pd.Series(0, index=d.index, dtype=int)
        s.iloc[-1] = 1  # 末根新翻多
        return s

    leg = Leg(coin="BTCUSDT", timeframe="1h", name="t", signal_fn=fn)
    p = TrendPortfolio(legs=[leg], regime_ema=None, dist_gate_atr=None,
                       stop_loss_pct=0.06, risk_per_trade=0.02, meta_model_path=str(mp))
    ev = p.current_signals({"BTCUSDT": _raw()})
    assert len(ev) == 1
    qm = ev[0]["quality_mult"]
    assert 0.3 - 1e-9 <= qm <= 2.0 + 1e-9
    # 建議部位 = 基礎(0.02/0.06≈0.333)× quality_mult
    assert ev[0]["notional_pct_of_equity"] == pytest.approx(round(0.333 * qm, 3), abs=0.01)


def test_entry_event_carries_scale_out_hint():
    """進場事件應帶分批放目標與提示:多單目標在上方(+20%),空單在下方。"""
    p_long = _portfolio_with_tail(last=1, prev=0)
    ev = p_long.current_signals({"BTCUSDT": _raw()})[0]
    assert ev["scale_out_frac"] == 0.5
    assert ev["scale_out_target"] == pytest.approx(ev["price"] * 1.20, rel=1e-6)  # 多單 +20%
    assert "分批放" not in ev["scale_out_hint"]  # 提示本身不含前綴(由顯示端加)
    assert "減碼" in ev["scale_out_hint"]

    p_short = _portfolio_with_tail(last=-1, prev=0)
    # 關 regime 才能讓空單在上升樣本過(_portfolio_with_tail 已關 gate)
    evs = p_short.current_signals({"BTCUSDT": _raw()})
    assert evs[0]["scale_out_target"] == pytest.approx(evs[0]["price"] * 0.80, rel=1e-6)  # 空單 -20%


def test_regime_filter_blocks_countertrend_short_in_uptrend():
    """預設 regime 濾網開啟:上升盤中的新空單應被濾掉(不產生事件)。"""
    def fn(d):
        s = pd.Series(0, index=d.index, dtype=int)
        s.iloc[-2] = 1
        s.iloc[-1] = -1  # 翻空
        return s

    leg = Leg(coin="BTCUSDT", timeframe="1h", name="t", signal_fn=fn)
    p = TrendPortfolio(legs=[leg], regime_ema=200, dist_gate_atr=None)  # 只測 regime
    # _raw 是嚴格上升盤 → close > EMA200 → 空單被 regime 濾掉
    assert p.current_signals({"BTCUSDT": _raw()}) == []
