#!/usr/bin/env python3
"""誠實回測引擎的單元測試。

每個案例都用「手算的已知答案」驗證,確保引擎的成交時點、停損停利、成本、
無前視偏誤等行為完全正確。使用者無法自行 audit,故驗證責任在測試。
"""

import math

import pandas as pd
import pytest

from engine.backtester import (
    EXIT_END_OF_DATA,
    EXIT_SIGNAL,
    EXIT_STOP_LOSS,
    EXIT_TAKE_PROFIT,
    EXIT_TIME_STOP,
    EventBacktester,
)
from engine.metrics import compute_metrics, max_drawdown


def _df(rows):
    """rows: list of (open, high, low, close)。timestamp 用整數序。"""
    return pd.DataFrame(
        {
            "timestamp": list(range(len(rows))),
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
        }
    )


# ---------- 無成本引擎,方便手算 ----------
def _bt(**kw):
    defaults = dict(
        fee_rate=0.0,
        slippage_rate=0.0,
        position_size_ratio=1.0,
        leverage=1.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        initial_capital=10000.0,
    )
    defaults.update(kw)
    return EventBacktester(**defaults)


def test_entry_is_next_bar_open_not_signal_bar_close():
    """訊號在第 0 根 → 進場必須在第 1 根 open,不是第 0 根 close。"""
    df = _df([(100, 100, 100, 100), (123, 123, 123, 123), (123, 125, 96, 100)])
    signals = pd.Series([1, 0, 0])
    res = _bt(stop_loss_pct=None, take_profit_pct=None).run(df, signals)
    assert res.trades[0].entry_idx == 1
    assert res.trades[0].entry_price == 123  # 第1根 open


def test_signal_on_last_bar_produces_no_trade():
    """最後一根才出訊號 → 沒有下一根可成交 → 不交易(無前視)。"""
    df = _df([(100, 100, 100, 100), (100, 100, 100, 100)])
    signals = pd.Series([0, 1])
    res = _bt().run(df, signals)
    assert res.trades == []


def test_long_take_profit_intrabar():
    """做多,第1根盤中觸及 +4% 停利 → 以停利價成交。"""
    df = _df([(100, 100, 100, 100), (100, 110, 99, 105)])
    signals = pd.Series([1, 0])
    res = _bt().run(df, signals)
    t = res.trades[0]
    assert t.exit_reason == EXIT_TAKE_PROFIT
    assert t.exit_price == pytest.approx(104.0)  # 100*1.04
    assert t.gross_return_pct == pytest.approx(0.04)
    assert t.equity_after == pytest.approx(10400.0)


def test_long_stop_loss_intrabar():
    """做多,第1根盤中跌破 -2% 停損 → 以停損價成交。"""
    df = _df([(100, 100, 100, 100), (100, 101, 97, 98)])
    signals = pd.Series([1, 0])
    res = _bt().run(df, signals)
    t = res.trades[0]
    assert t.exit_reason == EXIT_STOP_LOSS
    assert t.exit_price == pytest.approx(98.0)
    assert t.net_return_pct == pytest.approx(-0.02)


def test_long_stop_loss_gap_fills_at_open():
    """進場後某根開盤直接跳空跌破停損 → 以 open 成交(比停損價更差)。"""
    # bar0 訊號 → bar1 進場 open=100(SL=98),bar2 開盤跳空到 97(< 98)
    df = _df([(100, 100, 100, 100), (100, 101, 99, 100), (97, 98, 95, 96)])
    signals = pd.Series([1, 0, 0])
    res = _bt().run(df, signals)
    t = res.trades[0]
    assert t.entry_price == 100
    assert t.exit_reason == EXIT_STOP_LOSS
    assert t.exit_price == pytest.approx(97.0)  # open,不是 98
    assert t.gross_return_pct == pytest.approx(-0.03)


def test_same_bar_sl_and_tp_is_conservative_stop_loss():
    """同一根同時觸及停損與停利 → 保守假設停損先成交。"""
    df = _df([(100, 100, 100, 100), (100, 110, 90, 100)])
    signals = pd.Series([1, 0])
    res = _bt().run(df, signals)
    assert res.trades[0].exit_reason == EXIT_STOP_LOSS
    assert res.trades[0].exit_price == pytest.approx(98.0)


def test_short_take_profit():
    """做空,第1根盤中跌到 -4% 停利價(預設 TP=4%)→ 做空獲利 +4%。"""
    # 做空進場 100,TP=100*(1-0.04)=96;bar1 low=95 觸及 → 以 96 成交
    df = _df([(100, 100, 100, 100), (100, 101, 95, 97)])
    signals = pd.Series([-1, 0])
    res = _bt().run(df, signals)
    t = res.trades[0]
    assert t.direction == -1
    assert t.exit_reason == EXIT_TAKE_PROFIT
    assert t.exit_price == pytest.approx(96.0)
    assert t.gross_return_pct == pytest.approx(0.04)


def test_opposite_signal_exit_next_open():
    """無停損停利,收到反向訊號 → 下一根 open 平倉。"""
    df = _df(
        [
            (100, 100, 100, 100),  # 0: 訊號 +1
            (100, 101, 99, 100),  # 1: 進場 open=100
            (103, 104, 102, 103),  # 2: 反向訊號 -1
            (106, 107, 105, 106),  # 3: 退場 open=106
        ]
    )
    signals = pd.Series([1, 0, -1, 0])
    res = _bt(stop_loss_pct=None, take_profit_pct=None).run(df, signals)
    t = res.trades[0]
    assert t.exit_reason == EXIT_SIGNAL
    assert t.exit_idx == 3
    assert t.exit_price == pytest.approx(106.0)
    assert t.gross_return_pct == pytest.approx(0.06)


def test_time_stop_exit():
    """最長持倉 2 根,無停損停利反向 → 第 2 根持滿後下一根 open 平倉。"""
    df = _df(
        [
            (100, 100, 100, 100),  # 0 訊號+1
            (100, 100, 100, 100),  # 1 進場
            (101, 101, 101, 101),  # 2 held=1
            (102, 102, 102, 102),  # 3 held=2 觸發
            (105, 105, 105, 105),  # 4 退場 open=105
        ]
    )
    signals = pd.Series([1, 0, 0, 0, 0])
    res = _bt(
        stop_loss_pct=None, take_profit_pct=None, max_hold_bars=2, allow_opposite_exit=False
    ).run(df, signals)
    t = res.trades[0]
    assert t.exit_reason == EXIT_TIME_STOP
    assert t.exit_price == pytest.approx(105.0)


def test_end_of_data_exit_at_last_close():
    """資料用盡仍持倉 → 以最後一根 close 平倉並標記。"""
    df = _df([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 108)])
    signals = pd.Series([1, 0, 0])
    res = _bt(stop_loss_pct=None, take_profit_pct=None, allow_opposite_exit=False).run(
        df, signals
    )
    t = res.trades[0]
    assert t.exit_reason == EXIT_END_OF_DATA
    assert t.exit_price == pytest.approx(108.0)
    assert res.open_position_at_end is False  # 已結算


def test_costs_applied_both_sides():
    """手續費+滑價兩邊都收,手算驗證淨損益。"""
    df = _df([(100, 100, 100, 100), (100, 110, 99, 105)])
    signals = pd.Series([1, 0])
    bt = _bt(fee_rate=0.0004, slippage_rate=0.0002)  # 單邊 0.0006
    res = bt.run(df, signals)
    t = res.trades[0]
    # 名目=10000,qty=100,停利價=104
    # 毛利=0.04*10000=400;進場成本=10000*0.0006=6;出場成本=100*104*0.0006=6.24
    assert t.cost == pytest.approx(12.24)
    assert t.net_pnl == pytest.approx(400 - 12.24)
    assert t.equity_after == pytest.approx(10000 + 387.76)


def test_leverage_scales_pnl():
    """槓桿放大帳戶報酬:3x 槓桿下 +4% 價格變動 → 帳戶 +12%(未扣成本)。"""
    df = _df([(100, 100, 100, 100), (100, 110, 99, 105)])
    signals = pd.Series([1, 0])
    res = _bt(leverage=3.0, position_size_ratio=1.0).run(df, signals)
    assert res.trades[0].net_return_pct == pytest.approx(0.12)


def test_trailing_stop_locks_in_profit():
    """移動停損:上漲到 104 後回落,於「峰值 - 1*ATR」出場鎖住獲利。

    每根真實區間固定為 2,故 ATR 恆等於 2;trailing_atr_mult=1 → 追蹤距離=2。
    進場 100,峰值到 104(bar3 high),bar4 回落 low=101 ≤ 104-2=102 → 以 102 停損出場 = +2%。
    """
    df = _df(
        [
            (100, 101, 99, 100),  # 0 訊號 +1
            (100, 102, 100, 101),  # 1 進場 open=100
            (101, 103, 101, 102),  # 2 best→103
            (102, 104, 102, 103),  # 3 best→104(峰值)
            (103, 103, 101, 101),  # 4 trail=104-2=102,low=101 觸及 → 102 出場
        ]
    )
    signals = pd.Series([1, 0, 0, 0, 0])
    bt = _bt(
        stop_loss_pct=0.05,  # 初始硬停損 95,遠低於 trailing
        take_profit_pct=None,  # 不封頂,讓獲利奔跑
        trailing_atr_mult=1.0,
        atr_period=14,
        allow_opposite_exit=False,
    )
    res = bt.run(df, signals)
    t = res.trades[0]
    assert t.exit_reason == EXIT_STOP_LOSS
    assert t.exit_idx == 4
    assert t.exit_price == pytest.approx(102.0)
    assert t.gross_return_pct == pytest.approx(0.02)


def test_trailing_stop_does_not_trigger_while_trending():
    """持續上漲期間移動停損不應觸發(peak 與價格同步上移)。"""
    df = _df(
        [
            (100, 101, 99, 100),  # 0 訊號+1
            (100, 102, 100, 101),  # 1 進場
            (101, 103, 101, 102),  # 2
            (102, 104, 102, 103),  # 3
            (103, 105, 103, 104),  # 4 仍在漲,low=103 > 104-2=102
        ]
    )
    signals = pd.Series([1, 0, 0, 0, 0])
    bt = _bt(
        stop_loss_pct=0.05, take_profit_pct=None, trailing_atr_mult=1.0,
        allow_opposite_exit=False,
    )
    res = bt.run(df, signals)
    # 一路漲到資料結束,應以最後 close 平倉(未被 trailing 掃出)
    assert res.trades[0].exit_reason == EXIT_END_OF_DATA
    assert res.trades[0].exit_price == pytest.approx(104.0)


def test_risk_based_sizing_caps_loss():
    """風險基礎部位:每筆 risk 1%、初始停損 5% → 停損被打到時剛好虧 1% 權益。"""
    # 進場 100,停損 95;bar2 跌破 → 虧損應 = 1% 權益
    df = _df([(100, 100, 100, 100), (100, 101, 99, 100), (95, 96, 90, 92)])
    signals = pd.Series([1, 0, 0])
    bt = _bt(
        stop_loss_pct=0.05, take_profit_pct=None, risk_per_trade_pct=0.01,
        leverage=10.0, allow_opposite_exit=False,
    )
    res = bt.run(df, signals)
    t = res.trades[0]
    # 名目 = 0.01*10000/0.05 = 2000(未達 10x 上限 100000);停損跳空到 95 以 95 成交
    assert t.notional == pytest.approx(2000.0)
    assert t.exit_reason == EXIT_STOP_LOSS
    assert t.exit_price == pytest.approx(95.0)
    # gross = -5%,net 虧損 ≈ 2000*0.05 = 100 = 權益 1%
    assert t.net_return_pct == pytest.approx(-0.01)


def test_risk_based_sizing_respects_leverage_cap():
    """風險反推的名目若超過 equity*leverage,應被槓桿上限夾住。"""
    df = _df([(100, 100, 100, 100), (100, 110, 99, 105)])
    signals = pd.Series([1, 0])
    # risk 1% / 停損 0.1% → 反推名目=0.01*10000/0.001=100000,但 2x 上限=20000 → 夾到 20000
    bt = _bt(
        stop_loss_pct=0.001, take_profit_pct=0.04, risk_per_trade_pct=0.01, leverage=2.0,
    )
    res = bt.run(df, signals)
    assert res.trades[0].notional == pytest.approx(20000.0)


def test_compute_metrics_hand_values():
    """手算驗證 compute_metrics 的各欄位。"""
    returns = [0.1, -0.2, 0.1]
    equity = [100.0, 110.0, 88.0, 96.8]
    m = compute_metrics(returns, equity, initial_equity=100.0)
    assert m.n_trades == 3
    assert m.win_rate == pytest.approx(2 / 3)
    assert m.expectancy_pct == pytest.approx(0.0)
    assert m.avg_win_pct == pytest.approx(0.1)
    assert m.avg_loss_pct == pytest.approx(-0.2)
    assert m.payoff_ratio == pytest.approx(0.5)
    assert m.profit_factor == pytest.approx(1.0)
    assert m.max_drawdown == pytest.approx(0.2)  # (110-88)/110
    assert m.total_return == pytest.approx(-0.032)  # 96.8/100-1


def test_max_drawdown_function():
    assert max_drawdown([100, 120, 90, 150, 75]) == pytest.approx(0.5)  # (150-75)/150
    assert max_drawdown([100, 110, 120]) == pytest.approx(0.0)  # 一路創高
    assert max_drawdown([]) == 0.0


def test_no_signals_no_trades():
    df = _df([(100, 100, 100, 100)] * 5)
    res = _bt().run(df, pd.Series([0] * 5))
    assert res.trades == []
    assert res.metrics.n_trades == 0
    assert res.metrics.total_return == pytest.approx(0.0)


def test_multiple_sequential_trades_compound():
    """連續兩筆獲利交易,權益應複利累積。"""
    df = _df(
        [
            (100, 100, 100, 100),  # 0 訊號+1
            (100, 110, 99, 105),  # 1 進場100,停利104 → equity 10400
            (104, 104, 104, 104),  # 2 訊號+1(新)
            (104, 115, 103, 110),  # 3 進場104,停利 104*1.04=108.16 → +4%
        ]
    )
    signals = pd.Series([1, 0, 1, 0])
    res = _bt().run(df, signals)
    assert len(res.trades) == 2
    assert res.trades[0].equity_after == pytest.approx(10400.0)
    assert res.trades[1].equity_after == pytest.approx(10400.0 * 1.04)
