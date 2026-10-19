#!/usr/bin/env python3
"""事件驅動的誠實回測器 (event-driven honest backtester)。

與舊 trading/backtest.py 的關鍵差異(為何這個才可信):
1. **下一根 open 成交**:訊號在第 i 根「收盤後」才算得出,所以進出場一律在第 i+1 根的
   open 價,絕不用「剛收盤那根的 close」當成交價(舊版的隱性樂觀偏差)。
2. **盤中 high/low 判停損停利**:停損/停利用每根的 high/low 判定是否觸及,而非只看 close。
   若同一根同時觸及停損與停利,保守假設「停損先成交」(對結果不利的一邊)。
3. **跳空照實成交**:若開盤就跳過停損價,以 open 成交(比停損價更差),反映真實滑價風險。
4. **手續費 + 滑價必收**:每邊都收,預設貼近幣安 USDT 永續 taker 費率。

單位約定:所有 *_pct(停損/停利)都是「價格變動比率」,不除以槓桿;槓桿透過名目部位放大
帳戶損益。這比舊版「停損 ÷ 槓桿」的寫法更直覺也更誠實。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from engine.metrics import PerformanceMetrics, compute_metrics

# 退場原因常數(避免魔術字串)
EXIT_STOP_LOSS = "stop_loss"
EXIT_TAKE_PROFIT = "take_profit"
EXIT_SIGNAL = "signal"
EXIT_TIME_STOP = "time_stop"
EXIT_END_OF_DATA = "end_of_data"

LONG = 1
SHORT = -1


@dataclass
class Trade:
    """一筆完整交易的紀錄。pct 欄位為帳戶權益口徑。"""

    direction: int  # LONG(+1) 或 SHORT(-1)
    entry_time: object
    entry_price: float
    entry_idx: int
    exit_time: object
    exit_price: float
    exit_idx: int
    exit_reason: str
    notional: float  # 進場名目部位 = 進場權益 * position_size_ratio * leverage
    gross_return_pct: float  # 價格方向變動比率(未扣成本)
    cost: float  # 這筆交易的總成本金額(手續費+滑價,兩邊)
    net_pnl: float  # 淨損益金額(已扣成本)
    net_return_pct: float  # 淨損益 / 進場前權益
    hold_bars: int
    equity_after: float  # 平倉後帳戶權益


@dataclass
class BacktestResult:
    """回測輸出:交易清單、權益曲線、績效指標。"""

    trades: List[Trade]
    equity_curve: pd.Series  # index=平倉時間(含起始點),value=權益
    metrics: PerformanceMetrics
    open_position_at_end: bool = False

    def trades_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in self.trades])


@dataclass
class EventBacktester:
    """事件驅動回測器。

    參數:
        fee_rate: 單邊手續費率(名目部位),預設 0.0004(幣安 USDT 永續 taker ~0.04%)。
        slippage_rate: 單邊滑價率(名目部位),預設 0.0002(2 bps)。
        position_size_ratio: 每筆交易投入的保證金佔當前權益比例。
        leverage: 槓桿倍數,放大名目部位與損益。
        stop_loss_pct: 停損(價格變動比率),None 表示不設停損。
        take_profit_pct: 停利(價格變動比率),None 表示不設停利。
        max_hold_bars: 最長持倉根數,超過則在下一根 open 以時間停損平倉;None 表示不限。
        allow_opposite_exit: 收到反向訊號時是否平倉(於下一根 open)。
        initial_capital: 初始本金。
    """

    fee_rate: float = 0.0004
    slippage_rate: float = 0.0002
    position_size_ratio: float = 0.25
    leverage: float = 1.0
    stop_loss_pct: Optional[float] = 0.02
    take_profit_pct: Optional[float] = 0.04
    max_hold_bars: Optional[int] = None
    allow_opposite_exit: bool = True
    initial_capital: float = 10000.0
    # 移動停損(trailing stop):設 trailing_atr_mult 後啟用,停損價 = 進場後最佳價 ∓ mult*ATR,
    # 隨獲利方向單向收緊(只會往有利方向移動),讓順勢的大賺單奔跑。與 stop_loss_pct(初始硬停損)
    # 取較嚴格者。啟用 trailing 時 take_profit 仍可並存(通常設 None 讓獲利不封頂)。
    trailing_atr_mult: Optional[float] = None
    atr_period: int = 14
    # 風險基礎部位控管:設 risk_per_trade_pct(例如 0.01=每筆賭 1% 權益)後,名目部位依初始停損
    # 距離反推,使「停損被打到」時的虧損恰為 risk_per_trade_pct*權益 → 直接控制回撤。
    # 需同時設 stop_loss_pct。名目不超過 equity*leverage(槓桿上限)。設 None 則用 position_size_ratio。
    risk_per_trade_pct: Optional[float] = None

    @property
    def _cost_rate(self) -> float:
        """單邊總成本率 = 手續費 + 滑價。"""
        return self.fee_rate + self.slippage_rate

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        size_scale: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """執行回測。

        參數:
            df: 需含欄位 open/high/low/close,以及(可選)timestamp。索引須為 0..n-1。
            signals: 與 df 等長、同索引的訊號序列,值 ∈ {-1, 0, +1},代表「該根收盤後」
                     產生的訊號。+1=做多、-1=做空、0=無。
            size_scale: (可選)與 df 等長的逐根部位倍率,在「進場那根」乘上名目部位
                     (例如波動度目標化:低波加倉、高波減倉)。須為 t 當下可得(無前視,
                     由呼叫端負責 shift)。None 表示一律 1.0。放大後仍受 equity*leverage 上限。

        回傳 BacktestResult。
        """
        df = df.reset_index(drop=True)
        signals = pd.Series(list(signals), index=df.index).fillna(0).astype(int)
        if size_scale is None:
            scale_arr = [1.0] * len(df)
        else:
            scale_arr = (
                pd.Series(list(size_scale), index=df.index)
                .astype(float)
                .clip(lower=0.0)
                .fillna(0.0)
                .tolist()
            )

        times = (
            df["timestamp"] if "timestamp" in df.columns else pd.Series(df.index)
        )
        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        n = len(df)

        # 啟用移動停損時,預先算好每根的 ATR(trailing,無前視)
        atr_vals = None
        if self.trailing_atr_mult is not None:
            from engine.data import atr as _atr

            atr_vals = _atr(df, self.atr_period).to_numpy(dtype=float)

        equity = self.initial_capital
        trades: List[Trade] = []
        equity_points: List[float] = [equity]
        equity_times: List[object] = [times.iloc[0] if n > 0 else 0]

        position = None  # 進行中的部位
        i = 0
        # 逐根推進。訊號在第 i 根收盤後出現 → 進場在第 i+1 根 open。
        while i < n:
            if position is None:
                sig = int(signals.iloc[i])
                if sig != 0 and i + 1 < n:
                    entry_idx = i + 1
                    entry_price = opens[entry_idx]
                    notional = self._position_notional(equity, scale_arr[entry_idx])
                    position = {
                        "direction": sig,
                        "entry_idx": entry_idx,
                        "entry_price": entry_price,
                        "entry_time": times.iloc[entry_idx],
                        "notional": notional,
                        "equity_before": equity,
                    }
                    # 從進場當根(entry_idx)開始檢查退場
                    exit_info = self._scan_exit(
                        position, entry_idx, opens, highs, lows, closes, signals, n,
                        times, atr_vals,
                    )
                    trade = self._close_trade(position, exit_info, equity)
                    equity = trade.equity_after
                    trades.append(trade)
                    equity_points.append(equity)
                    equity_times.append(trade.exit_time)
                    # 下一輪從退場那根之後繼續找新訊號
                    i = trade.exit_idx
                    position = None
                    continue
            i += 1

        open_at_end = position is not None
        equity_curve = pd.Series(equity_points, index=pd.Index(equity_times))
        metrics = compute_metrics(
            trade_returns_pct=[t.net_return_pct for t in trades],
            equity_curve=equity_points,
            initial_equity=self.initial_capital,
            total_cost=sum(t.cost for t in trades),
            hold_bars=[t.hold_bars for t in trades],
        )
        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
            open_position_at_end=open_at_end,
        )

    def _scan_exit(
        self, position, entry_idx, opens, highs, lows, closes, signals, n, times,
        atr_vals=None,
    ) -> dict:
        """從 entry_idx 起逐根尋找退場點,回傳 {exit_idx, exit_price, exit_reason, exit_time}。

        退場優先序(每一根):
        1. 停損/停利(盤中觸及;跳空則以 open 成交;同根雙觸以停損為準=保守)。
           啟用移動停損時,停損價改用「進場後最佳價(到前一根為止)∓ mult*ATR」與初始硬停損
           取較嚴格者;最佳價只用到前一根,避免當根自身高點把停損抬高後又被同根低點打到(無前視)。
        2. 反向訊號 / 時間停損 → 於「下一根 open」平倉。
        3. 資料結束 → 以最後一根 close 平倉(標記 end_of_data)。
        """
        direction = position["direction"]
        entry_price = position["entry_price"]
        sl_price, tp_price = self._sl_tp_prices(direction, entry_price)
        trailing = self.trailing_atr_mult is not None and atr_vals is not None
        best_price = entry_price  # 進場後的最佳價(long=最高、short=最低),僅累計到前一根

        k = entry_idx
        while k < n:
            # --- 1. 停損/停利(盤中 high/low) ---
            effective_sl = sl_price
            if trailing:
                effective_sl = self._trailing_stop(
                    direction, best_price, atr_vals[k], sl_price
                )
            hit = self._check_sl_tp(
                direction, opens[k], highs[k], lows[k], effective_sl, tp_price
            )
            if hit is not None:
                reason, price = hit
                return {
                    "exit_idx": k,
                    "exit_price": price,
                    "exit_reason": reason,
                    "exit_time": times.iloc[k],
                }

            # --- 2. 反向訊號 / 時間停損 → 下一根 open 平倉 ---
            held = k - entry_idx
            opposite = self.allow_opposite_exit and int(signals.iloc[k]) == -direction
            time_stop = self.max_hold_bars is not None and held >= self.max_hold_bars
            if (opposite or time_stop) and k + 1 < n:
                reason = EXIT_SIGNAL if opposite else EXIT_TIME_STOP
                return {
                    "exit_idx": k + 1,
                    "exit_price": opens[k + 1],
                    "exit_reason": reason,
                    "exit_time": times.iloc[k + 1],
                }

            # 更新「進場後最佳價」:留到下一根才生效,避免當根自身高/低點影響當根停損判斷
            if direction == LONG:
                best_price = max(best_price, highs[k])
            else:
                best_price = min(best_price, lows[k])
            k += 1

        # --- 3. 資料結束,以最後一根 close 平倉 ---
        last = n - 1
        return {
            "exit_idx": last,
            "exit_price": float(closes[last]),
            "exit_reason": EXIT_END_OF_DATA,
            "exit_time": times.iloc[last],
        }

    def _sl_tp_prices(self, direction, entry_price):
        """由方向與進場價算出停損/停利絕對價格(None 表示未設)。"""
        sl_price = None
        tp_price = None
        if self.stop_loss_pct is not None:
            sl_price = (
                entry_price * (1 - self.stop_loss_pct)
                if direction == LONG
                else entry_price * (1 + self.stop_loss_pct)
            )
        if self.take_profit_pct is not None:
            tp_price = (
                entry_price * (1 + self.take_profit_pct)
                if direction == LONG
                else entry_price * (1 - self.take_profit_pct)
            )
        return sl_price, tp_price

    def _position_notional(self, equity: float, scale: float = 1.0) -> float:
        """計算本筆名目部位。

        若設 risk_per_trade_pct 且有初始停損距離,依「停損被打到時虧損=risk*權益」反推名目;
        否則用 equity*position_size_ratio*leverage。再乘上 scale(部位倍率,如波動度目標化),
        最後一律以 equity*leverage 為硬上限(槓桿不可超過)。
        """
        if self.risk_per_trade_pct is not None and self.stop_loss_pct:
            notional = (self.risk_per_trade_pct * equity) / self.stop_loss_pct
        else:
            notional = equity * self.position_size_ratio * self.leverage
        return min(notional * scale, equity * self.leverage)

    def _trailing_stop(self, direction, best_price, atr_value, initial_sl):
        """算出當根的動態停損價(移動停損)。

        long:best_price - mult*ATR;short:best_price + mult*ATR。
        與初始硬停損 initial_sl 取較嚴格者(long 取較高、short 取較低)。
        ATR 為 NaN(warmup)時退回初始硬停損。
        """
        if atr_value != atr_value:  # NaN 檢查
            return initial_sl
        offset = self.trailing_atr_mult * atr_value
        if direction == LONG:
            trail = best_price - offset
            return trail if initial_sl is None else max(initial_sl, trail)
        trail = best_price + offset
        return trail if initial_sl is None else min(initial_sl, trail)

    def _check_sl_tp(self, direction, bar_open, bar_high, bar_low, sl_price, tp_price):
        """判斷這一根是否觸及停損/停利,回傳 (reason, fill_price) 或 None。

        跳空:開盤已越過停損 → 以 open 成交(更差);開盤已越過停利 → 以 open 成交(更好)。
        同根同時觸及停損與停利 → 回停損(保守,假設先打到停損)。
        """
        if direction == LONG:
            # 停損在下方、停利在上方
            sl_gap = sl_price is not None and bar_open <= sl_price
            sl_intrabar = sl_price is not None and bar_low <= sl_price
            tp_gap = tp_price is not None and bar_open >= tp_price
            tp_intrabar = tp_price is not None and bar_high >= tp_price
            # 保守:先判停損
            if sl_gap:
                return (EXIT_STOP_LOSS, bar_open)
            if tp_gap:
                return (EXIT_TAKE_PROFIT, bar_open)
            if sl_intrabar:
                return (EXIT_STOP_LOSS, sl_price)
            if tp_intrabar:
                return (EXIT_TAKE_PROFIT, tp_price)
        else:  # SHORT:停損在上方、停利在下方
            sl_gap = sl_price is not None and bar_open >= sl_price
            sl_intrabar = sl_price is not None and bar_high >= sl_price
            tp_gap = tp_price is not None and bar_open <= tp_price
            tp_intrabar = tp_price is not None and bar_low <= tp_price
            if sl_gap:
                return (EXIT_STOP_LOSS, bar_open)
            if tp_gap:
                return (EXIT_TAKE_PROFIT, bar_open)
            if sl_intrabar:
                return (EXIT_STOP_LOSS, sl_price)
            if tp_intrabar:
                return (EXIT_TAKE_PROFIT, tp_price)
        return None

    def _close_trade(self, position, exit_info, equity) -> Trade:
        """結算一筆交易,回傳 Trade 並更新權益。"""
        direction = position["direction"]
        entry_price = position["entry_price"]
        notional = position["notional"]
        exit_price = exit_info["exit_price"]

        qty = notional / entry_price
        gross_return_pct = direction * (exit_price - entry_price) / entry_price
        gross_pnl = gross_return_pct * notional
        entry_cost = notional * self._cost_rate
        exit_cost = (qty * exit_price) * self._cost_rate
        cost = entry_cost + exit_cost
        net_pnl = gross_pnl - cost
        equity_before = position["equity_before"]
        net_return_pct = net_pnl / equity_before if equity_before > 0 else 0.0
        equity_after = equity + net_pnl

        return Trade(
            direction=direction,
            entry_time=position["entry_time"],
            entry_price=entry_price,
            entry_idx=position["entry_idx"],
            exit_time=exit_info["exit_time"],
            exit_price=exit_price,
            exit_idx=exit_info["exit_idx"],
            exit_reason=exit_info["exit_reason"],
            notional=notional,
            gross_return_pct=gross_return_pct,
            cost=cost,
            net_pnl=net_pnl,
            net_return_pct=net_return_pct,
            hold_bars=exit_info["exit_idx"] - position["entry_idx"],
            equity_after=equity_after,
        )
