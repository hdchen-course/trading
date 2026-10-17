#!/usr/bin/env python3
"""績效指標計算。

輸入「每筆交易的帳戶報酬率」與「權益曲線」,輸出一組誠實的績效指標。
重點:不只看勝率,還看期望值、賺賠比、獲利因子、最大回撤。這些才能判斷一個
訊號「扣掉成本後是否真的有邊際」。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Sequence


@dataclass
class PerformanceMetrics:
    """一次回測的績效總結。所有 *_pct 欄位皆為「帳戶權益」口徑的比率(非價格變動)。"""

    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float  # 勝率 = 勝場 / 總場

    initial_equity: float
    final_equity: float
    total_return: float  # 總報酬率 = final/initial - 1

    expectancy_pct: float  # 每筆交易的平均帳戶報酬率(含成本)→ >0 才有邊際
    avg_win_pct: float  # 勝場平均報酬率
    avg_loss_pct: float  # 敗場平均報酬率(負值)
    payoff_ratio: float  # 賺賠比 = 平均賺 / |平均賠|
    profit_factor: float  # 獲利因子 = 總獲利金額 / 總虧損金額

    max_drawdown: float  # 最大回撤(正值,例如 0.2 代表 -20%)
    sharpe_per_trade: float  # 每筆報酬的夏普(mean/std),無風險利率視為 0

    total_cost: float  # 累計手續費+滑價(金額)
    avg_hold_bars: float  # 平均持倉根數

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """人看的一行摘要(繁中)。"""
        return (
            f"交易{self.n_trades}筆 | 勝率{self.win_rate:.1%} | "
            f"總報酬{self.total_return:+.1%} | 期望值{self.expectancy_pct:+.3%}/筆 | "
            f"賺賠比{self.payoff_ratio:.2f} | 獲利因子{self.profit_factor:.2f} | "
            f"最大回撤{self.max_drawdown:.1%} | Sharpe{self.sharpe_per_trade:.2f}"
        )


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """計算權益曲線的最大回撤(回傳正值比率)。

    回撤 = (峰值 - 當前) / 峰值,取整段最大者。空或單點回傳 0。
    """
    peak = -math.inf
    worst = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        if peak > 0:
            drawdown = (peak - value) / peak
            if drawdown > worst:
                worst = drawdown
    return worst


def compute_metrics(
    trade_returns_pct: Sequence[float],
    equity_curve: Sequence[float],
    initial_equity: float,
    total_cost: float = 0.0,
    hold_bars: Sequence[int] | None = None,
) -> PerformanceMetrics:
    """由「每筆交易帳戶報酬率」與「權益曲線」算出績效指標。

    參數:
        trade_returns_pct: 每筆交易的帳戶報酬率(含成本),例如 0.012 = +1.2%。
        equity_curve: 權益序列,第一點應為 initial_equity,其後每筆交易後的權益。
        initial_equity: 初始本金。
        total_cost: 累計成本金額(手續費+滑價),僅供報表,不影響其他計算。
        hold_bars: 每筆交易的持倉根數(可選)。
    """
    returns: List[float] = list(trade_returns_pct)
    n_trades = len(returns)

    final_equity = equity_curve[-1] if len(equity_curve) > 0 else initial_equity
    total_return = (final_equity / initial_equity - 1.0) if initial_equity > 0 else 0.0

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = (n_wins / n_trades) if n_trades > 0 else 0.0

    expectancy_pct = (sum(returns) / n_trades) if n_trades > 0 else 0.0
    avg_win_pct = (sum(wins) / n_wins) if n_wins > 0 else 0.0
    avg_loss_pct = (sum(losses) / n_losses) if n_losses > 0 else 0.0
    payoff_ratio = (avg_win_pct / abs(avg_loss_pct)) if avg_loss_pct != 0 else math.inf

    # 獲利因子用「金額」口徑:以每筆報酬率近似為等權重,等同報酬率總和之比
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else math.inf

    mdd = max_drawdown(equity_curve)

    if n_trades > 1:
        mean_r = expectancy_pct
        variance = sum((r - mean_r) ** 2 for r in returns) / (n_trades - 1)
        std_r = math.sqrt(variance)
        sharpe = (mean_r / std_r) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    avg_hold = (sum(hold_bars) / len(hold_bars)) if hold_bars else 0.0

    return PerformanceMetrics(
        n_trades=n_trades,
        n_wins=n_wins,
        n_losses=n_losses,
        win_rate=win_rate,
        initial_equity=initial_equity,
        final_equity=final_equity,
        total_return=total_return,
        expectancy_pct=expectancy_pct,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        payoff_ratio=payoff_ratio,
        profit_factor=profit_factor,
        max_drawdown=mdd,
        sharpe_per_trade=sharpe,
        total_cost=total_cost,
        avg_hold_bars=avg_hold,
    )
