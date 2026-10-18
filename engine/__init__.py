"""誠實回測引擎 (honest backtest engine)。

這個套件是整個交易系統重建後的「可信地基」。設計原則:
1. 絕不偷看未來:訊號在第 i 根收盤後才算得出,成交一律在第 i+1 根 open。
2. 成本必算:手續費 + 滑價,兩邊都收。
3. 停損停利用盤中 high/low 判定,不是只看 close;同一根同時觸及時以較不利的一邊成交(保守)。
4. 一定回報最大回撤 (max drawdown) 與每筆期望值,不只看勝率。

舊的 trading/backtest.py 因為當根 close 成交、停損停利填精確價、無滑價,會系統性高估績效;
本引擎用來取代它並作為唯一可信的績效來源。
"""

from engine.backtester import BacktestResult, EventBacktester, Trade
from engine.metrics import PerformanceMetrics, compute_metrics

__all__ = [
    "EventBacktester",
    "BacktestResult",
    "Trade",
    "PerformanceMetrics",
    "compute_metrics",
]
