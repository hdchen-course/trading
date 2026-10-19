"""可解釋的訊號產生器集合。

每個產生器接受 OHLCV DataFrame,回傳與其等長、同索引的 pd.Series,值 ∈ {-1,0,+1}
(+1=做多進場、-1=做空進場、0=無),且只使用「第 i 根(含)以前」的資料(無前視)。
訊號代表「該根收盤後」的判斷,實際成交由 EventBacktester 安排在下一根 open。
"""

from signals.library import (
    bollinger_revert,
    donchian_breakout,
    ema_cross,
    ema_ribbon_trend,
    regime_filter,
    with_trend_filter,
)

__all__ = [
    "donchian_breakout",
    "ema_cross",
    "ema_ribbon_trend",
    "regime_filter",
    "bollinger_revert",
    "with_trend_filter",
]
